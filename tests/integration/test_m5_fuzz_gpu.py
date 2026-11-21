"""M5 GPU fuzz: randomized layout fixtures -> adapter descriptors ->
golden byte-offset oracle -> real transfers (research.md §10 fuzzing).

For every randomized geometry (all six layout families x kernel-split
ratios x shared/per-layer storages x dim-0 padding x dtypes):

1. build a REAL CUDA buffer whose per-layer views carry the certified
   strides of that family (the views are what `register_kv_caches` sees);
2. run `VllmLayoutAdapter.describe` on duck-typed engine metadata;
3. assert `placement.gpu_ops(b)` equals the golden (ptr, size) fragments
   computed independently from the buffer construction;
4. submit those ops through the real transfer engine (d2h) and assert the
   landing bytes identify the exact source offsets — a wrong stride or
   base pointer cannot pass this.

Block-outermost layouts are exercised at ratio 1 only: with kernel-block
splits their manager pages are not contiguous runs and the adapter now
rejects registration (the silent-corruption path this fuzz closed).

Run on the remote host:
    .venv/bin/pytest tests/integration/test_m5_fuzz_gpu.py -v
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.transfer.engine import PageCopyOp, RegionTransferEngine

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

LAYOUTS = ["LBNHC", "LBHNC", "LHBNC", "BLNHC", "BLHNC", "BHLNC"]


class _QuantMode:
    name = "NONE"


@dataclass
class Spec:
    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: object
    page_size_bytes: int

    kv_quant_mode = _QuantMode()

    @property
    def prefix_cacheable(self) -> bool:
        return True


@dataclass
class FakeTensor:
    layers: list[str]
    size: int
    layer_stride: int
    block_stride: int
    offset: int = 0


@dataclass
class FakeGroupSpec:
    layer_names: list[str]
    kv_cache_spec: Spec


@dataclass
class FakeConfig:
    num_blocks: int
    kv_cache_tensors: list[FakeTensor]
    kv_cache_groups: list
    kv_cache_layout: str


class Fixture:
    """One randomized certified engine buffer + its golden addressing.

    Buffer arrangement per family (elements of the view dtype):

    - LBNHC/LBHNC (layer-compact): layer ``l`` occupies ``rows*row``
      elements at ``l * (rows*row*elem + pad)`` bytes; blocks contiguous
      within a layer (manager stride = page).
    - LHBNC: layer ``l`` occupies ``H*rows*tokens*c``; head ``h`` of block
      ``b`` is one contiguous ``page//H`` run.
    - BLNHC/BLHNC: one block-outermost region; layer ``l`` of block ``b``
      at ``(b*L + l)*page``.
    - BHLNC: ``(b*H + h)*L*page + l*page`` per (block, head, layer).
    """

    def __init__(self, rng: random.Random):
        self.layout = rng.choice(LAYOUTS)
        self.L = rng.choice([2, 3])
        self.B = rng.choice([4, 6])
        self.H = rng.choice([1, 2])
        self.N = rng.choice([16, 64])
        block_outermost = self.layout.startswith("B")
        self.ratio = 1 if block_outermost else rng.choice([1, 2])
        self.shared = rng.random() < 0.5 or block_outermost
        # dim-0 padding: slack after each layer region (per-layer form only)
        self.pad = (
            rng.choice([0, 256, 1024]) if (not self.shared and self.ratio == 1) else 0
        )
        self.elem = rng.choice([1, 2])
        self.dtype = torch.uint8 if self.elem == 1 else torch.float16
        self.head_size = rng.choice([64, 128])
        self.content = (self.head_size + self.head_size) * self.elem
        self.page = self.N * self.H * self.content
        self.spec = Spec(
            block_size=self.N,
            num_kv_heads=self.H,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=self.dtype,
            page_size_bytes=self.page,
        )
        self.tokens = self.N // self.ratio
        self.rows = self.B * self.ratio
        self.c = self.content // self.elem
        self.row = self.tokens * self.H * self.c  # kernel row, elements

        if self.layout in ("LBNHC", "LBHNC"):
            layer_bytes = self.rows * self.row * self.elem + self.pad
            total = self.L * layer_bytes
            self._layer_base_bytes = [l * layer_bytes for l in range(self.L)]
            self._md_layer_stride = layer_bytes
            self._md_block_stride = self.page + (self.pad if not self.shared else 0)
        elif self.layout == "LHBNC":
            layer_bytes = self.H * self.rows * self.tokens * self.c * self.elem
            total = self.L * layer_bytes
            self._layer_base_bytes = [l * layer_bytes for l in range(self.L)]
            self._md_layer_stride = layer_bytes
            self._md_block_stride = self.page // self.H
        else:  # BLNHC / BLHNC / BHLNC: one shared block-outermost region
            layer_bytes = self.row * self.elem  # one (block, layer) run
            total = (
                self.B * self.H * self.L * self.page
                if self.layout == "BHLNC"
                else self.B * self.L * self.page
            )
            self._layer_base_bytes = [l * layer_bytes for l in range(self.L)]
            self._md_layer_stride = self.page
            self._md_block_stride = (
                self.H * self.L * self.page if self.layout == "BHLNC"
                else self.L * self.page
            )

        self._flat = torch.zeros((total + 15) // 16 * 16, dtype=torch.uint8, device="cuda")
        # Distinctive content: every byte identifies its absolute offset.
        self._flat.copy_(
            torch.arange(self._flat.numel(), dtype=torch.int64, device="cuda") % 251
        )

        shape = (self.rows, self.H, self.tokens, self.c)
        strides = self._view_strides()
        typed = self._flat if self.elem == 1 else self._flat.view(torch.float16)
        self.views: dict[str, torch.Tensor] = {}
        for li, name in enumerate(self.layer_names):
            self.views[name] = typed.as_strided(
                shape, strides, self._layer_base_bytes[li] // self.elem
            )

    # ---- construction-derived facts (the golden model) -------------------

    @property
    def layer_names(self) -> list[str]:
        return [f"model.layers.{i}.self_attn.attn" for i in range(self.L)]

    def _view_strides(self) -> tuple[int, int, int, int]:
        L, row, tokens, c, H = self.L, self.row, self.tokens, self.c, self.H
        if self.layout == "LBNHC":
            return (row, c, H * c, 1)
        if self.layout == "LBHNC":
            return (row, tokens * c, c, 1)
        if self.layout == "LHBNC":
            return (tokens * c, self.rows * tokens * c, c, 1)
        if self.layout == "BLNHC":
            return (L * row, c, H * c, 1)
        if self.layout == "BLHNC":
            return (L * row, tokens * c, c, 1)
        if self.layout == "BHLNC":
            return (H * L * row, L * row, c, 1)
        raise AssertionError(self.layout)

    def golden_ops(self, block_id: int) -> list[tuple[int, int]]:
        """(byte offset in the flat storage, size) per store fragment."""
        out: list[tuple[int, int]] = []
        page, L, H = self.page, self.L, self.H
        for li in range(L):
            base = self._layer_base_bytes[li]
            if self.layout in ("LBNHC", "LBHNC"):
                out.append((base + block_id * page, page))
            elif self.layout == "LHBNC":
                head_step = self.rows * self.tokens * self.c * self.elem
                for h in range(H):
                    out.append((base + h * head_step + block_id * (page // H), page // H))
            elif self.layout in ("BLNHC", "BLHNC"):
                out.append((base + block_id * L * page, page))
            else:  # BHLNC
                for h in range(H):
                    out.append(
                        (base + h * L * page + block_id * H * L * page, page)
                    )
        return out


@needs_cuda
class TestAdapterFuzzGoldenOffsets:
    def _run_case(self, rng: random.Random) -> None:
        fx = Fixture(rng)
        tensors = (
            [
                FakeTensor(
                    layers=list(fx.layer_names),
                    size=fx._flat.numel(),
                    layer_stride=fx._md_layer_stride,
                    block_stride=fx._md_block_stride,
                )
            ]
            if fx.shared
            else [
                FakeTensor(
                    layers=[name],
                    size=fx._flat.numel(),
                    layer_stride=fx._md_layer_stride,
                    block_stride=fx._md_block_stride,
                )
                for name in fx.layer_names
            ]
        )
        cfg = FakeConfig(
            num_blocks=fx.B,
            kv_cache_tensors=tensors,
            kv_cache_groups=[FakeGroupSpec(list(fx.layer_names), fx.spec)],
            kv_cache_layout=fx.layout,
        )
        adapter = VllmLayoutAdapter(device_index=0)
        (desc,) = adapter.describe(fx.views, cfg)

        # (1) fingerprint fragments tile the page payload; geometry carries
        # the constructed strides verbatim.
        assert sum(desc.fingerprint.fragments) == fx.page
        assert desc.fingerprint.page_bytes == (fx.page, fx.page)
        assert desc.geometry.block_stride == fx._md_block_stride

        # (2) placement addressing equals the golden model, every block.
        flat_base = fx._flat.data_ptr()
        placement = adapter.placements[0]
        for b in range(fx.B):
            got = [(ptr - flat_base, size) for ptr, size in placement.gpu_ops(b)]
            want = fx.golden_ops(b)
            assert got == want, f"{fx.layout} block {b}: placement {got} != golden {want}"

        # (3) a real d2h transfer of the planned ops lands the golden bytes.
        engine = RegionTransferEngine(device_index=0)
        ops: list[PageCopyOp] = []
        spans: list[tuple[int, int, int]] = []  # (host off, gpu off, size)
        host = torch.zeros(fx.page * fx.L * fx.B, dtype=torch.uint8, pin_memory=True)
        cursor = 0
        for b in range(fx.B):
            for ptr, size in placement.gpu_ops(b):
                ops.append(
                    PageCopyOp(src_ptr=ptr, dst_ptr=host.data_ptr() + cursor, nbytes=size)
                )
                spans.append((cursor, ptr - flat_base, size))
                cursor += size
        engine.submit("d2h", ops, fence=False).wait()
        flat_cpu = fx._flat.cpu()
        for host_off, src_off, size in spans:
            got = host[host_off : host_off + size]
            want = flat_cpu[src_off : src_off + size]
            assert torch.equal(got, want), (
                f"{fx.layout}: bytes at host {host_off} (gpu offset {src_off}) "
                f"do not identify their source offset"
            )

    def test_fuzz_matrix_golden_offsets(self):
        for seed in range(36):
            self._run_case(random.Random(0xF00D + seed))

    def test_fuzz_random_geometries(self):
        rng = random.Random(2026)
        for _ in range(24):
            self._run_case(rng)

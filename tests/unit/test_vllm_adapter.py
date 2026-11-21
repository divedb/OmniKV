"""Unit tests for the vLLM layout adapter, using duck-typed mocks.

Torch-free so they run on any host. The mocks mirror the *installed*
v0.34.0 release API (`KVCacheGroupSpec` with singular `kv_cache_spec`)
and main's post-#61452 form (`kv_cache_specs` dict), with the exact
geometry measured on the remote host (docs/TESTLOG.md 2026-11-20):
  Qwen2.5-1.5B: 28 layers, LBNHC, block 64, 2 KV heads, head 128, bf16,
  page 65536 B, layer_stride 235864064, block_stride 65536, num_blocks 3488.

M2: every layout family is exercised with certified strides, including
head-fragmented LHBNC/BHLNC, block-outermost layouts, dim-0 padding and
stride-mismatch rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.errors import RegistrationError

PAGE = 65536
NUM_LAYERS = 28
LAYER_STRIDE = 235_864_064
NUM_BLOCKS = 3488
LAYER_NAMES = [f"model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)]

N, H, C = 64, 2, 512  # tokens, kv heads, content bytes per (token, head)


class FakeDtype:
    """Duck-typed torch dtype (bf16, 2 bytes)."""

    itemsize = 2

    def __str__(self) -> str:
        return "bfloat16"

    def __repr__(self) -> str:  # fingerprints repr the tuple; keep it stable
        return "bfloat16"


class FakeDevice:
    index = 0


def _strides(layout: str) -> tuple[int, int, int, int]:
    """Certified per-layer view strides for logical (B, H, N, C)."""
    page = N * H * C
    if layout == "LBNHC":
        return (page, C, H * C, 1)
    if layout == "LBHNC":
        return (page, N * C, C, 1)
    if layout == "LHBNC":
        return (N * C, NUM_BLOCKS * N * C, C, 1)
    if layout == "BLNHC":
        return (NUM_LAYERS * page, C, H * C, 1)
    if layout == "BLHNC":
        return (NUM_LAYERS * page, N * C, C, 1)
    if layout == "BHLNC":
        return (H * NUM_LAYERS * page, NUM_LAYERS * page, C, 1)
    raise AssertionError(layout)


class FakeByteDtype:
    """Duck-typed dtype for views whose fakes already express strides in
    bytes (itemsize 1 keeps the element->byte scaling an identity)."""

    itemsize = 1

    def __str__(self) -> str:
        return "uint8"


class FakeCachesTensor:
    """What the adapter reads off a registered per-layer KV view."""

    device = FakeDevice()
    dtype = FakeByteDtype()

    def __init__(self, ptr: int, layout: str = "LBNHC"):
        self._ptr = ptr
        self._layout = layout
        self._shape: tuple[int, ...] | None = None
        self._strides: tuple[int, ...] | None = None

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        return (NUM_BLOCKS, H, N, C)

    def stride(self):
        if self._strides is not None:
            return self._strides
        return _strides(self._layout)

    def data_ptr(self) -> int:
        return self._ptr


class _QuantMode:
    def __init__(self, name):
        self.name = name


class FullAttentionSpec:
    """FullAttentionSpec-shaped mock (v0.34.0 release form)."""

    def __init__(self, block_size=64, num_kv_heads=H, head_size=128):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size
        self.dtype = FakeDtype()
        self.kv_quant_mode = _QuantMode("NONE")
        self.page_size_bytes = block_size * num_kv_heads * 2 * head_size * 2

    @property
    def prefix_cacheable(self) -> bool:
        return True


@dataclass
class FakeGroupSpec:
    layer_names: list[str]
    kv_cache_spec: FullAttentionSpec


@dataclass
class FakeGroupMain:
    """Main-branch (post-#61452) form."""

    kv_cache_specs: dict[str, FullAttentionSpec]


@dataclass
class FakeTensor:
    layers: list[str]
    size: int
    layer_stride: int
    block_stride: int
    offset: int = 0


@dataclass
class FakeConfig:
    num_blocks: int
    kv_cache_tensors: list[FakeTensor]
    kv_cache_groups: list
    kv_cache_layout: str = "LBNHC"
    hash_block_size: int = 64


def make_release_config(layout: str = "LBNHC", **over) -> FakeConfig:
    page = N * H * C
    if layout.startswith("L"):
        tensor = FakeTensor(
            layers=list(LAYER_NAMES),
            size=NUM_BLOCKS * page * NUM_LAYERS,
            layer_stride=LAYER_STRIDE if layout != "LHBNC" else NUM_BLOCKS * N * C,
            block_stride=page if layout != "LHBNC" else N * C,
        )
    else:
        tensor = FakeTensor(
            layers=list(LAYER_NAMES),
            size=NUM_BLOCKS * page * NUM_LAYERS,
            layer_stride=page,
            block_stride=page * NUM_LAYERS,
        )
    cfg = FakeConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[FakeGroupSpec(list(LAYER_NAMES), FullAttentionSpec())],
        kv_cache_layout=layout,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_kv_caches(layout: str = "LBNHC") -> dict[str, FakeCachesTensor]:
    return {
        name: FakeCachesTensor(0x1000 + i * PAGE, layout)
        for i, name in enumerate(LAYER_NAMES)
    }


class TestDescribe:
    def test_release_form(self):
        adapter = VllmLayoutAdapter()
        (desc,) = adapter.describe(make_kv_caches(), make_release_config())
        assert desc.group_id == 0
        assert desc.semantics.kind.value == "paged_kv"
        assert desc.semantics.block_size == 64
        assert desc.fingerprint.layout == "LBNHC"
        assert desc.fingerprint.dtype == ("bfloat16",)
        assert desc.fingerprint.page_bytes == (PAGE, PAGE)
        assert desc.fingerprint.fragments == (PAGE,)
        geo = desc.geometry
        assert geo.layer_stride == LAYER_STRIDE  # strides verbatim, not derived
        assert geo.num_blocks == NUM_BLOCKS
        assert len(adapter.last_layer_ptrs) == NUM_LAYERS
        assert adapter.last_block_stride == PAGE

    def test_main_form(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [
            FakeGroupMain({n: FullAttentionSpec() for n in LAYER_NAMES})
        ]
        (desc,) = VllmLayoutAdapter().describe(make_kv_caches(), cfg)
        assert desc.geometry.num_layers == NUM_LAYERS

    def test_main_form_all_layouts(self):
        """M5 version-shim sweep: the post-#61452 group form must describe
        every layout family with the same geometry as the release form."""
        for layout in ("LBNHC", "LBHNC", "LHBNC", "BLNHC", "BLHNC", "BHLNC"):
            release_cfg = make_release_config(layout)
            main_cfg = make_release_config(layout)
            main_cfg.kv_cache_groups = [
                FakeGroupMain({n: FullAttentionSpec() for n in LAYER_NAMES})
            ]
            (rel,) = VllmLayoutAdapter().describe(make_kv_caches(layout), release_cfg)
            (main,) = VllmLayoutAdapter().describe(make_kv_caches(layout), main_cfg)
            assert rel.fingerprint == main.fingerprint, layout
            assert rel.geometry == main.geometry, layout

    def test_main_form_multiple_groups(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [
            FakeGroupMain({n: FullAttentionSpec() for n in LAYER_NAMES[:14]}),
            FakeGroupMain({n: FullAttentionSpec() for n in LAYER_NAMES[14:]}),
        ]
        descs = VllmLayoutAdapter().describe(make_kv_caches(), cfg)
        assert [d.group_id for d in descs] == [0, 1]

    def test_main_form_mixed_specs_rejected(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [
            FakeGroupMain(
                {
                    n: (FullAttentionSpec() if i % 2 else FullAttentionSpec(head_size=64))
                    for i, n in enumerate(LAYER_NAMES)
                }
            )
        ]
        with pytest.raises(RegistrationError, match="mixed per-layer formats"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_unknown_group_form_rejected(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [SimpleNamespace()]
        with pytest.raises(RegistrationError, match="unrecognized KV cache group"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_multiple_groups(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [
            FakeGroupSpec(LAYER_NAMES[:14], FullAttentionSpec()),
            FakeGroupSpec(LAYER_NAMES[14:], FullAttentionSpec()),
        ]
        adapter = VllmLayoutAdapter()
        descs = adapter.describe(make_kv_caches(), cfg)
        assert [d.group_id for d in descs] == [0, 1]
        assert len(adapter.placements) == 2

    def test_rejects_non_full_attention_spec(self):
        mamba_cls = type("MambaSpec", (FullAttentionSpec,), {})
        cfg = make_release_config()
        cfg.kv_cache_groups = [FakeGroupSpec(LAYER_NAMES, mamba_cls())]
        with pytest.raises(RegistrationError, match="PAGED_KV"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_mixed_per_layer_specs(self):
        cfg = make_release_config()
        spec_a = FullAttentionSpec()
        spec_b = FullAttentionSpec(head_size=64)  # genuinely different format
        cfg.kv_cache_groups = [
            FakeGroupMain(
                {n: (spec_a if i % 2 else spec_b) for i, n in enumerate(LAYER_NAMES)}
            )
        ]
        with pytest.raises(RegistrationError, match="mixed per-layer formats"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_describes_per_tensor_fp8(self):
        spec = FullAttentionSpec()
        spec.kv_quant_mode = _QuantMode("FP8_PER_TENSOR")
        cfg = make_release_config()
        cfg.kv_cache_groups = [FakeGroupSpec(LAYER_NAMES, spec)]
        (desc,) = VllmLayoutAdapter().describe(make_kv_caches(), cfg)
        assert desc.fingerprint.quant == "FP8_PER_TENSOR"

    def test_rejects_diff_kv_heads(self):
        spec = FullAttentionSpec(head_size=128)
        spec.head_size_v = 64
        cfg = make_release_config()
        cfg.kv_cache_groups = [FakeGroupSpec(LAYER_NAMES, spec)]
        with pytest.raises(RegistrationError, match="diff-KV"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_unresolved_layout(self):
        with pytest.raises(RegistrationError, match="not resolved"):
            VllmLayoutAdapter().describe(
                make_kv_caches(), make_release_config(kv_cache_layout=None)
            )

    def test_stride_certification_rejects_mismatch(self):
        caches = make_kv_caches("LBNHC")
        # Corrupt one layer's certified stride: NHD strides flipped.
        caches[LAYER_NAMES[3]]._layout = "LBHNC"
        with pytest.raises(RegistrationError, match="do not match the certified"):
            VllmLayoutAdapter().describe(caches, make_release_config("LBNHC"))

    def test_block_outermost_layouts(self):
        for layout in ("BLNHC", "BLHNC"):
            (desc,) = VllmLayoutAdapter().describe(
                make_kv_caches(layout), make_release_config(layout)
            )
            assert desc.fingerprint.layout == layout
            assert desc.fingerprint.fragments == (PAGE,)
            assert desc.geometry.layer_stride == PAGE
            assert desc.geometry.block_stride == PAGE * NUM_LAYERS


class TestHeadFragmentedLayouts:
    @pytest.mark.parametrize("layout", ["LHBNC", "BHLNC"])
    def test_fragments_and_placement(self, layout):
        adapter = VllmLayoutAdapter()
        (desc,) = adapter.describe(
            make_kv_caches(layout), make_release_config(layout)
        )
        assert desc.fingerprint.fragments == (N * C, N * C)
        assert desc.fingerprint.page_bytes == (PAGE, PAGE)
        placement = adapter.last_placement
        assert placement is not None
        assert placement.store_page_bytes == NUM_LAYERS * PAGE  # all layers
        # Per layer: one fragment per head; block 7 addresses follow the
        # certified strides.
        ops = placement.gpu_ops(block_id=7)
        assert len(ops) == NUM_LAYERS * H
        head_stride = _strides(layout)[1]
        block_frag_stride = _strides(layout)[0]
        base = 0x1000  # layer 0 ptr
        assert ops[0] == (base + 7 * block_frag_stride, N * C)
        assert ops[1] == (base + head_stride + 7 * block_frag_stride, N * C)

    def test_padded_page_rejected_for_fragmented(self):
        cfg = make_release_config("LHBNC")
        spec = cfg.kv_cache_groups[0].kv_cache_spec
        # A page padded to a whole multiple (2x) passes the divisibility
        # check but is still padded: fragmented layouts reject it.
        spec.page_size_bytes = PAGE * 2
        with pytest.raises(RegistrationError, match="unpadded page"):
            VllmLayoutAdapter().describe(make_kv_caches("LHBNC"), cfg)


class TestKernelBlockSplit:
    def test_split_view_certified_and_addressed(self):
        # The measured real-engine case: block 64, kernel blocks of 32
        # (ratio 2), strides exactly half the manager-granularity values.
        caches = {}
        kern = N // 2
        for i, name in enumerate(LAYER_NAMES):
            view = FakeCachesTensor(0x1000 + i * PAGE)
            view._shape = (NUM_BLOCKS * 2, H, kern, C)
            view._strides = (kern * H * C, C, H * C, 1)  # (32768, 512, 1024, 1)
            caches[name] = view
        adapter = VllmLayoutAdapter()
        (desc,) = adapter.describe(caches, make_release_config("LBNHC"))
        # Manager-block addressing: stride doubles back to the full page.
        ops = adapter.last_placement.gpu_ops(block_id=5)
        assert len(ops) == NUM_LAYERS
        assert ops[0] == (0x1000 + 5 * PAGE, PAGE)
        assert desc.fingerprint.page_bytes == (PAGE, PAGE)

    def test_inconsistent_rows_rejected(self):
        caches = {}
        kern = N // 2
        for i, name in enumerate(LAYER_NAMES):
            view = FakeCachesTensor(0x1000)
            view._shape = (NUM_BLOCKS * 3, H, kern, C)  # 3 not a valid ratio
            view._strides = (kern * H * C, C, H * C, 1)
            caches[name] = view
        with pytest.raises(RegistrationError, match="not divisible"):
            VllmLayoutAdapter().describe(caches, make_release_config("LBNHC"))

    def test_block_outermost_kernel_split_rejected(self):
        """M5 fail-closed: block-outermost layouts with kernel-block
        splits scatter a manager block's page across kernel rows (stride
        L*row between them), so single-run per-block addressing would
        silently copy interleaved bytes. Registration must refuse."""
        for layout in ("BLNHC", "BLHNC", "BHLNC"):
            caches = {}
            kern = N // 2
            for i, name in enumerate(LAYER_NAMES):
                view = FakeCachesTensor(0x1000 + i * PAGE)
                view._shape = (NUM_BLOCKS * 2, H, kern, C)
                view._strides = _strides_for_split(layout, kern)
                caches[name] = view
            with pytest.raises(RegistrationError, match="kernel-block splits"):
                VllmLayoutAdapter().describe(caches, make_release_config(layout))

    def test_layer_compact_kernel_split_supported(self):
        """The same split on a layer-compact layout stays addressable:
        kernel rows of one manager block are contiguous within a layer."""
        caches = {}
        kern = N // 2
        for i, name in enumerate(LAYER_NAMES):
            view = FakeCachesTensor(0x1000 + i * PAGE)
            view._shape = (NUM_BLOCKS * 2, H, kern, C)
            view._strides = _strides_for_split("LBHNC", kern)
            caches[name] = view
        adapter = VllmLayoutAdapter()
        (_desc,) = adapter.describe(caches, make_release_config("LBHNC"))
        ops = adapter.placements[0].gpu_ops(block_id=3)
        assert ops[0][1] == PAGE  # whole manager page, one run


def _strides_for_split(layout: str, kern: int) -> tuple[int, int, int, int]:
    """Certified strides for a rank-4 view at kernel granularity
    (rows = num_blocks * ratio, tokens = N // ratio)."""
    row = kern * H * C
    if layout == "BLNHC":
        return (NUM_LAYERS * row, C, H * C, 1)
    if layout == "BLHNC":
        return (NUM_LAYERS * row, kern * C, C, 1)
    if layout == "BHLNC":
        return (H * NUM_LAYERS * row, NUM_LAYERS * row, C, 1)
    if layout == "LBHNC":
        return (row, kern * C, C, 1)
    raise AssertionError(layout)


class TestWorkerForms:
    def test_worker_per_layer_tensor_form(self):
        # The extensible-KV-cache worker projection: one tensor per layer.
        tensors = [
            FakeTensor(
                layers=[name],
                size=NUM_BLOCKS * PAGE,
                layer_stride=PAGE,
                block_stride=PAGE,
            )
            for name in LAYER_NAMES
        ]
        cfg = make_release_config(kv_cache_tensors=tensors)
        adapter = VllmLayoutAdapter()
        (desc,) = adapter.describe(make_kv_caches(), cfg)
        assert desc.geometry.shared_storage is False
        assert desc.geometry.block_stride == PAGE
        assert len(adapter.last_layer_ptrs) == NUM_LAYERS
        assert adapter.last_block_stride == PAGE

    def test_dim0_padding_block_stride(self):
        # Extensible-cache padding: block stride exceeds the page payload.
        tensors = [
            FakeTensor(
                layers=[name],
                size=NUM_BLOCKS * (PAGE + 4096),
                layer_stride=PAGE + 4096,
                block_stride=PAGE + 4096,
            )
            for name in LAYER_NAMES
        ]
        cfg = make_release_config(kv_cache_tensors=tensors)
        adapter = VllmLayoutAdapter()
        (desc,) = adapter.describe(make_kv_caches(), cfg)
        assert desc.geometry.block_stride == PAGE + 4096
        # Fragment size stays the page payload; stride carries the padding.
        ops = adapter.last_placement.gpu_ops(block_id=1)
        assert ops[0][1] == PAGE

    def test_layer_in_multiple_tensors_rejected(self):
        tensors = [
            FakeTensor(
                layers=list(LAYER_NAMES),
                size=NUM_BLOCKS * PAGE * NUM_LAYERS,
                layer_stride=LAYER_STRIDE,
                block_stride=PAGE,
            ),
            FakeTensor(
                layers=[LAYER_NAMES[0]],
                size=NUM_BLOCKS * PAGE,
                layer_stride=PAGE,
                block_stride=PAGE,
            ),
        ]
        cfg = make_release_config(kv_cache_tensors=tensors)
        with pytest.raises(RegistrationError, match="multiple KVCacheTensors"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_layer_without_tensor_rejected(self):
        tensors = [
            FakeTensor(
                layers=[LAYER_NAMES[0]], size=1, layer_stride=PAGE, block_stride=PAGE
            )
        ]
        cfg = make_release_config(kv_cache_tensors=tensors)
        with pytest.raises(RegistrationError, match="have no KVCacheTensor"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_missing_registered_layer_rejected(self):
        caches = make_kv_caches()
        caches.pop(LAYER_NAMES[5])
        with pytest.raises(RegistrationError, match="missing"):
            VllmLayoutAdapter().describe(caches, make_release_config())

    def test_no_matching_tensor_rejected(self):
        tensor = FakeTensor(
            layers=["other.layer"],
            size=PAGE,
            layer_stride=PAGE,
            block_stride=PAGE,
        )
        cfg = make_release_config(kv_cache_tensors=[tensor])
        with pytest.raises(RegistrationError, match="have no KVCacheTensor"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

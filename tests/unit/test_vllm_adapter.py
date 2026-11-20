"""Unit tests for the vLLM layout adapter, using duck-typed mocks.

Torch-free so they run on any host. The mocks mirror the *installed*
v0.34.0 release API (`KVCacheGroupSpec` with singular `kv_cache_spec`)
and main's post-#61452 form (`kv_cache_specs` dict), with the exact
geometry measured on the remote host (docs/TESTLOG.md 2026-11-20):
  Qwen2.5-1.5B: 28 layers, LBNHC, block 64, 2 KV heads, head 128, bf16,
  page 65536 B, layer_stride 235864064, block_stride 65536, num_blocks 3488.
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


class FakeDtype:
    """Duck-typed torch dtype (bf16, 2 bytes)."""

    itemsize = 2

    def __str__(self) -> str:
        return "bfloat16"

    def __repr__(self) -> str:  # fingerprints repr the tuple; keep it stable
        return "bfloat16"


class FakeDevice:
    index = 0


class FakeCachesTensor:
    """What the adapter reads off a registered per-layer KV view."""

    device = FakeDevice()

    def __init__(self, ptr: int):
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


class _QuantMode:
    def __init__(self, name):
        self.name = name


class FullAttentionSpec:
    """FullAttentionSpec-shaped mock (v0.34.0 release form)."""

    def __init__(self, block_size=64, num_kv_heads=2, head_size=128):
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


def make_release_config(**over) -> FakeConfig:
    cfg = FakeConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            FakeTensor(
                layers=list(LAYER_NAMES),
                size=NUM_BLOCKS * PAGE * NUM_LAYERS,
                layer_stride=LAYER_STRIDE,
                block_stride=PAGE,
            )
        ],
        kv_cache_groups=[FakeGroupSpec(list(LAYER_NAMES), FullAttentionSpec())],
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_kv_caches() -> dict[str, FakeCachesTensor]:
    return {name: FakeCachesTensor(0x1000 + i * PAGE) for i, name in enumerate(LAYER_NAMES)}


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

    def test_unknown_group_form_rejected(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [SimpleNamespace()]
        with pytest.raises(RegistrationError, match="unrecognized KV cache group"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_multiple_groups(self):
        cfg = make_release_config()
        cfg.kv_cache_groups = [
            FakeGroupSpec(LAYER_NAMES[:14], FullAttentionSpec()),
            FakeGroupSpec(LAYER_NAMES[14:], FullAttentionSpec()),
        ]
        with pytest.raises(RegistrationError, match="one KV cache group"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_non_full_attention_spec(self):
        mamba_cls = type("MambaSpec", (FullAttentionSpec,), {})
        cfg = make_release_config()
        cfg.kv_cache_groups = [FakeGroupSpec(LAYER_NAMES, mamba_cls())]
        with pytest.raises(RegistrationError, match="PAGED_KV"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_head_fragmented_layout(self):
        with pytest.raises(RegistrationError, match="fragments"):
            VllmLayoutAdapter().describe(
                make_kv_caches(), make_release_config(kv_cache_layout="LHBNC")
            )

    def test_rejects_unresolved_layout(self):
        with pytest.raises(RegistrationError, match="not resolved"):
            VllmLayoutAdapter().describe(
                make_kv_caches(), make_release_config(kv_cache_layout=None)
            )

    def test_rejects_quantized(self):
        spec = FullAttentionSpec()
        spec.kv_quant_mode = _QuantMode("FP8_PER_TENSOR")
        cfg = make_release_config()
        cfg.kv_cache_groups = [FakeGroupSpec(LAYER_NAMES, spec)]
        with pytest.raises(RegistrationError, match="quantized"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_diff_kv_heads(self):
        spec = FullAttentionSpec(head_size=128)
        spec.head_size_v = 64
        cfg = make_release_config()
        cfg.kv_cache_groups = [FakeGroupSpec(LAYER_NAMES, spec)]
        with pytest.raises(RegistrationError, match="diff-KV"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_rejects_overlapping_layer_compact_strides(self):
        tensor = FakeTensor(
            layers=list(LAYER_NAMES),
            size=NUM_BLOCKS * PAGE * NUM_LAYERS,
            layer_stride=LAYER_STRIDE,
            block_stride=PAGE // 2,  # page cannot fit
        )
        cfg = make_release_config(kv_cache_tensors=[tensor])
        with pytest.raises(RegistrationError):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

    def test_block_outermost_layout_accepted(self):
        # BLNHC: block stride spans every layer; layer stride == page.
        tensor = FakeTensor(
            layers=list(LAYER_NAMES),
            size=NUM_BLOCKS * PAGE * NUM_LAYERS,
            layer_stride=PAGE,
            block_stride=PAGE * NUM_LAYERS,
        )
        cfg = make_release_config(
            kv_cache_tensors=[tensor], kv_cache_layout="BLNHC"
        )
        (desc,) = VllmLayoutAdapter().describe(make_kv_caches(), cfg)
        assert desc.fingerprint.layout == "BLNHC"
        assert (
            desc.geometry.layer_block_offset(5, 3)
            == 5 * PAGE + 3 * PAGE * NUM_LAYERS
        )

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
        with pytest.raises(RegistrationError, match="no KVCacheTensor"):
            VllmLayoutAdapter().describe(make_kv_caches(), cfg)

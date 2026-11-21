"""M4 restore retention and byte geometry, including compressed padded MLA."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
from test_vllm_adapter import (
    FakeCachesTensor,
    FakeConfig,
    FakeGroupSpec,
    FakeTensor,
    FullAttentionSpec,
)

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.codec import DEFAULT_REGISTRY, swap_layout
from omnikv.errors import RegistrationError
from omnikv.semantics import required_blocks
from omnikv.types import StateKind, StateSemantics


def fixture(kind="MLAAttentionSpec", *, compressed=1, padded=0, quant="NONE", role=None):
    spec = type(kind, (FullAttentionSpec,), {})(block_size=16, num_kv_heads=1, head_size=8)
    spec.head_size_v = 0 if "MLA" in kind else 8
    spec.tokens_per_state = compressed
    states = int(16 / compressed)
    content = (spec.head_size + spec.head_size_v) * 2
    spec.state_content_size_bytes = content
    spec.page_size_bytes = states * content + padded
    spec.kv_quant_mode = SimpleNamespace(name=quant)
    if "Sliding" in kind:
        spec.sliding_window = 33
        spec.extra_retained_tokens = 3
    group = FakeGroupSpec(["layer"], spec)
    group.role = role
    cfg = FakeConfig(
        4,
        [
            FakeTensor(
                ["layer"], spec.page_size_bytes * 4, spec.page_size_bytes * 4, spec.page_size_bytes
            )
        ],
        [group],
    )
    view = FakeCachesTensor(0x1000)
    view._shape = (4, 1, states, content)
    view._strides = (spec.page_size_bytes, content, content, 1)
    return cfg, {"layer": view}


@pytest.mark.parametrize("compressed", [1, 4, Fraction(1, 2)])
@pytest.mark.parametrize("padding", [0, 24])
def test_latent_geometry(compressed, padding):
    cfg, views = fixture(compressed=compressed, padded=padding)
    adapter = VllmLayoutAdapter()
    (desc,) = adapter.describe(views, cfg)
    assert desc.semantics.kind == StateKind.LATENT_KV
    assert desc.semantics.tokens_per_state == compressed
    assert desc.fingerprint.head_geometry[2] == 0
    size = int(16 / compressed) * 16 + padding
    assert desc.fingerprint.fragments == (size,)
    assert adapter.placements[0].gpu_ops(3) == [(0x1000 + 3 * size, size)]
    assert DEFAULT_REGISTRY.find(desc.fingerprint, swap_layout(desc.fingerprint, "LBHNC")) is None


def test_sparse_index_opaque():
    cfg, views = fixture(compressed=4, role=SimpleNamespace(name="HISPARSE_INDEXER"))
    (desc,) = VllmLayoutAdapter().describe(views, cfg)
    assert desc.semantics.kind == StateKind.SPARSE_INDEX


def test_sw_semantics_affect_fingerprint():
    cfg, views = fixture("SlidingWindowSpec")
    (first,) = VllmLayoutAdapter().describe(views, cfg)
    cfg.kv_cache_groups[0].kv_cache_spec.sliding_window += 1
    (second,) = VllmLayoutAdapter().describe(views, cfg)
    assert first.fingerprint != second.fingerprint
    assert first.semantics.extra_retained_tokens == 3


@pytest.mark.parametrize(
    "window,extra,boundary,expected",
    [
        (33, 0, 128, [6, 7]),
        (34, 0, 128, [5, 6, 7]),
        (33, 1, 128, [5, 6, 7]),
        (1, 0, 128, []),
        (512, 0, 128, list(range(8))),
    ],
)
def test_window_restore_trim(window, extra, boundary, expected):
    sem = StateSemantics(
        StateKind.SLIDING_WINDOW_KV, 16, 1, 16, window_tokens=window, extra_retained_tokens=extra
    )
    assert list(required_blocks(sem, boundary)) == expected
    assert all(i >= 7 for i in required_blocks(sem, boundary, computed=112))


def test_recurrent_fails_closed():
    cfg, views = fixture("MambaSpec")
    with pytest.raises(RegistrationError, match="boundary metadata"):
        VllmLayoutAdapter().describe(views, cfg)


def test_namespace_and_scale_identity():
    cfg, views = fixture(quant="FP8_PER_TENSOR")
    (desc,) = VllmLayoutAdapter().describe(views, cfg)
    a = replace(desc.fingerprint, namespace=("model", 0), scale_layout="k=1,v=2")
    assert a.identity != replace(a, namespace=("model", 1)).identity
    assert a.identity != replace(a, scale_layout="k=2,v=2").identity


def test_bad_content_shape_rejected():
    cfg, views = fixture()
    views["layer"]._shape = (4, 1, 16, 32)
    with pytest.raises(RegistrationError, match="content"):
        VllmLayoutAdapter().describe(views, cfg)


@pytest.mark.parametrize(
    "quant", ["INT8_PER_TOKEN_HEAD", "FP8_PER_TOKEN_HEAD", "INT4_PER_TOKEN_HEAD"]
)
def test_inline_quant_requires_explicit_content(quant):
    cfg, views = fixture(quant=quant)
    with pytest.raises(RegistrationError, match="inline"):
        VllmLayoutAdapter().describe(views, cfg)
    cfg.kv_cache_groups[0].kv_cache_spec.state_content_bytes = 16
    (desc,) = VllmLayoutAdapter().describe(views, cfg)
    assert desc.fingerprint.quant == quant
    assert DEFAULT_REGISTRY.find(desc.fingerprint, swap_layout(desc.fingerprint, "LBHNC")) is None


def test_custom_quant_fails_closed():
    cfg, views = fixture(quant="CUSTOM")
    cfg.kv_cache_groups[0].kv_cache_spec.state_content_bytes = 16
    with pytest.raises(RegistrationError, match="scale contract"):
        VllmLayoutAdapter().describe(views, cfg)

"""Unit tests for page codecs (omnikv.codec) — round-trip properties."""

from __future__ import annotations

import pytest

from omnikv.codec import (
    DEFAULT_REGISTRY,
    CodecRegistry,
    HndToNhdCodec,
    NhdToHndCodec,
    swap_layout,
)
from omnikv.errors import CompatibilityError
from omnikv.types import FORMAT_VERSION, FormatFingerprint, StateKind

try:
    import torch
except ImportError:
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="codec needs torch")

N, H, C = 64, 2, 512
PAGE = N * H * C


def nhd_fp(**over) -> FormatFingerprint:
    kwargs = {
        "state_kind": StateKind.PAGED_KV,
        "layout": "LBNHC",
        "dtype": ("bfloat16",),
        "quant": "NONE",
        "head_geometry": (H, 128, 128, N),
        "page_bytes": (PAGE, PAGE),
        "scale_layout": None,
        "page_fragments": (PAGE,),
        "version": FORMAT_VERSION,
    }
    kwargs.update(over)
    return FormatFingerprint(**kwargs)


def hnd_fp() -> FormatFingerprint:
    return swap_layout(nhd_fp(), "LBHNC")


def random_page(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (PAGE,), dtype=torch.uint8, generator=g)


class TestRegistry:
    def test_find_both_directions(self):
        assert isinstance(DEFAULT_REGISTRY.find(nhd_fp(), hnd_fp()), NhdToHndCodec)
        assert isinstance(DEFAULT_REGISTRY.find(hnd_fp(), nhd_fp()), HndToNhdCodec)

    def test_find_rejects_incompatible(self):
        assert DEFAULT_REGISTRY.find(nhd_fp(dtype=("float16",)), hnd_fp()) is None
        assert DEFAULT_REGISTRY.find(nhd_fp(quant="FP8_PER_TENSOR"), hnd_fp()) is None
        # No codec serves fragmented sources:
        frag = swap_layout(nhd_fp(), "LHBNC")
        assert DEFAULT_REGISTRY.find(frag, hnd_fp()) is None
        # Unknown layout pair:
        empty = CodecRegistry()
        assert empty.find(nhd_fp(), hnd_fp()) is None

    def test_source_fingerprints_for(self):
        srcs = DEFAULT_REGISTRY.source_fingerprints_for(hnd_fp())
        assert [fp.layout for fp in srcs] == ["LBNHC"]
        assert srcs[0].identity == nhd_fp().identity

    def test_swap_layout_adjusts_fragments(self):
        frag = swap_layout(nhd_fp(), "LHBNC")
        assert frag.fragments == (PAGE // H, PAGE // H)


class TestRepack:
    @needs_torch
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_round_trip_is_identity(self, seed):
        page = random_page(seed)
        fp_src, fp_dst = nhd_fp(), hnd_fp()
        mid = torch.empty_like(page)
        back = torch.empty_like(page)
        DEFAULT_REGISTRY.find(fp_src, fp_dst).convert(page, mid, fp_src, fp_dst)
        DEFAULT_REGISTRY.find(fp_dst, fp_src).convert(mid, back, fp_dst, fp_src)
        assert torch.equal(page, back)

    @needs_torch
    def test_repack_matches_reference_permutation(self):
        # Reference: element (n, h, c) of an NHD page lands at (h, n, c) of
        # the HND page — the defining property of the layout families.
        page = random_page(42)
        fp_src, fp_dst = nhd_fp(), hnd_fp()
        out = torch.empty_like(page)
        DEFAULT_REGISTRY.find(fp_src, fp_dst).convert(page, out, fp_src, fp_dst)
        ref = page.view(N, H, C).permute(1, 0, 2).reshape(-1)
        assert torch.equal(out, ref)

    @needs_torch
    def test_size_mismatch_rejected(self):
        fp_src, fp_dst = nhd_fp(), hnd_fp()
        bad = torch.empty(PAGE - 1, dtype=torch.uint8)
        out = torch.empty(PAGE, dtype=torch.uint8)
        with pytest.raises(CompatibilityError, match="size mismatch"):
            DEFAULT_REGISTRY.find(fp_src, fp_dst).convert(bad, out, fp_src, fp_dst)

    def test_geometry_validation(self):
        # A page payload not divisible into (token, head) rows is rejected.
        bad = nhd_fp(page_bytes=(60000, 60000), page_fragments=(60000,))
        codec = NhdToHndCodec()
        with pytest.raises(CompatibilityError, match="not divisible"):
            codec.geometry(bad)

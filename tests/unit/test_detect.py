"""Unit tests for stride certification (omnikv.detect)."""

from __future__ import annotations

import pytest

from omnikv.detect import certify_page_view
from omnikv.errors import LayoutError

try:
    import torch
except ImportError:
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="detect needs torch")

N, H, C = 64, 2, 512
BLOCKS = 16


def page_tensor(family: str):
    """A [B, H, N, C] view whose strides implement ``family``."""
    physical = torch.randint(0, 256, (BLOCKS * N * H * C,), dtype=torch.uint8)
    if family == "NHD":
        return physical.view(BLOCKS, N, H, C).permute(0, 2, 1, 3)
    if family == "HND":
        return physical.view(BLOCKS, H, N, C)
    raise AssertionError(family)


class TestCertify:
    @needs_torch
    def test_certifies_nhd(self):
        certified = certify_page_view(page_tensor("NHD"))
        assert certified.family == "NHD"
        assert (certified.num_blocks, certified.heads, certified.tokens) == (
            BLOCKS,
            H,
            N,
        )

    @needs_torch
    def test_certifies_hnd(self):
        certified = certify_page_view(page_tensor("HND"))
        assert certified.family == "HND"

    @needs_torch
    def test_ambiguous_square_shape_raises(self):
        # N == H with strides that satisfy BOTH readings -> must not guess.
        buf = torch.randint(0, 256, (200000,), dtype=torch.uint8)
        view = buf.as_strided((4, 8, 8, 512), (32768, 4096, 4096, 1))
        with pytest.raises(LayoutError, match="ambiguous"):
            certify_page_view(view)
        # An explicit hint resolves it:
        certified = certify_page_view(view, hint="NHD")
        assert certified.family == "NHD"

    @needs_torch
    def test_non_contiguous_inner_dim_rejected(self):
        # Transposing the last two dims makes stride[-1] != 1.
        view = torch.empty(4, H, N, C, dtype=torch.uint8).transpose(2, 3)
        with pytest.raises(LayoutError, match="inner dim not contiguous"):
            certify_page_view(view)

    @needs_torch
    def test_uncertifiable_strides_rejected(self):
        view = torch.empty(4, H, N, C, dtype=torch.uint8).transpose(1, 2)
        view = view.transpose(1, 2).transpose(2, 3)  # scrambled strides
        with pytest.raises(LayoutError):
            certify_page_view(view)

    @needs_torch
    def test_dim0_padding_rejected(self):
        # Padded block stride must come from engine metadata, not detection
        # (research.md §4.3: detection tier never guesses padding).
        buf = torch.empty(BLOCKS * (N * H * C + 256), dtype=torch.uint8)
        view = buf.as_strided((BLOCKS, H, N, C), (N * H * C + 256, C, H * C, 1))
        with pytest.raises(LayoutError, match="non-dense"):
            certify_page_view(view)

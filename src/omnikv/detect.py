"""Stride certification for raw engine tensors (detection fallback).

Used when engine metadata is unavailable (non-vLLM engines, debugging).
This is the *detection* tier of the source-of-truth priority
(research.md §4.3): it never runs silently ahead of metadata, and shapes
that admit more than one reading raise instead of guessing (the RBLN
``num_blocks == 2`` ambiguity class, research.md §2.2).

The invariants mirror LMCache's ``contiguity.py`` findings: a certifiable
page view is contiguous in its inner dims, and either fully dense
(block-compact) or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnikv.errors import LayoutError


@dataclass(frozen=True)
class CertifiedPage:
    """A tensor page view certified as one layout family."""

    family: str  # "NHD" or "HND"
    num_blocks: int
    tokens: int
    heads: int
    content_bytes: int


def certify_page_view(tensor, hint: str | None = None) -> CertifiedPage:
    """Certify a [B, H, N, C-bytes] uint8 view as NHD or HND.

    Args:
        tensor: a torch tensor view with logical shape (B, H, N, C).
        hint: optional ``"NHD"``/``"HND"`` to disambiguate N == H shapes.

    Raises:
        LayoutError: the view is not contiguous in its inner dims, its
            strides admit both readings without a hint (ambiguous), or
            admit neither (uncertified).
    """
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    if len(shape) != 4:
        raise LayoutError(f"expected a rank-4 page view, got shape {shape}")
    blocks, heads, tokens, content = shape
    if content <= 0 or heads <= 0 or tokens <= 0 or blocks <= 0:
        raise LayoutError(f"degenerate page shape {shape}")
    if strides[3] != 1:
        raise LayoutError(
            f"inner dim not contiguous (stride {strides[3]}); only "
            "inner-contiguous pages are certifiable"
        )

    nhd_ok = strides[2] == heads * content  # token row spans all heads
    hnd_ok = strides[1] == tokens * content  # head run spans all tokens
    dense = strides[0] == heads * tokens * content  # page is one dense run

    if nhd_ok and hnd_ok and heads == tokens:
        if hint not in ("NHD", "HND"):
            raise LayoutError(
                f"ambiguous page view: N == H == {heads} admits both NHD "
                "and HND readings; pass an explicit hint (research.md "
                "§2.2 detection must not guess)"
            )
        family = hint
    elif nhd_ok:
        family = "NHD"
    elif hnd_ok:
        family = "HND"
    else:
        raise LayoutError(
            f"uncertified page view: shape {shape} strides {strides} match "
            "neither the NHD nor the HND invariants"
        )
    if not dense:
        raise LayoutError(
            f"non-dense page view (block stride {strides[0]} != page); "
            "dim-0 padding must come from engine metadata, not detection"
        )
    return CertifiedPage(
        family=family,
        num_blocks=blocks,
        tokens=tokens,
        heads=heads,
        content_bytes=content,
    )

"""Page codecs: explicit, registered conversions between store-page formats
(research.md §4.4). Conversion happens ONLY here — the passthrough path
contains none.

Each codec converts one *store page* (the concatenated fragments of one
block, as pinned by the source `FormatFingerprint`) into the destination
fingerprint's store page. The initial codec family is the NHD<->HND token
row repack for plain PAGED_KV pages:

  LBNHC (NHD) store page: token-major, [N, H, C] — token ``n``'s H head
      rows are adjacent at ``n*H*C + h*C``.
  LBHNC (HND) store page: head-major, [H, N, C] — head ``h``'s N token
      rows are adjacent at ``h*N*C + n*C``.

The repack is a pure permutation of C-sized rows, executed with torch
views on host memory (the store pool is a pinned torch tensor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from omnikv.errors import CompatibilityError
from omnikv.types import FormatFingerprint, StateKind

try:
    import torch
except ImportError:  # pragma: no cover - codec execution needs torch
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CodecKey:
    """The (src, dst) layout pair a codec serves."""

    src_layout: str
    dst_layout: str


class PageCodec:
    """Converts store pages between two fingerprint classes.

    Class attributes declare the layout pair; `convertible` narrows to
    exact fingerprint compatibility (everything but layout + fragments
    equal). `convert` moves one store page ``src`` -> ``dst``; both are
    1-D uint8 torch tensor slices of equal length.
    """

    src_layout: str = ""
    dst_layout: str = ""

    def convertible(self, src: FormatFingerprint, dst: FormatFingerprint) -> bool:
        if (src.layout, dst.layout) != (self.src_layout, self.dst_layout):
            return False
        if src.state_kind is not StateKind.PAGED_KV:
            return False
        if dst.state_kind is not src.state_kind:
            return False
        return (
            src.dtype == dst.dtype
            and src.quant == dst.quant
            and src.head_geometry == dst.head_geometry
            and src.page_bytes == dst.page_bytes
            and src.scale_layout == dst.scale_layout
            and src.version == dst.version
        )

    def geometry(self, fp: FormatFingerprint) -> tuple[int, int, int]:
        """(N tokens, H heads, C row bytes) of one page.

        C is the per-(token, head) content size derived from the page
        payload; fragments describe GPU-side run decomposition and are
        irrelevant to the row permutation (fragmented layouts are not
        served by this codec family anyway).
        """
        heads, _, _, block_size = fp.head_geometry
        payload = fp.page_bytes[0]
        if payload % (block_size * heads) != 0:
            raise CompatibilityError(
                f"page payload {payload} not divisible into (token, head) "
                f"rows (block_size={block_size}, heads={heads})"
            )
        return block_size, heads, payload // (block_size * heads)

    def convert(
        self,
        src,  # torch.Tensor (uint8, 1-D)
        dst,  # torch.Tensor (uint8, 1-D)
        fp_src: FormatFingerprint,
        fp_dst: FormatFingerprint,
    ) -> None:
        n, h, c = self.geometry(fp_src)
        total = n * h * c
        if src.numel() != total or dst.numel() != total:
            raise CompatibilityError(
                f"page size mismatch: src={src.numel()} dst={dst.numel()} "
                f"expected={total}"
            )
        if fp_src.layout == "LBNHC":
            dst.view(h, n, c).copy_(src.view(n, h, c).permute(1, 0, 2))
        else:
            dst.view(n, h, c).copy_(src.view(h, n, c).permute(1, 0, 2))


class NhdToHndCodec(PageCodec):
    src_layout = "LBNHC"
    dst_layout = "LBHNC"


class HndToNhdCodec(PageCodec):
    src_layout = "LBHNC"
    dst_layout = "LBNHC"


def swap_layout(fp: FormatFingerprint, layout: str) -> FormatFingerprint:
    """The same physical format under another layout family (fragments
    adjusted). Used to derive alternate-fingerprint lookup keys."""
    fragments: tuple[int, ...]
    if layout in ("LHBNC", "BHLNC"):
        heads = fp.head_geometry[0]
        row = fp.page_bytes[0] // heads
        fragments = tuple(row for _ in range(heads))
    else:
        fragments = (fp.page_bytes[0],)
    return replace(fp, layout=layout, page_fragments=fragments)


class CodecRegistry:
    """Registry of (src, dst) page codecs. Lookups never guess: an absent
    codec is a hard compatibility error (research.md §7.2)."""

    def __init__(self) -> None:
        self._codecs: dict[CodecKey, PageCodec] = {}

    def register(self, codec: PageCodec) -> None:
        self._codecs[CodecKey(codec.src_layout, codec.dst_layout)] = codec

    def find(
        self, src: FormatFingerprint, dst: FormatFingerprint
    ) -> PageCodec | None:
        codec = self._codecs.get(CodecKey(src.layout, dst.layout))
        if codec is not None and codec.convertible(src, dst):
            return codec
        return None

    def source_fingerprints_for(self, dst: FormatFingerprint) -> list[FormatFingerprint]:
        """Source fingerprints whose pages can be converted into ``dst``."""
        out = []
        for key in self._codecs:
            if key.dst_layout == dst.layout:
                src = swap_layout(dst, key.src_layout)
                if self._codecs[key].convertible(src, dst):
                    out.append(src)
        return out


# The default registry: both directions of the NHD/HND repack.
DEFAULT_REGISTRY = CodecRegistry()
DEFAULT_REGISTRY.register(NhdToHndCodec())
DEFAULT_REGISTRY.register(HndToNhdCodec())

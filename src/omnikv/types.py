"""Core value types: the layout-agnostic addressing model (research.md §4.2).

The core engine never sees tensor dimensions or K/V ordering. It sees:

- `StateKind` / `StateSemantics` — what the bytes *mean* and what a correct
  restore requires (semantic compatibility).
- `FormatFingerprint` — the physical format identity; equality gates
  zero-copy passthrough (physical compatibility).
- `PageGeometry` — how (group, layer, block) maps to bytes inside a backing
  allocation: ``offset + layer_index * layer_stride + block_id * block_stride``.
- `CopyRun` / `GpuRegion` / `BlockView` — resolved per-block addressing.

All types are immutable value objects; nothing here launches work or owns
tensors (execution lives in `omnikv.transfer`, ownership in the connector).

Verified against vLLM main @898a7c6 `kv_cache_interface.py` and the
v0.34.0 release (`KVCacheTensor`, `KVCacheLayout`); empirically confirmed
by the 2026-11-20 smoke test (layer_stride != num_blocks * page).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class StateKind(Enum):
    """Semantic class of a cache group's contents (research.md §3.1)."""

    PAGED_KV = "paged_kv"
    LATENT_KV = "latent_kv"
    SLIDING_WINDOW_KV = "sliding_window_kv"
    RECURRENT_STATE = "recurrent_state"
    SPARSE_INDEX = "sparse_index"
    AUX_STATE = "aux_state"


# Physical layout families. Names mirror vLLM's `KVCacheLayout` stride
# permutations over logical axes [L, B, H, N, C] (RFC #42082); "NHD"/"HND"
# are the per-layer spellings used by LMCache and connector APIs.
LAYOUT_FAMILIES = frozenset(
    {"LBHNC", "LBNHC", "LHBNC", "BLHNC", "BLNHC", "BHLNC"}
)

# Bump on any change to fingerprint derivation or run semantics.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class FormatFingerprint:
    """Physical compatibility identity of one cache group's pages.

    Two pages may be byte-copied between each other without reinterpretation
    iff their fingerprints compare equal (same state content encoding, same
    stride permutation, same dtype/quant, same geometry, same version).
    """

    state_kind: StateKind
    layout: str
    dtype: tuple[str, ...]
    quant: str
    head_geometry: tuple[int, ...]
    page_bytes: tuple[int, ...]
    scale_layout: str | None
    version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.layout not in LAYOUT_FAMILIES:
            raise ValueError(
                f"unknown layout family {self.layout!r}; "
                f"expected one of {sorted(LAYOUT_FAMILIES)}"
            )

    @property
    def identity(self) -> str:
        """Stable short id used in store keys and log messages."""
        import hashlib

        payload = repr(
            (
                self.state_kind.value,
                self.layout,
                self.dtype,
                self.quant,
                self.head_geometry,
                self.page_bytes,
                self.scale_layout,
                self.version,
            )
        )
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


@dataclass(frozen=True)
class StateSemantics:
    """The restore contract of one cache group (research.md §3.1/§7.3)."""

    kind: StateKind
    block_size: int
    tokens_per_state: float
    slots_per_block: int
    window_tokens: int | None = None
    prefix_cacheable: bool = True
    dcp_sharded: bool = False
    tp_replicated: bool = False


@dataclass(frozen=True)
class PageGeometry:
    """Placement of a group's pages inside one backing allocation.

    Mirrors vLLM's `KVCacheTensor`: layer ``l``'s page for block ``b``
    starts at ``offset + l * layer_stride + b * block_stride`` bytes.
    Strides are authoritative — they are NOT derivable from ``num_blocks``
    (measured: the extensible KV cache pads ~111 block slots per layer on
    vLLM 0.34.0 / Qwen2.5-1.5B; see docs/TESTLOG.md 2026-11-20) nor from
    the page size (block-outermost layouts stride blocks across layers).
    """

    num_blocks: int
    page_bytes: int
    layer_stride: int
    block_stride: int
    storage_id: int
    offset: int
    layer_names: tuple[str, ...]
    # None => each layer lives in its own storage (separate per-layer
    # tensors); the (storage_id, offset) then addresses layer 0 only and
    # adapters must describe per-layer storages instead.
    shared_storage: bool = True

    def __post_init__(self) -> None:
        if self.page_bytes <= 0 or self.layer_stride <= 0:
            raise ValueError("page_bytes and layer_stride must be positive")
        if self.block_stride <= 0:
            raise ValueError("block_stride must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not self.layer_names:
            raise ValueError("layer_names must not be empty")

    def layer_block_offset(self, layer_index: int, block_id: int) -> int:
        """Byte offset of one (layer, block) page inside the storage."""
        return (
            self.offset
            + layer_index * self.layer_stride
            + block_id * self.block_stride
        )

    @property
    def num_layers(self) -> int:
        return len(self.layer_names)


@dataclass(frozen=True)
class CopyRun:
    """A strided byte correspondence between two pages.

    For ``i in range(num_fragments)``: fragment ``i`` spans
    ``[local_offset + i*local_stride, +fragment_size)`` in the local page
    and ``[remote_offset + i*remote_stride, +fragment_size)`` remotely.
    Mirrors vLLM's offloading `CopyRun` (research.md §2.1.6).
    """

    local_offset: int
    remote_offset: int
    fragment_size: int
    num_fragments: int = 1
    local_stride: int = 0
    remote_stride: int = 0

    def __post_init__(self) -> None:
        if self.fragment_size <= 0:
            raise ValueError("fragment_size must be positive")
        if self.num_fragments < 1:
            raise ValueError("num_fragments must be >= 1")
        if self.num_fragments > 1 and (
            self.local_stride < self.fragment_size
            or self.remote_stride < self.fragment_size
        ):
            raise ValueError("strided runs must not self-overlap")

    @property
    def total_bytes(self) -> int:
        return self.fragment_size * self.num_fragments


@dataclass(frozen=True)
class GpuRegion:
    """Absolute addressing of one page inside a registered allocation."""

    storage_id: int
    device_index: int
    offset: int
    size: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.size <= 0:
            raise ValueError("invalid region bounds")


@dataclass(frozen=True)
class BlockView:
    """Resolved addressing for one physical block of one cache group.

    Pure value object: addressing + compatibility only. Produced exclusively
    by a `omnikv.adapter.LayoutAdapter`; the core never constructs these
    from raw tensors.
    """

    group_id: int
    block_id: int
    semantics: StateSemantics
    fingerprint: FormatFingerprint
    geometry: PageGeometry
    # Page-relative byte correspondence GPU page <-> store page. Empty means
    # "whole page, identity" (the M1 passthrough case).
    runs: tuple[CopyRun, ...] = ()

    @property
    def is_single_run(self) -> bool:
        """True when the page is one contiguous byte run (memcpy-able)."""
        return not self.runs or (
            len(self.runs) == 1
            and self.runs[0].num_fragments == 1
            and self.runs[0].fragment_size == self.geometry.page_bytes
        )

    def gpu_regions(self, device_index: int = 0) -> tuple[GpuRegion, ...]:
        """Absolute GPU regions for this block, one per layer (M1: whole
        contiguous pages; run-decomposed regions arrive with M2 codecs)."""
        geo = self.geometry
        if not geo.shared_storage:
            raise NotImplementedError(
                "per-layer storages are not supported until M2"
            )
        if not self.is_single_run:
            raise NotImplementedError(
                "run-decomposed regions are not supported until M2"
            )
        return tuple(
            GpuRegion(
                storage_id=geo.storage_id,
                device_index=device_index,
                offset=geo.layer_block_offset(layer_index, self.block_id),
                size=geo.page_bytes,
            )
            for layer_index in range(geo.num_layers)
        )


@dataclass(frozen=True)
class GroupDescriptor:
    """Everything the core knows about one engine cache group."""

    group_id: int
    semantics: StateSemantics
    fingerprint: FormatFingerprint
    geometry: PageGeometry

    def block_view(self, block_id: int) -> BlockView:
        return BlockView(
            group_id=self.group_id,
            block_id=block_id,
            semantics=self.semantics,
            fingerprint=self.fingerprint,
            geometry=self.geometry,
        )

    def block_views(self, block_ids: Sequence[int]) -> tuple[BlockView, ...]:
        return tuple(self.block_view(b) for b in block_ids)

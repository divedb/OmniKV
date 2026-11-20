"""CPU page store: staging pages + a hash-addressed index (M1).

Store page layout: the L pages of one engine block, stacked layer-major
(``layer l`` at byte offset ``l * page_bytes``). The stored page is opaque:
it is the producer's native page bytes, tagged with the producer's
`FormatFingerprint` via the `BlockKey`. Cross-fingerprint lookups simply
miss (different namespace); nothing is ever reinterpreted (research.md §5,
design C).

Two correctness rules are enforced here (research.md §7.4):

- Readiness-before-visibility: pages become lookup-visible only via
  `publish`, called after the D2H copy completes.
- Ownership: pages counted toward a scheduler-side match are *reserved*
  against eviction until their H2D copy lands.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict

from omnikv.errors import StoreError
from omnikv.keys import BlockKey
from omnikv.types import GroupDescriptor

try:  # torch is required at runtime (pinned pages + CUDA); optional for
    # CPU-only unit tests, where a plain ctypes buffer stands in.
    import torch
except ImportError:  # pragma: no cover - local test hosts only
    torch = None  # type: ignore[assignment]


class _Pool:
    """Contiguous byte pool with a stable base pointer.

    torch tensor (pinned when CUDA is available) in production; a ctypes
    buffer on hosts without torch. Exposes ``data_ptr()`` and ``nbytes``.
    """

    def __init__(self, nbytes: int, pin_memory: bool):
        if nbytes <= 0:
            raise ValueError("pool size must be positive")
        self.nbytes = nbytes
        if torch is not None:
            self._tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin_memory)
            self._buf = None
        else:
            if pin_memory:
                raise StoreError("pin_memory requires torch (CUDA host)")
            self._tensor = None
            self._buf = ctypes.create_string_buffer(nbytes)

    def data_ptr(self) -> int:
        if self._tensor is not None:
            return self._tensor.data_ptr()
        return ctypes.addressof(self._buf)


class CpuPageStore:
    """A fixed-capacity pool of whole engine blocks (all layers).

    Single-threaded M1 semantics: the scheduler-role lookup and the
    worker-role publish run on the engine's threads, serialized by the
    engine step (the in-process M1 deployment documented in
    docs/ENVIRONMENT.md). Cross-process splits arrive with M3.
    """

    def __init__(
        self,
        descriptor: GroupDescriptor,
        num_pages: int,
        pin_memory: bool = True,
    ):
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        self.descriptor = descriptor
        self.num_pages = num_pages
        self.page_bytes = descriptor.geometry.page_bytes
        self.slot_bytes = self.page_bytes * descriptor.geometry.num_layers
        # One allocation for the whole pool: pages are contiguous slots,
        # which keeps D2H/H2D batches pointer-arithmetic only.
        self.pool = _Pool(self.slot_bytes * num_pages, pin_memory=pin_memory)
        # LRU: key -> page index (most-recently-used last).
        self._lru: OrderedDict[BlockKey, int] = OrderedDict()
        self._index: dict[int, BlockKey] = {}
        self._free: list[int] = list(range(num_pages - 1, -1, -1))
        self._inflight: set[int] = set()
        self._reserved: set[int] = set()

    # -- lookup ---------------------------------------------------------

    def longest_prefix(self, keys: list[BlockKey]) -> list[int]:
        """Page indices of the longest contiguous prefix of ``keys`` present.

        The contiguity requirement is the cache-prefix semantics: a hole
        means everything after it is unusable for this request. Hit pages
        are touched in the LRU.
        """
        pages: list[int] = []
        for key in keys:
            idx = self._lru.get(key)
            if idx is None:
                break
            pages.append(idx)
            self._lru.move_to_end(key)
        return pages

    # -- ownership (epochs) ----------------------------------------------

    def reserve_pages(self, pages: list[int]) -> None:
        """Pin pages against eviction (scheduler lookup -> worker copy)."""
        self._reserved.update(pages)

    def release_pages(self, pages: list[int]) -> None:
        """Drop reservations after the copy completed (or was abandoned)."""
        self._reserved.difference_update(pages)

    # -- publish / evict -------------------------------------------------

    def acquire_page(self) -> int:
        """Reserve a free page slot for an in-flight save.

        Evicts the least-recently-used *unreserved* entry when the pool is
        full. The slot is *not* lookup-visible until `publish`.
        """
        if not self._free:
            victim_idx = next(
                (i for i in self._lru.values() if i not in self._reserved), None
            )
            if victim_idx is None:
                raise StoreError(
                    "page pool exhausted: every page is reserved for an "
                    "in-flight load; increase omnikv.cpu_pages"
                )
            victim_key = self._index[victim_idx]
            del self._lru[victim_key]
            self._index.pop(victim_idx, None)
            self._free.append(victim_idx)
        idx = self._free.pop()
        self._inflight.add(idx)
        return idx

    def publish(self, key: BlockKey, page_index: int) -> None:
        """Make an in-flight page lookup-visible (idempotent overwrite)."""
        if page_index not in self._inflight:
            raise StoreError(f"page {page_index} was not acquired")
        self._inflight.discard(page_index)
        old_key = self._index.get(page_index)
        if old_key is not None:
            del self._lru[old_key]
        self._lru[key] = page_index
        self._index[page_index] = key

    def release(self, page_index: int) -> None:
        """Return an acquired-but-unpublished page to the free list."""
        self._inflight.discard(page_index)
        self._free.append(page_index)

    # -- addressing ------------------------------------------------------

    def page_ptr(self, page_index: int) -> int:
        """Host pointer of one stored page slot (all layers, layer-major)."""
        return self.pool.data_ptr() + page_index * self.slot_bytes

    def pool_slice(self, page_index: int):
        """The store page slot as a 1-D uint8 tensor slice (codec I/O)."""
        if self.pool._tensor is None:  # local-test ctypes pool
            raise StoreError("pool_slice requires a torch-backed pool")
        return self.pool._tensor[page_index * self.slot_bytes : (page_index + 1) * self.slot_bytes]

    def touch(self, key: BlockKey) -> None:
        """Mark a key most-recently-used (eviction-order hygiene)."""
        if key in self._lru:
            self._lru.move_to_end(key)

    def layer_offset(self, layer_index: int) -> int:
        """Byte offset of one layer's page within a stored page slot."""
        return layer_index * self.page_bytes

    # -- stats / inspection ----------------------------------------------

    def __len__(self) -> int:
        return len(self._lru)

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def reserved_pages(self) -> int:
        return len(self._reserved)

    def contains(self, key: BlockKey) -> bool:
        return key in self._lru

    def page_of(self, key: BlockKey) -> int | None:
        return self._lru.get(key)

"""Spill tier: hot (pinned CPU) + cold (disk) stores under one interface.

Exposes the connector-facing surface of `CpuPageStore`. Evictions from the
hot tier spill to disk instead of vanishing; lookups consult hot then cold,
promoting cold hits into hot pages so the codec and transfer paths always
operate on pinned host memory (research.md §9 M3 "pluggable storage
backends"; the FlexKV three-tier pattern, minus the remote tier).
"""

from __future__ import annotations

try:  # torch is required at runtime (pinned pages); optional for the
    # torch-free local unit-test hosts.
    import torch
except ImportError:  # pragma: no cover - local test hosts only
    torch = None  # type: ignore[assignment]

from omnikv.errors import StoreError
from omnikv.keys import BlockKey
from omnikv.store.cpu import CpuPageStore
from omnikv.store.disk import DiskPageStore


class SpillStore:
    """Hot CpuPageStore + cold DiskPageStore with promotion on hit."""

    def __init__(self, hot: CpuPageStore, cold: DiskPageStore):
        self.hot = hot
        self.cold = cold
        hot.on_evict = self._spill

    # -- spill / promote ----------------------------------------------------

    def _spill(self, key: BlockKey, page: int) -> None:
        if self.hot.pool._tensor is None:
            return  # local-test ctypes pool: no bytes to spill
        slot = self.hot.pool_slice(page)
        self.cold.write(key, slot.numpy().tobytes())

    def _promote(self, key: BlockKey) -> int:
        """Read a cold page into a fresh hot page and return its index."""
        data = self.cold.read(key)
        self.cold.drop(key)
        try:
            hot_page = self.hot.acquire_page()
        except StoreError:
            # Hot tier full of reserved pages: keep the cold entry and
            # report the miss (the caller treats it as absent).
            self.cold.write(key, data)
            raise
        if self.hot.pool._tensor is None:
            raise StoreError("promotion requires a torch-backed hot pool")
        self.hot.pool._tensor[
            hot_page * self.hot.slot_bytes : (hot_page + 1) * self.hot.slot_bytes
        ].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        self.hot.publish(key, hot_page)  # publish clears the inflight mark
        return hot_page

    # -- connector-facing interface (CpuPageStore subset) -------------------

    def __len__(self) -> int:
        return len(self.hot) + len(self.cold)

    @property
    def free_pages(self) -> int:
        return self.hot.free_pages

    @property
    def reserved_pages(self) -> int:
        return self.hot.reserved_pages

    @property
    def page_bytes(self) -> int:
        return self.hot.page_bytes

    @property
    def slot_bytes(self) -> int:
        return self.hot.slot_bytes

    def contains(self, key: BlockKey) -> bool:
        return self.hot.contains(key) or self.cold.contains(key)

    def page_of(self, key: BlockKey) -> int | None:
        page = self.hot.page_of(key)
        if page is not None:
            return page
        if self.cold.contains(key):
            try:
                return self._promote(key)
            except StoreError:
                return None
        return None

    def touch(self, key: BlockKey) -> None:
        self.hot.touch(key)
        self.cold.touch(key)

    def reserve_pages(self, pages: list[int]) -> None:
        self.hot.reserve_pages(pages)

    def release_pages(self, pages: list[int]) -> None:
        self.hot.release_pages(pages)

    def acquire_page(self) -> int:
        return self.hot.acquire_page()

    def publish(self, key: BlockKey, page_index: int) -> None:
        # Publishing to hot shadows any cold copy of the same key.
        self.cold.drop(key)
        self.hot.publish(key, page_index)

    def release(self, page_index: int) -> None:
        self.hot.release(page_index)

    def page_ptr(self, page_index: int) -> int:
        return self.hot.page_ptr(page_index)

    def pool_slice(self, page_index: int):
        return self.hot.pool_slice(page_index)

    def layer_offset(self, layer_index: int) -> int:
        return self.hot.layer_offset(layer_index)

    def audit(self) -> None:
        self.hot.audit()
        free, lru = self.cold.free_pages, len(self.cold)
        if free + lru != self.cold.num_pages:
            raise StoreError(
                f"cold tier page leak: free={free} lru={lru} "
                f"total={self.cold.num_pages}"
            )

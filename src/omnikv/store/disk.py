"""Disk page store: an SSD-backed cold tier (M3).

Pages live at fixed offsets of one file per fingerprint (``<dir>/<fp>.pages``),
written with ``os.pwrite`` on spill and read with ``os.pread`` on promotion.
The hash->page index is in memory (a restart = cold tier; the disk file is
a cache, not a database — research.md §9 M3 "SSD store, io_uring/GDS
optional": plain pread/pwrite first, faster I/O paths are drop-in later).
"""

from __future__ import annotations

import os
from collections import OrderedDict

from omnikv.errors import StoreError
from omnikv.keys import BlockKey
from omnikv.types import GroupDescriptor


class DiskPageStore:
    """Fixed-capacity page file with LRU eviction, mirroring the interface
    subset the spill tier needs (contains / read / write / evict)."""

    def __init__(self, descriptor: GroupDescriptor, directory: str, num_pages: int):
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        os.makedirs(directory, exist_ok=True)
        self.descriptor = descriptor
        self.num_pages = num_pages
        self.slot_bytes = descriptor.geometry.page_bytes * (
            descriptor.geometry.num_layers
        )
        self.path = os.path.join(directory, f"{descriptor.fingerprint.identity}.pages")
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        self._lru: OrderedDict[BlockKey, int] = OrderedDict()
        self._free: list[int] = list(range(num_pages - 1, -1, -1))

    # -- capacity / lookup ------------------------------------------------

    def __len__(self) -> int:
        return len(self._lru)

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def contains(self, key: BlockKey) -> bool:
        return key in self._lru

    def touch(self, key: BlockKey) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)

    # -- data path ---------------------------------------------------------

    def write(self, key: BlockKey, data) -> None:
        """Store one page slot (bytes-like of slot_bytes). LRU-evicts."""
        if len(data) != self.slot_bytes:
            raise StoreError(
                f"disk page write of {len(data)} B != slot {self.slot_bytes} B"
            )
        if key in self._lru:
            idx = self._lru[key]
        else:
            if not self._free:
                evict_key, idx = self._lru.popitem(last=False)
                del evict_key
                self._free.append(idx)
            idx = self._free.pop()
            self._lru[key] = idx
        self._lru.move_to_end(key)
        os.pwrite(self._fd, bytes(data), idx * self.slot_bytes)

    def read(self, key: BlockKey) -> bytes:
        idx = self._lru.get(key)
        if idx is None:
            raise StoreError(f"disk page miss for {key!r}")
        self._lru.move_to_end(key)
        return os.pread(self._fd, self.slot_bytes, idx * self.slot_bytes)

    def drop(self, key: BlockKey) -> None:
        idx = self._lru.pop(key, None)
        if idx is not None:
            self._free.append(idx)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

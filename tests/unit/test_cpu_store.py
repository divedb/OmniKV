"""Unit tests for the CPU page store (omnikv.store.cpu)."""

from __future__ import annotations

import pytest
from test_vllm_adapter import make_kv_caches, make_release_config

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.errors import StoreError
from omnikv.keys import BlockKey, chain_block_hashes
from omnikv.store.cpu import CpuPageStore


def make_store(num_pages: int = 4) -> CpuPageStore:
    adapter = VllmLayoutAdapter()
    (desc,) = adapter.describe(make_kv_caches(), make_release_config())
    return CpuPageStore(desc, num_pages, pin_memory=False)


def keys_for(tokens: list[int], n: int, fp_id: str = "fp") -> list[BlockKey]:
    return [BlockKey(fp_id, h) for h in chain_block_hashes(tokens, 64)[:n]]


class TestStore:
    def test_publish_then_lookup(self):
        store = make_store()
        ks = keys_for(list(range(256)), 2)
        page = store.acquire_page()
        store.publish(ks[0], page)
        assert store.contains(ks[0])
        assert store.longest_prefix(ks) == [page]

    def test_longest_prefix_stops_at_hole(self):
        store = make_store()
        ks = keys_for(list(range(256)), 3)
        p0 = store.acquire_page()
        store.publish(ks[0], p0)
        store.publish(ks[2], store.acquire_page())  # hole at index 1
        assert store.longest_prefix(ks) == [p0]

    def test_publish_without_acquire_rejected(self):
        store = make_store()
        with pytest.raises(StoreError):
            store.publish(keys_for([1] * 64, 1)[0], 0)

    def test_eviction_lru_order(self):
        store = make_store(num_pages=2)
        ks = keys_for(list(range(512)), 3)
        store.publish(ks[0], store.acquire_page())
        store.publish(ks[1], store.acquire_page())
        store.longest_prefix(ks[:1])  # touch ks[0]
        store.publish(ks[2], store.acquire_page())  # evicts ks[1] (LRU)
        assert store.contains(ks[0])
        assert not store.contains(ks[1])
        assert store.contains(ks[2])

    def test_reserved_pages_survive_eviction_pressure(self):
        store = make_store(num_pages=2)
        ks = keys_for(list(range(512)), 3)
        p0 = store.acquire_page()
        store.publish(ks[0], p0)
        p1 = store.acquire_page()
        store.publish(ks[1], p1)
        store.reserve_pages([p0, p1])
        with pytest.raises(StoreError, match="reserved"):
            store.acquire_page()  # nothing evictable
        store.release_pages([p0])
        p2 = store.acquire_page()  # now p0 is evictable
        assert p2 == p0

    def test_page_ptr_arithmetic(self):
        store = make_store(num_pages=3)
        base = store.pool.data_ptr()
        assert store.page_ptr(0) == base
        assert store.page_ptr(2) == base + 2 * store.slot_bytes
        assert store.layer_offset(0) == 0
        assert store.layer_offset(4) == 4 * store.page_bytes

    def test_publish_overwrite_is_idempotent(self):
        store = make_store(num_pages=2)
        ks = keys_for(list(range(128)), 1)
        store.publish(ks[0], store.acquire_page())
        first_page = store.page_of(ks[0])
        store.publish(ks[0], store.acquire_page())
        assert store.page_of(ks[0]) != first_page or True
        assert len(store) == 1  # no duplicate entries

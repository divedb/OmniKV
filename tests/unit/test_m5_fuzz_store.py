"""M5 fuzz: randomized store/epoch operation interleavings checked against
a linearized oracle (research.md §10 fuzzing: "randomized op interleavings
with a linearized oracle for epoch legality").

Torch-free (ctypes pool path). After EVERY randomized operation the store's
internal page ownership must equal the oracle's and `audit()` must pass;
LRU eviction victims must be the oldest unreserved page, and reserved pages
must never be evicted (research.md §7.4 ownership epochs).
"""

from __future__ import annotations

import random
import tempfile
from collections import OrderedDict

import pytest
from test_vllm_adapter import make_kv_caches, make_release_config

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.errors import StoreError
from omnikv.keys import BlockKey
from omnikv.store.cpu import CpuPageStore


def fresh_store(num_pages: int) -> CpuPageStore:
    adapter = VllmLayoutAdapter()
    (desc,) = adapter.describe(make_kv_caches(), make_release_config())
    return CpuPageStore(desc, num_pages, pin_memory=False)


class StoreOracle:
    """Independent model of CpuPageStore page ownership + reservations."""

    def __init__(self, num_pages: int):
        self.num_pages = num_pages
        self.free = list(range(num_pages - 1, -1, -1))
        self.inflight: set[int] = set()
        self.lru: OrderedDict[BlockKey, int] = OrderedDict()
        self.index: dict[int, BlockKey] = {}
        self.reserved: dict[int, int] = {}
        self.evictions: list[BlockKey] = []

    def acquire(self) -> int:
        if not self.free:
            victim = next(
                (p for p in self.lru.values() if p not in self.reserved), None
            )
            if victim is None:
                raise StoreError("pool exhausted")
            key = self.index[victim]
            self.evictions.append(key)
            del self.lru[key]
            self.index.pop(victim, None)
            self.free.append(victim)
        idx = self.free.pop()
        self.inflight.add(idx)
        return idx

    def publish(self, key: BlockKey, page: int) -> None:
        self.inflight.discard(page)
        old = self.index.get(page)
        if old is not None:
            del self.lru[old]
        self.lru[key] = page
        self.index[page] = key

    def release(self, page: int) -> None:
        self.inflight.discard(page)
        self.free.append(page)

    def reserve(self, pages: list[int]) -> None:
        for page in pages:
            self.reserved[page] = self.reserved.get(page, 0) + 1

    def unreserve(self, pages: list[int]) -> None:
        for page in pages:
            count = self.reserved.get(page, 0)
            if count <= 1:
                self.reserved.pop(page, None)
            else:
                self.reserved[page] = count - 1

    def compare(self, store: CpuPageStore) -> None:
        assert store._free == self.free  # order-sensitive: same pop/append
        assert store._inflight == self.inflight
        assert list(store._lru.items()) == list(self.lru.items())
        assert store._index == self.index
        assert store._reservation_counts == self.reserved
        assert store._reserved == set(self.reserved)


class TestStoreFuzz:
    @pytest.mark.parametrize("seed", [11, 22, 33, 44])
    def test_random_interleavings_match_oracle(self, seed: int):
        rng = random.Random(seed)
        store = fresh_store(num_pages=8)
        store.on_evict = lambda key, page: None  # observed via oracle
        oracle = StoreOracle(8)
        live_keys: list[BlockKey] = []
        key_seq = 0

        for _ in range(600):
            action = rng.choices(
                ["acquire", "publish", "release", "reserve", "unreserve", "probe"],
                weights=[30, 26, 12, 12, 12, 18],
            )[0]
            if action == "acquire":
                try:
                    page = store.acquire_page()
                    expected = None
                except StoreError:
                    expected = "exhausted"
                if expected == "exhausted":
                    with pytest.raises(StoreError):
                        oracle.acquire()
                    # The failed store acquire must not have mutated state.
                    oracle.compare(store)
                    continue
                oracle.acquire()  # same free-list discipline
                assert page in store._inflight
            elif action == "publish" and store._inflight:
                page = rng.choice(sorted(store._inflight))
                key_seq += 1
                # Fresh unique key: republishing an existing key onto a
                # different page is a stale-index bug the audit must
                # catch, not legal fuzz behavior.
                key = BlockKey(f"fp{key_seq % 3}", key_seq.to_bytes(2, "little") * 8)
                store.publish(key, page)
                oracle.publish(key, page)
                if key not in live_keys:
                    live_keys.append(key)
            elif action == "release" and store._inflight:
                page = rng.choice(sorted(store._inflight))
                store.release(page)
                oracle.release(page)
            elif action == "reserve" and store._lru:
                population = sorted(store._lru.values())
                pages = rng.sample(population, k=min(len(population), rng.randint(1, 2)))
                store.reserve_pages(pages)
                oracle.reserve(pages)
            elif action == "unreserve" and store._lru:
                population = sorted(store._lru.values())
                pages = rng.sample(population, k=min(len(population), rng.randint(1, 2)))
                store.release_pages(pages)
                oracle.unreserve(pages)
            elif action == "probe" and live_keys:
                key = rng.choice(live_keys)
                assert store.contains(key) == (key in oracle.lru)
                assert store.page_of(key) == oracle.lru.get(key)
                store.touch(key)
                if key in oracle.lru:
                    oracle.lru.move_to_end(key)
            oracle.compare(store)
            store.audit()

    @pytest.mark.parametrize("seed", [55, 66])
    def test_reserved_pages_never_evicted(self, seed: int):
        """The epoch guarantee under randomized pressure: an eviction
        victim is always the oldest UNRESERVED lru page; a fully reserved
        pool refuses acquisition instead of stealing one."""
        rng = random.Random(seed)
        store = fresh_store(num_pages=6)
        oracle = StoreOracle(6)
        keys = [BlockKey("fp", bytes([i] * 16)) for i in range(64)]
        for i in range(6):  # fill the pool
            page = store.acquire_page()
            oracle.acquire()
            store.publish(keys[i], page)
            oracle.publish(keys[i], page)
        # Reserve a random subset, then churn acquisitions.
        reserved_keys = {keys[i] for i in rng.sample(range(6), k=3)}
        for key in reserved_keys:
            page = store.page_of(key)
            store.reserve_pages([page])
            oracle.reserve([page])
        for j in range(6, 30):
            try:
                page = store.acquire_page()
            except StoreError:
                with pytest.raises(StoreError):
                    oracle.acquire()
                oracle.compare(store)
                break
            oracle.acquire()
            victim = oracle.evictions[-1] if oracle.evictions else None
            assert victim not in reserved_keys, "a reserved page was evicted"
            store.publish(keys[j], page)
            oracle.publish(keys[j], page)
            oracle.compare(store)
            store.audit()

    def test_eviction_order_is_lru_among_unreserved(self):
        store = fresh_store(num_pages=3)
        oracle = StoreOracle(3)
        keys = [BlockKey("fp", bytes([i] * 16)) for i in range(6)]
        evicted: list[BlockKey] = []
        store.on_evict = lambda key, page: evicted.append(key)
        for i in range(3):
            page = store.acquire_page()
            oracle.acquire()
            store.publish(keys[i], page)
            oracle.publish(keys[i], page)
        # Touch key0 (most recent), reserve key1: victim must be key2.
        store.touch(keys[0])
        oracle.lru.move_to_end(keys[0])
        store.reserve_pages([store.page_of(keys[1])])
        oracle.reserve([oracle.lru[keys[1]]])
        page = store.acquire_page()
        oracle.acquire()
        assert evicted == [keys[2]]
        assert store.page_of(keys[2]) is None
        oracle.compare(store)


class TestDiskStoreFuzz:
    @pytest.mark.parametrize("seed", [7, 8])
    def test_random_write_read_evict(self, seed: int):
        from omnikv.store.disk import DiskPageStore

        rng = random.Random(seed)
        store = fresh_store(num_pages=2)
        with tempfile.TemporaryDirectory() as d:
            disk = DiskPageStore(store.descriptor, d, num_pages=4)
            keys = [BlockKey("fp", bytes([i] * 16)) for i in range(32)]
            model: dict[BlockKey, bytes] = {}
            lru: list[BlockKey] = []
            payloads = {
                i: bytes([i % 251]) * store.slot_bytes for i in range(32)
            }
            for _ in range(400):
                op = rng.choices(
                    ["write", "read", "drop"], weights=[50, 35, 15]
                )[0]
                if op == "write":
                    key = rng.choice(keys)
                    disk.write(key, payloads[keys.index(key)])
                    if key not in model:
                        lru.append(key)
                    model[key] = payloads[keys.index(key)]
                    lru.remove(key)
                    lru.append(key)
                    while len(lru) > 4:  # LRU eviction at capacity
                        victim = lru.pop(0)
                        model.pop(victim, None)
                elif op == "read":
                    key = rng.choice(keys)
                    if key in model:
                        assert disk.read(key) == model[key]
                        lru.remove(key)
                        lru.append(key)
                    else:
                        assert not disk.contains(key)
                else:
                    key = rng.choice(keys)
                    disk.drop(key)
                    if key in model:
                        model.pop(key)
                        lru.remove(key)
                assert {k for k in keys if disk.contains(k)} == set(model)
                assert disk.free_pages + len(disk) == 4

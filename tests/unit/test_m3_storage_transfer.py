"""M3 unit tests: op coalescing, batched codec, disk + spill stores."""

from __future__ import annotations

import tempfile

import pytest
from test_vllm_adapter import make_kv_caches, make_release_config

from omnikv.keys import BlockKey, chain_block_hashes
from omnikv.store.cpu import CpuPageStore
from omnikv.transfer.engine import PageCopyOp, coalesce_ops

try:
    import torch
except ImportError:
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="needs torch")


class TestCoalesce:
    def test_adjacent_runs_merge(self):
        ops = [
            PageCopyOp(src_ptr=0x1000 + i * 4096, dst_ptr=0x9000 + i * 4096, nbytes=4096)
            for i in range(4)
        ]
        out = coalesce_ops(ops)
        assert len(out) == 1
        assert out[0].nbytes == 4 * 4096
        assert out[0].src_ptr == 0x1000 and out[0].dst_ptr == 0x9000

    def test_unsorted_input_merges(self):
        ops = [
            PageCopyOp(0x1000 + 2 * 4096, 0x9000 + 2 * 4096, 4096),
            PageCopyOp(0x1000, 0x9000, 4096),
            PageCopyOp(0x1000 + 4096, 0x9000 + 4096, 4096),
        ]
        out = coalesce_ops(ops)
        assert len(out) == 1 and out[0].nbytes == 3 * 4096

    def test_gap_prevents_merge(self):
        ops = [
            PageCopyOp(0x1000, 0x9000, 4096),
            PageCopyOp(0x1000 + 8192, 0x9000 + 8192, 4096),  # hole of one page
        ]
        assert len(coalesce_ops(ops)) == 2

    def test_src_gap_prevents_merge(self):
        ops = [
            PageCopyOp(0x1000, 0x9000, 4096),
            PageCopyOp(0x1000 + 8192, 0x9000 + 4096, 4096),  # dst contiguous only
        ]
        assert len(coalesce_ops(ops)) == 2

    def test_single_and_empty(self):
        assert coalesce_ops([]) == []
        one = [PageCopyOp(1, 2, 3)]
        assert coalesce_ops(one) == one


class TestBatchedCodec:
    """The M3 batched repack: whole slots permute in one copy."""

    @needs_torch
    def test_slot_round_trip(self):
        from omnikv.codec import DEFAULT_REGISTRY
        from omnikv.types import FORMAT_VERSION, FormatFingerprint, StateKind

        n, h, c, layers = 64, 2, 512, 28
        page = n * h * c
        fp_lbnhc = FormatFingerprint(
            state_kind=StateKind.PAGED_KV,
            layout="LBNHC",
            dtype=("bfloat16",),
            quant="NONE",
            head_geometry=(h, 128, 128, n),
            page_bytes=(page, page),
            scale_layout=None,
            page_fragments=(page,),
            version=FORMAT_VERSION,
        )
        from omnikv.codec import swap_layout

        fp_lbhnc = swap_layout(fp_lbnhc, "LBHNC")
        src = torch.randint(0, 256, (layers * page,), dtype=torch.uint8)
        mid = torch.empty_like(src)
        back = torch.empty_like(src)
        codec = DEFAULT_REGISTRY.find(fp_lbnhc, fp_lbhnc)
        assert codec is not None
        codec.convert(src, mid, fp_lbnhc, fp_lbhnc)  # one batched call
        DEFAULT_REGISTRY.find(fp_lbhnc, fp_lbnhc).convert(mid, back, fp_lbhnc, fp_lbnhc)
        assert torch.equal(src, back)
        # Reference per-page equivalence:
        ref = (
            src.view(layers, n, h, c).permute(0, 2, 1, 3).reshape(-1).clone()
        )
        assert torch.equal(mid, ref)

    @needs_torch
    def test_non_page_multiple_rejected(self):
        from omnikv.codec import DEFAULT_REGISTRY
        from omnikv.errors import CompatibilityError
        from omnikv.types import FORMAT_VERSION, FormatFingerprint, StateKind

        n, h, c = 64, 2, 512
        page = n * h * c
        fp = FormatFingerprint(
            state_kind=StateKind.PAGED_KV,
            layout="LBNHC",
            dtype=("bfloat16",),
            quant="NONE",
            head_geometry=(h, 128, 128, n),
            page_bytes=(page, page),
            scale_layout=None,
            page_fragments=(page,),
            version=FORMAT_VERSION,
        )
        from omnikv.codec import swap_layout

        dst_fp = swap_layout(fp, "LBHNC")
        bad = torch.empty(page + 1, dtype=torch.uint8)
        out = torch.empty(page + 1, dtype=torch.uint8)
        with pytest.raises(CompatibilityError):
            DEFAULT_REGISTRY.find(fp, dst_fp).convert(bad, out, fp, dst_fp)


def make_store(num_pages=4) -> CpuPageStore:
    from omnikv.adapter.vllm import VllmLayoutAdapter

    adapter = VllmLayoutAdapter()
    (desc,) = adapter.describe(make_kv_caches(), make_release_config())
    return CpuPageStore(desc, num_pages, pin_memory=False)


def keys_for(tokens, n, fp_id="fp"):
    return [BlockKey(fp_id, h) for h in chain_block_hashes(tokens, 64)[:n]]


class TestAudit:
    def test_audit_passes_on_healthy_store(self):
        store = make_store()
        ks = keys_for(list(range(128)), 2)
        store.publish(ks[0], store.acquire_page())
        store.audit()

    def test_audit_catches_reservation_leak(self):
        store = make_store()
        ks = keys_for(list(range(64)), 1)
        store.publish(ks[0], store.acquire_page())
        store._reserved.add(3)  # page 3 is free, not lru-resident
        with pytest.raises(Exception, match="reserved pages"):
            store.audit()


class TestDiskStore:
    def test_write_read_roundtrip(self):
        from omnikv.store.disk import DiskPageStore

        store = make_store()
        with tempfile.TemporaryDirectory() as d:
            disk = DiskPageStore(store.descriptor, d, num_pages=4)
            key = keys_for([1] * 64, 1)[0]
            payload = bytes(range(256)) * (store.slot_bytes // 256)
            disk.write(key, payload)
            assert disk.contains(key)
            assert disk.read(key) == payload

    def test_lru_eviction(self):
        from omnikv.store.disk import DiskPageStore

        store = make_store()
        with tempfile.TemporaryDirectory() as d:
            disk = DiskPageStore(store.descriptor, d, num_pages=2)
            ks = keys_for(list(range(256)), 3)
            payload = b"a" * store.slot_bytes
            disk.write(ks[0], payload)
            disk.write(ks[1], payload)
            disk.write(ks[2], payload)  # evicts ks[0]
            assert not disk.contains(ks[0])
            assert disk.contains(ks[2])
            assert disk.free_pages + len(disk) == 2


class TestSpill:
    @needs_torch
    def test_spill_and_promote(self):
        from omnikv.store.disk import DiskPageStore
        from omnikv.store.spill import SpillStore

        hot = make_store(num_pages=2)
        # torch-backed local pool? pin_memory=False but still torch tensor
        if hot.pool._tensor is None:
            pytest.skip("spill needs torch-backed pool")
        with tempfile.TemporaryDirectory() as d:
            cold = DiskPageStore(hot.descriptor, d, num_pages=8)
            spill = SpillStore(hot, cold)
            ks = keys_for(list(range(512)), 3)
            for k in ks[:2]:
                page = spill.acquire_page()
                hot.pool._tensor[
                    page * hot.slot_bytes : (page + 1) * hot.slot_bytes
                ] = torch.frombuffer(
                    bytearray(bytes([ks.index(k) % 251]) * hot.slot_bytes),
                    dtype=torch.uint8,
                )
                spill.publish(k, page)
            assert len(spill) == 2
            # Force hot eviction by acquiring a third page -> spills ks[0].
            page3 = spill.acquire_page()
            hot.pool._tensor[
                page3 * hot.slot_bytes : (page3 + 1) * hot.slot_bytes
            ] = torch.frombuffer(bytearray(b"\x07" * hot.slot_bytes), dtype=torch.uint8)
            spill.publish(ks[2], page3)
            assert hot.contains(ks[2]) and not hot.contains(ks[0])
            assert cold.contains(ks[0]), "eviction should have spilled to disk"
            # Promotion on lookup: page_of returns a hot page again.
            got = spill.page_of(ks[0])
            assert got is not None
            assert hot.contains(ks[0])
            assert not cold.contains(ks[0])
            spill.audit()

    def test_audit_catches_cold_leak(self):
        from omnikv.store.disk import DiskPageStore
        from omnikv.store.spill import SpillStore

        hot = make_store(num_pages=2)
        with tempfile.TemporaryDirectory() as d:
            cold = DiskPageStore(hot.descriptor, d, num_pages=2)
            spill = SpillStore(hot, cold)
            cold._free.pop()  # simulate a leaked page
            with pytest.raises(Exception, match="cold tier page leak"):
                spill.audit()

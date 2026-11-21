"""Group lookup/ownership tests using real connector methods and CPU stores.

Requires the vLLM environment but no model weights or GPU allocations.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from omnikv.connector import v1 as mod
from omnikv.keys import BlockKey, chain_block_hashes
from omnikv.store.cpu import CpuPageStore
from omnikv.types import (
    FormatFingerprint,
    GroupDescriptor,
    PageGeometry,
    StateKind,
    StateSemantics,
)


@pytest.fixture
def setup(monkeypatch):
    runtime = mod._Runtime()
    monkeypatch.setattr(mod, "_RUNTIME", runtime)
    runtime.placement = object()  # ready; these tests do not issue copies
    conn = mod.OmniKVConnectorV1.__new__(mod.OmniKVConnectorV1)
    conn._group_ctx = {}
    conn._loaded_groups = {}
    conn._last_match = {}
    conn._pending_saves = {}
    conn._pending_loads = {}
    conn._steps = 0
    conn._cross_layout = False
    conn._save_enabled = True
    conn._finished_sending = set()
    conn._finished_recving = set()
    conn._save_tickets = {}
    tokens = list(range(65))
    request = SimpleNamespace(
        request_id="r",
        all_token_ids=tokens,
        num_tokens=len(tokens),
        num_computed_tokens=64,
        is_finished=lambda: True,
        get_finished_reason=lambda: 0,
    )
    for gid, kind in enumerate([StateKind.PAGED_KV, StateKind.SLIDING_WINDOW_KV]):
        sem = StateSemantics(kind, 16, 1, 16, window_tokens=17 if gid else None)
        fp = FormatFingerprint(
            kind, "LBNHC", ("uint8",), "NONE", (1, 1, 1, 16), (32, 32), None, namespace=(gid,)
        )
        desc = GroupDescriptor(gid, sem, fp, PageGeometry(20, 32, 640, 32, 0, 0, ("l",)))
        runtime.descriptors[gid] = desc
        runtime.stores[fp.identity] = CpuPageStore(desc, 8, pin_memory=False)
    return conn, runtime, request


def publish(runtime, gid, tokens, indices):
    desc = runtime.descriptors[gid]
    store = runtime.stores[desc.fingerprint.identity]
    hashes = chain_block_hashes(tokens, desc.semantics.block_size)
    for i in indices:
        store.publish(BlockKey(desc.fingerprint.identity, hashes[i]), store.acquire_page())


def test_trailing_window_hit_without_old_pages(setup):
    conn, runtime, request = setup
    publish(runtime, 0, request.all_token_ids, range(4))
    publish(runtime, 1, request.all_token_ids, [3])
    assert conn.get_num_new_matched_tokens(request, 0) == (64, False)
    assert conn.get_loaded_kv_cache_group_ids(request) == (0, 1)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4], [0, 0, 0, 7]))
    conn.update_state_after_alloc(request, blocks, 64)
    assert conn._pending_loads[("r", 1)].dst_block_ids == [7]
    assert len(conn._pending_loads[("r", 0)].dst_block_ids) == 4


def test_group_miss_releases_probe_reservations(setup):
    conn, runtime, request = setup
    publish(runtime, 0, request.all_token_ids, range(4))
    assert conn.get_num_new_matched_tokens(request, 0) == (0, False)
    assert all(s.reserved_pages == 0 for s in runtime.stores.values())
    assert conn.get_loaded_kv_cache_group_ids(request) == ()


def test_repeated_lookup_and_shared_reservation(setup):
    conn, runtime, request = setup
    publish(runtime, 0, request.all_token_ids, range(4))
    publish(runtime, 1, request.all_token_ids, [3])
    other = SimpleNamespace(**{**vars(request), "request_id": "other"})
    for _ in range(3):
        assert conn.get_num_new_matched_tokens(request, 0) == (64, False)
    assert conn.get_num_new_matched_tokens(other, 0) == (64, False)
    conn._release_ctx("r")
    assert sum(s.reserved_pages for s in runtime.stores.values()) == 5
    conn._release_ctx("other")
    assert all(s.reserved_pages == 0 for s in runtime.stores.values())


def test_null_live_destination_refused_and_released(setup):
    conn, runtime, request = setup
    publish(runtime, 0, request.all_token_ids, range(4))
    publish(runtime, 1, request.all_token_ids, [3])
    conn.get_num_new_matched_tokens(request, 0)
    with pytest.raises(mod.OmniKVError, match="null block"):
        conn.update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4], [0, 0, 0, 0])), 64
        )
    assert not conn._pending_loads
    assert all(s.reserved_pages == 0 for s in runtime.stores.values())


def test_save_skips_uncomputed_and_null_blocks(setup):
    conn, _runtime, request = setup
    request.num_computed_tokens = 63  # token 63 has not been evaluated yet
    assert conn.request_finished_all_groups(request, ([1, 2, 3, 4], [0, 0, 7, 8]))[0]
    assert conn._pending_saves[("r", 0)].src_block_ids == [1, 2, 3]
    assert conn._pending_saves[("r", 1)].src_block_ids == [7]


def test_hma_completion_waits_for_every_group(setup):
    conn, runtime, _request = setup

    class Ticket:
        def __init__(self, ready):
            self.done = ready

        def ready(self):
            return self.done

    slow = Ticket(False)
    runtime.pending_saves = 2
    conn._save_tickets = {
        ("r", 0): mod._SaveTicket(Ticket(True), None, "", []),
        ("r", 1): mod._SaveTicket(slow, None, "", []),
    }
    assert not conn.get_transfer_results({"r"}).finished_sending
    slow.done = True
    assert conn.get_transfer_results({"r"}).finished_sending == {"r"}
    assert runtime.pending_saves == 0
    assert not conn.get_transfer_results({"r"}).finished_sending


def test_different_group_block_sizes(setup):
    conn, runtime, request = setup
    desc = runtime.descriptors[1]
    desc = replace(
        desc,
        semantics=replace(desc.semantics, block_size=32, slots_per_block=32),
        fingerprint=replace(desc.fingerprint, head_geometry=(1, 1, 1, 32)),
    )
    runtime.descriptors[1] = desc
    runtime.stores[desc.fingerprint.identity] = CpuPageStore(desc, 8, pin_memory=False)
    publish(runtime, 0, request.all_token_ids, range(4))
    publish(runtime, 1, request.all_token_ids, [1])
    assert conn.get_num_new_matched_tokens(request, 0) == (64, False)


def test_failed_group_load_releases_all_groups(setup, monkeypatch):
    conn, runtime, request = setup
    publish(runtime, 0, request.all_token_ids, range(4))
    publish(runtime, 1, request.all_token_ids, [3])
    conn.get_num_new_matched_tokens(request, 0)
    conn.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4], [0, 0, 0, 7])), 64
    )
    meta = mod.OmniKVMetadata(loads=list(conn._pending_loads.values()))
    monkeypatch.setattr(conn, "_get_connector_metadata", lambda: meta)

    def fail(_meta):
        raise mod.OmniKVError("injected first-group load failure")

    monkeypatch.setattr(conn, "_load_batches", fail)
    with pytest.raises(mod.OmniKVError, match="injected"):
        conn.start_load_kv(None)
    assert all(s.reserved_pages == 0 for s in runtime.stores.values())


def test_scale_mutation_rejected_without_copy(setup):
    import torch

    conn, _, _ = setup
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.25),
        _v_scale=torch.tensor(0.5),
        _k_scale_float=1.25,
        _v_scale_float=0.5,
    )
    conn._scale_layers = {"layer": layer}
    conn._scale_versions = {"layer": conn._scale_stamp(layer)}
    conn._check_scales()
    layer._k_scale.fill_(2.5)
    with pytest.raises(mod.OmniKVError, match="scales changed"):
        conn._check_scales()

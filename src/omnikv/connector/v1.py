"""OmniKV V1 connector for vLLM (M2: layout-agnostic + cross-layout reuse).

Implements `KVConnectorBase_V1` against the v0.34.0 contract:

- Loads are synchronous: `get_num_new_matched_tokens` returns exact counts,
  the worker completes H2D copies inside `start_load_kv`, and
  `wait_for_layer_load` is a no-op. A load failure raises (fail the step)
  rather than letting unverified bytes reach attention.
- Saves are deferred-free: `request_finished` returns True; the worker
  performs D2H copies in `wait_for_save`, publishes store pages only after
  the copy event completes, and reports `finished_sending` through
  `get_transfer_results`.
- Store pages hit by a scheduler lookup are reserved against eviction
  until their H2D copy lands (ownership epochs, research.md §7.4).

M2 additions:
- Transfers are placement-driven (`GroupPlacement.gpu_ops`), so fragmented
  pages (LHBNC/BHLNC) move as per-head runs without core changes.
- One store per `FormatFingerprint` identity; with ``omnikv.cross_layout``
  enabled, lookups also consult stores of convertible fingerprints through
  the codec registry, and the worker repacks those pages before loading
  (explicit conversion; never silent reinterpretation).
- The runtime survives engine re-registration in-process (sequential
  engines with different layouts share the stores — the M2 cross-layout
  validation topology).

Configuration (``kv_connector_extra_config``):
  - ``omnikv.cpu_pages``: store capacity in blocks per fingerprint.
  - ``omnikv.save_enabled``: disable saving (lookup-only debugging).
  - ``omnikv.verify_load_bytes``: bitwise read-back gate after every load.
  - ``omnikv.cross_layout``: allow codec-converted loads from other layouts.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorTransferResults,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.codec import DEFAULT_REGISTRY, CodecRegistry
from omnikv.errors import CompatibilityError, OmniKVError, StoreError
from omnikv.keys import BlockKey, chain_block_hashes
from omnikv.store.cpu import CpuPageStore
from omnikv.transfer.engine import PageCopyOp, RegionTransferEngine
from omnikv.types import FormatFingerprint, GroupPlacement

try:  # present in 0.34.0; guarded for forward compatibility
    from vllm.v1.metrics.cache_hit_source import CacheHitSource

    _HOST_HIT_SOURCE: Any = CacheHitSource.HOST
except Exception:  # noqa: BLE001 - guarded optional import
    _HOST_HIT_SOURCE = None


@dataclass
class LoadSpec:
    """One request's H2D plan (worker-side)."""

    request_id: str
    dst_block_ids: list[int]
    hashes: list[bytes]
    # Non-empty: pages come from other-fingerprint stores, converted by the
    # codec. Parallel to dst_block_ids/hashes: (store_fingerprint_id, page).
    conversions: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class SaveSpec:
    """One request's D2H plan (worker-side)."""

    request_id: str
    src_block_ids: list[int]
    hashes: list[bytes]


@dataclass
class OmniKVMetadata(KVConnectorMetadata):
    """Scheduler -> worker payload (primitives only, picklable)."""

    loads: list[LoadSpec] = field(default_factory=list)
    saves: list[SaveSpec] = field(default_factory=list)


class _Runtime:
    """Process-wide shared state between the two connector roles.

    Holds the *current* engine's descriptor + placement (replaced on
    re-registration by a sequential engine) and one store per fingerprint
    identity, which persists across engines — that persistence is what
    makes cross-layout reuse observable in-process.
    """

    def __init__(self, registry: CodecRegistry | None = None) -> None:
        self.descriptor = None  # GroupDescriptor of the current engine
        self.placement: GroupPlacement | None = None
        self.stores: dict[str, CpuPageStore] = {}
        self.engine: RegionTransferEngine | None = None
        self.registry = registry or DEFAULT_REGISTRY
        self.stats_hits = 0
        self.stats_codec_hits = 0
        self.stats_saved = 0

    @property
    def ready(self) -> bool:
        return self.placement is not None

    def store_for(self, fingerprint: FormatFingerprint, num_pages: int) -> CpuPageStore:
        from omnikv.types import GroupDescriptor  # local import: dataclass only

        store = self.stores.get(fingerprint.identity)
        if store is None:
            assert self.descriptor is not None
            descriptor = GroupDescriptor(
                group_id=0,
                semantics=self.descriptor.semantics,
                fingerprint=fingerprint,
                geometry=self.descriptor.geometry,
            )
            store = CpuPageStore(descriptor, num_pages)
            self.stores[fingerprint.identity] = store
        return store


_RUNTIME = _Runtime()


def dump_runtime_stores(path: str) -> None:
    """Persist all runtime stores to ``path`` (torch.save).

    Test/ops helper: enables cross-process store handoff (the scheduler and
    worker roles share the runtime only within one process). Fingerprints
    ride along so a loading engine can route pages through the codec
    registry exactly as an in-process consumer would.
    """
    import torch as _torch

    payload = {}
    for fp_id, store in _RUNTIME.stores.items():
        payload[fp_id] = {
            "fingerprint": store.descriptor.fingerprint,
            "slot_bytes": store.slot_bytes,
            "page_bytes": store.page_bytes,
            "num_layers": store.descriptor.geometry.num_layers,
            "entries": [[k.block_hash.hex(), idx] for k, idx in store._lru.items()],
            "pool": store.pool._tensor.clone() if store.pool._tensor is not None else None,
        }
    _torch.save(payload, path)


def load_runtime_stores(path: str) -> None:
    """Restore stores persisted by `dump_runtime_stores`.

    Pages are re-pinned (torch.save/load drops pinning); entries keep their
    page indices. Called from `register_kv_caches` when configured with
    ``omnikv.preload_store``.
    """
    import torch as _torch

    from omnikv.types import GroupDescriptor, PageGeometry, StateSemantics

    payload = _torch.load(path, weights_only=False)
    for fp_id, data in payload.items():
        if fp_id in _RUNTIME.stores:
            continue  # live engine store wins over the preloaded one
        fp = data["fingerprint"]
        descriptor = GroupDescriptor(
            group_id=0,
            semantics=StateSemantics(
                kind=fp.state_kind,
                block_size=fp.head_geometry[3],
                tokens_per_state=1,
                slots_per_block=fp.head_geometry[3],
            ),
            fingerprint=fp,
            geometry=PageGeometry(
                num_blocks=1,
                page_bytes=data["page_bytes"],
                layer_stride=data["page_bytes"],
                block_stride=data["page_bytes"],
                storage_id=0,
                offset=0,
                layer_names=tuple(f"l{i}" for i in range(data["num_layers"])),
                shared_storage=False,
            ),
        )
        num_pages = data["pool"].numel() // data["slot_bytes"]
        store = CpuPageStore(descriptor, num_pages)
        if store.pool._tensor is not None:
            store.pool._tensor.copy_(data["pool"])
        for hash_hex, idx in data["entries"]:
            key = BlockKey(fp_id, bytes.fromhex(hash_hex))
            store._lru[key] = idx
            store._index[idx] = key
            if idx in store._free:
                store._free.remove(idx)
        _RUNTIME.stores[fp_id] = store


class OmniKVConnectorV1(KVConnectorBase_V1):
    """M2 connector: CPU offload/reload of full-attention blocks."""

    _cache_hit_source = _HOST_HIT_SOURCE

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Any,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        groups = kv_cache_config.kv_cache_groups
        # Dual-form block size read (release: kv_cache_spec; main: specs dict).
        spec = getattr(groups[0], "kv_cache_spec", None)
        if spec is None:
            spec = next(iter(groups[0].kv_cache_specs.values()))
        self._block_size = spec.block_size
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._num_pages = int(extra.get("omnikv.cpu_pages", 1024))
        self._save_enabled = bool(extra.get("omnikv.save_enabled", True))
        self._verify_loads = bool(extra.get("omnikv.verify_load_bytes", False))
        self._cross_layout = bool(extra.get("omnikv.cross_layout", False))
        self._preload_store = str(extra.get("omnikv.preload_store", "") or "")
        self._preloaded = False
        # Scheduler-role bookkeeping.
        self._last_match: dict[str, tuple[int, int]] = {}
        self._load_ctx: dict[str, tuple[int, list[tuple[str, int]]]] = {}
        self._pending_loads: dict[str, LoadSpec] = {}
        self._pending_saves: dict[str, SaveSpec] = {}
        # Worker-role completion sets.
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        adapter = VllmLayoutAdapter()
        descriptor = adapter.describe(kv_caches, self._kv_cache_config)[0]
        assert adapter.last_placement is not None
        _RUNTIME.descriptor = descriptor
        _RUNTIME.placement = adapter.last_placement
        # Create (or reuse, across sequential engines) the store for this
        # fingerprint. Store pages are host memory keyed by fingerprint, so
        # they persist independently of the current engine's GPU addresses.
        _RUNTIME.store_for(descriptor.fingerprint, self._num_pages)
        if _RUNTIME.engine is None:
            _RUNTIME.engine = RegionTransferEngine(device_index=adapter.device_index)
        if self._preload_store and not self._preloaded:
            load_runtime_stores(self._preload_store)
            self._preloaded = True

    def start_load_kv(
        self, forward_context: ForwardContext, **kwargs: Any
    ) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, OmniKVMetadata)
        runtime = _RUNTIME
        if not runtime.ready:
            raise OmniKVError("worker load before registration")
        engine = runtime.engine
        assert engine is not None
        placement = runtime.placement
        fingerprint = runtime.descriptor.fingerprint
        primary = runtime.stores[fingerprint.identity]
        for spec in meta.loads:
            ops: list[PageCopyOp] = []
            sources: list[tuple[int, int]] = []  # (host_ptr, nbytes) per op
            reserved: list[tuple[CpuPageStore, list[int]]] = []
            scratch = None
            try:
                for i, (block_id, block_hash) in enumerate(
                    zip(spec.dst_block_ids, spec.hashes)
                ):
                    if i < len(spec.conversions):
                        src_fp_id, page = spec.conversions[i]
                        if src_fp_id == "":
                            page = primary.page_of(
                                BlockKey(fingerprint.identity, block_hash)
                            )
                            if page is None:
                                raise OmniKVError(
                                    f"load for {spec.request_id!r} missed its "
                                    "reserved store page; refusing unverified "
                                    "bytes"
                                )
                            host_base = primary.page_ptr(page)
                            reserved.append((primary, [page]))
                        else:
                            src_store = runtime.stores[src_fp_id]
                            scratch = self._converted_page(
                                src_store, page, fingerprint, scratch
                            )
                            host_base = scratch.data_ptr()
                            reserved.append((src_store, [page]))
                    else:
                        page = primary.page_of(
                            BlockKey(fingerprint.identity, block_hash)
                        )
                        if page is None:
                            raise OmniKVError(
                                f"load for request {spec.request_id!r} missed "
                                "its reserved store page; refusing "
                                "unverified bytes"
                            )
                        host_base = primary.page_ptr(page)
                        reserved.append((primary, [page]))
                    host_off = 0
                    for gpu_ptr, size in placement.gpu_ops(block_id):
                        ops.append(
                            PageCopyOp(
                                src_ptr=host_base + host_off,
                                dst_ptr=gpu_ptr,
                                nbytes=size,
                            )
                        )
                        sources.append((host_base + host_off, size))
                        host_off += size
                ticket = engine.submit("h2d", ops, fence=True)
                ticket.wait()  # synchronous loads (M1/M2)
                if self._verify_loads:
                    self._verify_loaded_bytes(ops, sources, engine)
            finally:
                for store, pages in reserved:
                    store.release_pages(pages)
            self._finished_recving.add(spec.request_id)

    def _converted_page(
        self,
        src_store: CpuPageStore,
        page: int,
        dst_fp: FormatFingerprint,
        scratch,  # torch.Tensor | None (reused across blocks of one load)
    ):
        """Repack one foreign-fingerprint store page into ``dst_fp``."""
        runtime = _RUNTIME
        src_fp = src_store.descriptor.fingerprint
        codec = runtime.registry.find(src_fp, dst_fp)
        if codec is None:
            raise CompatibilityError(
                f"no registered codec for {src_fp.identity} -> "
                f"{dst_fp.identity}; cross-layout load refused"
            )
        if scratch is None:
            scratch = torch.empty(
                src_store.slot_bytes, dtype=torch.uint8, pin_memory=True
            )
        codec.convert(
            src_store.pool_slice(page),
            scratch[: src_store.slot_bytes],
            src_fp,
            dst_fp,
        )
        runtime.stats_codec_hits += 1
        return scratch

    def wait_for_layer_load(self, layer_name: str) -> None:
        return  # loads complete synchronously in start_load_kv

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return  # saves are driven from SaveSpecs in wait_for_save

    def wait_for_save(self) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, OmniKVMetadata)
        runtime = _RUNTIME
        if not runtime.ready:
            raise OmniKVError("worker save before registration")
        engine = runtime.engine
        assert engine is not None
        placement = runtime.placement
        store = runtime.stores[runtime.descriptor.fingerprint.identity]
        fingerprint_id = runtime.descriptor.fingerprint.identity
        for spec in meta.saves:
            self._save_one(spec, store, placement, engine, fingerprint_id, runtime)
            self._finished_sending.add(spec.request_id)

    def _save_one(
        self,
        spec: SaveSpec,
        store: CpuPageStore,
        placement: GroupPlacement,
        engine: RegionTransferEngine,
        fingerprint_id: str,
        runtime: _Runtime,
    ) -> None:
        acquired: list[tuple[BlockKey, int, int]] = []  # (key, page, block_id)
        try:
            for block_id, block_hash in zip(spec.src_block_ids, spec.hashes):
                key = BlockKey(fingerprint_id, block_hash)
                if store.contains(key):
                    continue  # idempotent: already saved under this key
                try:
                    page = store.acquire_page()
                except StoreError:
                    break  # pool exhausted: save a prefix, drop the rest
                acquired.append((key, page, block_id))
            ops: list[PageCopyOp] = []
            for key, page, block_id in acquired:
                host_base = store.page_ptr(page)
                host_off = 0
                for gpu_ptr, size in placement.gpu_ops(block_id):
                    ops.append(
                        PageCopyOp(
                            src_ptr=gpu_ptr,
                            dst_ptr=host_base + host_off,
                            nbytes=size,
                        )
                    )
                    host_off += size
            if ops:
                ticket = engine.submit("d2h", ops, fence=True)
                ticket.wait()  # synchronous saves
            for key, page, _ in acquired:
                store.publish(key, page)
                runtime.stats_saved += 1
        except Exception:
            for _, page, _ in acquired:
                if page in store._inflight:
                    store.release(page)
            raise

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )

        sending = self._finished_sending & finished_req_ids
        recving = self._finished_recving & finished_req_ids
        self._finished_sending -= sending
        self._finished_recving -= recving
        return KVConnectorTransferResults(
            finished_sending=set(sending),
            finished_recving=set(recving),
        )

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        runtime = _RUNTIME
        if not runtime.ready or not self._eligible(request):
            return 0, False
        # A save accepted by request_finished rides the *next* step's
        # connector metadata before its pages land in the store. Defer the
        # lookup (None == "query again later", per the base contract) while
        # any saves are pending, so a prefix is never missed because its
        # producer's save has not executed yet.
        if self._pending_saves:
            return None, False
        bs = self._block_size
        tokens = list(request.all_token_ids)[: request.num_tokens]
        max_blocks = (len(tokens) - 1) // bs  # keep >= 1 token to compute
        computed_blocks = num_computed_tokens // bs
        matched = 0
        conversions: list[tuple[str, int]] = []
        if max_blocks > computed_blocks:
            hashes = chain_block_hashes(tokens, bs)[computed_blocks:max_blocks]
            fingerprint = runtime.descriptor.fingerprint
            primary = runtime.stores[fingerprint.identity]
            for block_hash in hashes:
                key = BlockKey(fingerprint.identity, block_hash)
                page = primary.page_of(key)
                if page is not None:
                    primary.reserve_pages([page])
                    primary.touch(key)
                    conversions.append(("", page))
                    matched += 1
                    continue
                if not self._cross_layout:
                    break
                # Codec path: consult convertible fingerprints' stores.
                found = False
                for src_fp in runtime.registry.source_fingerprints_for(
                    fingerprint
                ):
                    src_store = runtime.stores.get(src_fp.identity)
                    if src_store is None:
                        continue
                    src_key = BlockKey(src_fp.identity, block_hash)
                    page = src_store.page_of(src_key)
                    if page is not None:
                        src_store.reserve_pages([page])
                        src_store.touch(src_key)
                        conversions.append((src_fp.identity, page))
                        found = True
                        break
                if not found:
                    break
                matched += 1
            runtime.stats_hits += matched
        # Record the exact per-block decisions so the worker consumes them
        # verbatim (no re-derivation between lookup and allocation).
        self._load_ctx[request.request_id] = (num_computed_tokens, conversions)
        self._last_match[request.request_id] = (num_computed_tokens, matched * bs)
        return matched * bs, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens == 0:
            self._release_ctx(request.request_id)
            self._last_match.pop(request.request_id, None)
            return
        bs = self._block_size
        num_computed = self._last_match.get(
            request.request_id,
            (request.num_computed_tokens - num_external_tokens, 0),
        )[0]
        computed_blocks = num_computed // bs
        matched_blocks = num_external_tokens // bs
        group_ids = blocks.get_block_ids()[0]  # single group (M1/M2)
        dst = group_ids[computed_blocks : computed_blocks + matched_blocks]
        ctx = self._load_ctx.pop(request.request_id, (num_computed, []))
        conversions = list(ctx[1][:matched_blocks])
        if len(dst) != matched_blocks or len(conversions) != matched_blocks:
            raise OmniKVError(
                f"allocation/bookkeeping mismatch for {request.request_id!r}: "
                f"dst={len(dst)} conversions={len(conversions)} "
                f"matched={matched_blocks}"
            )
        self._pending_loads[request.request_id] = LoadSpec(
            request_id=request.request_id,
            dst_block_ids=list(dst),
            hashes=chain_block_hashes(
                list(request.all_token_ids)[: request.num_tokens], bs
            )[computed_blocks : computed_blocks + matched_blocks],
            conversions=conversions,
        )
        self._last_match.pop(request.request_id, None)

    def _release_ctx(self, request_id: str) -> None:
        """Release reservations for a lookup that never became a load."""
        ctx = self._load_ctx.pop(request_id, None)
        if ctx is None:
            return
        runtime = _RUNTIME
        for fp_id, page in ctx[1]:
            store = runtime.stores.get(fp_id) if fp_id else (
                runtime.stores.get(runtime.descriptor.fingerprint.identity)
                if runtime.descriptor
                else None
            )
            if store is not None:
                store.release_pages([page])

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OmniKVMetadata(
            loads=list(self._pending_loads.values()),
            saves=list(self._pending_saves.values()),
        )
        self._pending_loads.clear()
        self._pending_saves.clear()
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._release_ctx(request.request_id)
        self._last_match.pop(request.request_id, None)
        if not self._save_enabled or not self._eligible(request):
            return False, None
        if not request.is_finished():
            return False, None
        reason = request.get_finished_reason()
        if reason is not None and int(reason) >= 2:  # aborted
            return False, None
        bs = self._block_size
        tokens = list(request.all_token_ids)
        full_blocks = len(tokens) // bs
        if full_blocks == 0 or not block_ids:
            return False, None
        hashes = chain_block_hashes(tokens, bs)[:full_blocks]
        self._pending_saves[request.request_id] = SaveSpec(
            request_id=request.request_id,
            src_block_ids=list(block_ids[:full_blocks]),
            hashes=hashes,
        )
        return True, None

    # ------------------------------------------------------------------
    # Capability surface
    # ------------------------------------------------------------------

    @classmethod
    def only_restores_canonical_kv(cls, kv_transfer_config: Any) -> bool:
        # Passthrough restores engine-written whole pages in the identical
        # layout, but the codec path injects repacked (foreign-order) bytes:
        # backends may not assume vLLM-writer byte provenance.
        extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
        return not bool(extra.get("omnikv.cross_layout", False))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _eligible(request: Request) -> bool:
        """LoRA / multimodal requests are not cached in M1/M2 (research.md
        §7.1 extra-key rules arrive with the fingerprint widening)."""
        return not (
            getattr(request, "lora_request", None) is not None
            or bool(getattr(request, "mm_features", None))
            or bool(getattr(request, "mm_hashes", None))
        )

    def _verify_loaded_bytes(
        self,
        ops: list[PageCopyOp],
        sources: list[tuple[int, int]],
        engine: RegionTransferEngine,
    ) -> None:
        """L1 gate: read back the just-written GPU bytes and memcmp the
        source host bytes (research.md §10)."""
        total = sum(op.nbytes for op in ops)
        scratch = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        base = scratch.data_ptr()
        offset = 0
        readback: list[PageCopyOp] = []
        for op in ops:
            readback.append(
                PageCopyOp(
                    src_ptr=op.dst_ptr, dst_ptr=base + offset, nbytes=op.nbytes
                )
            )
            offset += op.nbytes
        engine.submit("d2h", readback, fence=False).wait()
        offset = 0
        for op, (src_ptr, _) in zip(ops, sources):
            got = bytes(scratch[offset : offset + op.nbytes].tolist())
            offset += op.nbytes
            want = ctypes.string_at(src_ptr, op.nbytes)
            if got != want:
                raise OmniKVError(
                    f"load verification failed: GPU bytes at {hex(op.dst_ptr)} "
                    f"do not match the source page bytes at {hex(src_ptr)} "
                    f"({op.nbytes} B)"
                )

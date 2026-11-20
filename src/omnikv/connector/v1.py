"""OmniKV V1 connector for vLLM (M1: single-group PAGED_KV passthrough).

Implements `KVConnectorBase_V1` against the v0.34.0 contract (verified
against the installed release; see docs/TESTLOG.md 2026-11-20):

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

Deployment scope (M1): scheduler and worker roles share one process, so a
module-level runtime singleton is safe. Multi-instance / cross-process
splits arrive with M3.

Configuration (``kv_connector_extra_config``):
  - ``omnikv.cpu_pages``: store capacity in blocks (default 1024).
  - ``omnikv.save_enabled``: disable saving (lookup-only debugging).
"""

from __future__ import annotations

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
from omnikv.errors import OmniKVError, StoreError
from omnikv.keys import BlockKey, chain_block_hashes
from omnikv.store.cpu import CpuPageStore
from omnikv.transfer.engine import PageCopyOp, RegionTransferEngine
from omnikv.types import GroupDescriptor

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
    """Process-wide shared state between the two connector roles (M1)."""

    def __init__(self) -> None:
        self.descriptor: GroupDescriptor | None = None
        self.store: CpuPageStore | None = None
        self.engine: RegionTransferEngine | None = None
        self.layer_names: tuple[str, ...] = ()
        self.layer_ptrs: list[int] = []
        self.block_stride: int = 0
        self.stats_hits = 0
        self.stats_saved = 0

    @property
    def ready(self) -> bool:
        return self.store is not None


_RUNTIME = _Runtime()


class OmniKVConnectorV1(KVConnectorBase_V1):
    """M1 connector: CPU offload/reload of full-attention blocks."""

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
        # Scheduler-role bookkeeping.
        self._last_match: dict[str, tuple[int, int]] = {}
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
        if _RUNTIME.ready:
            if (
                list(adapter.last_layer_ptrs.values()) != _RUNTIME.layer_ptrs
                or adapter.last_block_stride != _RUNTIME.block_stride
            ):
                raise OmniKVError(
                    "re-registration with different GPU addresses; the "
                    "extensible KV cache must not move pools after "
                    "registration (M1 limitation)"
                )
            return
        _RUNTIME.descriptor = descriptor
        _RUNTIME.layer_names = descriptor.geometry.layer_names
        _RUNTIME.layer_ptrs = [
            adapter.last_layer_ptrs[name] for name in _RUNTIME.layer_names
        ]
        _RUNTIME.block_stride = adapter.last_block_stride
        _RUNTIME.store = CpuPageStore(descriptor, self._num_pages)
        _RUNTIME.engine = RegionTransferEngine(device_index=adapter.device_index)

    def start_load_kv(
        self, forward_context: ForwardContext, **kwargs: Any
    ) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, OmniKVMetadata)
        runtime = _RUNTIME
        if not runtime.ready:
            raise OmniKVError("worker load before registration")
        store, engine = runtime.store, runtime.engine
        assert store is not None and engine is not None
        fingerprint_id = runtime.descriptor.fingerprint.identity
        for spec in meta.loads:
            ops: list[PageCopyOp] = []
            pages: list[int] = []
            for block_id, block_hash in zip(spec.dst_block_ids, spec.hashes):
                page = store.page_of(
                    BlockKey(fingerprint_id, block_hash)
                )
                if page is None:
                    # Pages hit by the scheduler lookup are reserved against
                    # eviction, so a miss here is an invariant failure. Fail
                    # the step rather than run on unverified bytes.
                    raise OmniKVError(
                        f"load for request {spec.request_id!r} missed its "
                        "reserved store page; refusing unverified bytes"
                    )
                pages.append(page)
                host_base = store.page_ptr(page)
                for li, gpu_base in enumerate(_RUNTIME.layer_ptrs):
                    ops.append(
                        PageCopyOp(
                            src_ptr=host_base + store.layer_offset(li),
                            dst_ptr=gpu_base + block_id * runtime.block_stride,
                            nbytes=store.page_bytes,
                        )
                    )
            ticket = engine.submit("h2d", ops, fence=True)
            ticket.wait()  # M1: synchronous loads
            store.release_pages(pages)
            self._finished_recving.add(spec.request_id)

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
        store, engine = runtime.store, runtime.engine
        assert store is not None and engine is not None
        fingerprint_id = runtime.descriptor.fingerprint.identity
        for spec in meta.saves:
            self._save_one(spec, store, engine, fingerprint_id, runtime)
            self._finished_sending.add(spec.request_id)

    def _save_one(
        self,
        spec: SaveSpec,
        store: CpuPageStore,
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
            ops = [
                PageCopyOp(
                    src_ptr=gpu_base + block_id * runtime.block_stride,
                    dst_ptr=store.page_ptr(page) + store.layer_offset(li),
                    nbytes=store.page_bytes,
                )
                for key, page, block_id in acquired
                for li, gpu_base in enumerate(runtime.layer_ptrs)
            ]
            if ops:
                ticket = engine.submit("d2h", ops, fence=True)
                ticket.wait()  # M1: synchronous saves
            for key, page, _ in acquired:
                store.publish(key, page)
                runtime.stats_saved += 1
        except Exception:
            for _, page, _ in acquired:
                if page not in store._inflight:
                    continue
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
        bs = self._block_size
        tokens = list(request.all_token_ids)[: request.num_tokens]
        # Never match into the final token: the scheduler needs >= 1 token
        # to compute (standard conservative connector behavior).
        max_blocks = (len(tokens) - 1) // bs
        computed_blocks = num_computed_tokens // bs
        matched = 0
        if max_blocks > computed_blocks:
            hashes = chain_block_hashes(tokens, bs)[computed_blocks:max_blocks]
            keys = [
                BlockKey(runtime.descriptor.fingerprint.identity, h)
                for h in hashes
            ]
            pages = runtime.store.longest_prefix(keys)
            runtime.store.reserve_pages(pages)
            runtime.stats_hits += len(pages)
            matched = len(pages)
        self._last_match[request.request_id] = (num_computed_tokens, matched * bs)
        return matched * bs, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens == 0:
            self._last_match.pop(request.request_id, None)
            return
        bs = self._block_size
        num_computed = self._last_match.get(
            request.request_id,
            (request.num_computed_tokens - num_external_tokens, 0),
        )[0]
        computed_blocks = num_computed // bs
        matched_blocks = num_external_tokens // bs
        group_ids = blocks.get_block_ids()[0]  # single group (M1)
        dst = group_ids[computed_blocks : computed_blocks + matched_blocks]
        hashes = chain_block_hashes(
            list(request.all_token_ids)[: request.num_tokens], bs
        )[computed_blocks : computed_blocks + matched_blocks]
        if len(dst) != matched_blocks or len(hashes) != matched_blocks:
            raise OmniKVError(
                f"allocation/bookkeeping mismatch for {request.request_id!r}: "
                f"dst={len(dst)} hashes={len(hashes)} matched={matched_blocks}"
            )
        self._pending_loads[request.request_id] = LoadSpec(
            request_id=request.request_id,
            dst_block_ids=list(dst),
            hashes=list(hashes),
        )

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
        # M1 restores engine-written whole pages in the identical layout.
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _eligible(request: Request) -> bool:
        """LoRA / multimodal requests are not cached in M1 (research.md §7.1
        extra-key rules arrive with the fingerprint widening in M2)."""
        return not (
            getattr(request, "lora_request", None) is not None
            or bool(getattr(request, "mm_features", None))
            or bool(getattr(request, "mm_hashes", None))
        )

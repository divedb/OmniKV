"""OmniKV V1 connector for vLLM (M4: semantic multi-group cache reuse).

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

Layout and semantic behavior:
- Transfers are placement-driven (`GroupPlacement.gpu_ops`), so fragmented
  pages (LHBNC/BHLNC) move as per-head runs without core changes.
- One store per `FormatFingerprint` identity; with ``omnikv.cross_layout``
  enabled, lookups also consult stores of convertible fingerprints through
  the codec registry, and the worker repacks those pages before loading
  (explicit conversion; never silent reinterpretation).
- The runtime survives engine re-registration in-process (sequential
  engines with different layouts share the stores — the M2 cross-layout
  validation topology).

M4 adds independent group plans, trailing-window lookup, HMA completion
aggregation, model/layer namespaces and immutable per-tensor FP8 scale capture.
The scheduler and worker still share one in-process runtime; this is not a
distributed cache directory. Recurrent states remain explicitly unsupported.

Configuration (``kv_connector_extra_config``):
  - ``omnikv.cpu_pages``: store capacity in blocks per fingerprint.
  - ``omnikv.save_enabled``: disable saving (lookup-only debugging).
  - ``omnikv.verify_load_bytes``: bitwise read-back gate after every load.
  - ``omnikv.cross_layout``: allow codec-converted loads from other layouts.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field, replace
from math import lcm
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorTransferResults,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.codec import DEFAULT_REGISTRY, CodecRegistry
from omnikv.errors import CompatibilityError, OmniKVError, StoreError
from omnikv.keys import BlockKey, chain_block_hashes, extend_block_hashes
from omnikv.semantics import required_blocks
from omnikv.store.cpu import CpuPageStore
from omnikv.transfer.engine import (
    PageCopyOp,
    RegionTransferEngine,
    plan_batch,
)
from omnikv.types import FormatFingerprint, GroupDescriptor, GroupPlacement

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
    # Reserved source pages, parallel to dst_block_ids/hashes. Explicit
    # source identities also permit safe cleanup of stale-generation plans.
    conversions: list[tuple[str, int]] = field(default_factory=list)
    # Engine generation at build time; the worker refuses to execute plans
    # from a superseded engine (M3 hardening: two live engines must never
    # share placement).
    generation: int = 0
    group_id: int = 0


@dataclass
class SaveSpec:
    """One request's D2H plan (worker-side)."""

    request_id: str
    src_block_ids: list[int]
    hashes: list[bytes]
    generation: int = 0
    group_id: int = 0


@dataclass
class OmniKVMetadata(KVConnectorMetadata):
    """Scheduler -> worker payload (primitives only, picklable)."""

    loads: list[LoadSpec] = field(default_factory=list)
    saves: list[SaveSpec] = field(default_factory=list)


@dataclass
class _SaveTicket:
    """State of one asynchronous save awaiting D2H completion."""

    ticket: Any  # TransferTicket
    store: Any  # CpuPageStore | SpillStore
    fingerprint_id: str
    acquired: list[tuple[BlockKey, int]]  # (key, page) to publish
    scratch: torch.Tensor | None = None  # keeps any codec buffers alive


class _Runtime:
    """Process-wide shared state between the two connector roles.

    Holds the *current* engine's descriptor + placement (replaced on
    re-registration by a sequential engine) and one store per fingerprint
    identity, which persists across engines — that persistence is what
    makes cross-layout reuse observable in-process.
    """

    def __init__(self, registry: CodecRegistry | None = None) -> None:
        self.descriptors: dict[int, GroupDescriptor] = {}
        self.placements: dict[int, GroupPlacement] = {}
        self.descriptor: GroupDescriptor | None = None  # GroupDescriptor of the current engine
        self.placement: GroupPlacement | None = None
        self.stores: dict[str, Any] = {}
        self.engine: RegionTransferEngine | None = None
        self.registry = registry or DEFAULT_REGISTRY
        # Bumped on every register_kv_caches; specs carry the generation
        # they were built under and workers refuse stale ones (M3).
        self.generation = 0
        # Async-save tickets in flight (worker submits, get_transfer_results
        # publishes). The scheduler defers lookups while this is non-zero so
        # a prefix is never missed because its producer's D2H has not
        # completed (readiness-before-visibility, research.md §7.4).
        self.pending_saves = 0
        # M5: shared pinned scratch buffers, grown on demand. Repeated
        # cudaHostAlloc in the load/verify path costs milliseconds per step;
        # each use is transient within one synchronous call. Two named
        # slots: the L1 read-back gate must never reallocate the buffer
        # that a live conversion's source pointers still point into.
        self._scratch: dict[str, Any] = {}
        self.stats_hits = 0
        self.stats_codec_hits = 0
        self.stats_saved = 0
        self.stats_save_failures = 0

    def pinned_scratch(self, nbytes: int, slot: str = "convert"):
        """A pinned uint8 scratch view of at least ``nbytes`` (M5).

        Grows geometrically; a returned view is only valid until the next
        call on the same slot (single-threaded engine hot path).
        """
        import torch as _torch

        buf = self._scratch.get(slot)
        current = 0 if buf is None else buf.numel()
        if current < nbytes:
            buf = _torch.empty(max(nbytes, 2 * current), dtype=_torch.uint8, pin_memory=True)
            self._scratch[slot] = buf
        assert buf is not None
        return buf[:nbytes]

    @property
    def ready(self) -> bool:
        return self.placement is not None

    def store_for(self, fingerprint: FormatFingerprint, num_pages: int) -> CpuPageStore:
        from omnikv.types import GroupDescriptor  # local import: dataclass only

        store = self.stores.get(fingerprint.identity)
        if store is None:
            assert self.descriptor is not None
            descriptor = GroupDescriptor(
                group_id=self.descriptor.group_id,
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


class OmniKVConnectorV1(KVConnectorBase_V1, SupportsHMA):
    """CPU/SSD offload of full, latent, windowed and sparse-index groups."""

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
        self._vllm_config = vllm_config
        self._group_ctx: dict[str, tuple[int, dict, int]] = {}
        self._loaded_groups: dict[str, tuple[int, ...]] = {}
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._num_pages = int(extra.get("omnikv.cpu_pages", 1024))
        self._save_enabled = bool(extra.get("omnikv.save_enabled", True))
        self._verify_loads = bool(extra.get("omnikv.verify_load_bytes", False))
        self._cross_layout = bool(extra.get("omnikv.cross_layout", False))
        self._preload_store = str(extra.get("omnikv.preload_store", "") or "")
        self._preloaded = False
        self._spill_dir = str(extra.get("omnikv.spill_dir", "") or "")
        self._spill_pages = int(extra.get("omnikv.spill_pages", 4096))
        # Scheduler-role bookkeeping.
        self._last_match: dict[str, tuple[int, int]] = {}
        self._pending_loads: dict[tuple[str, int], LoadSpec] = {}
        self._pending_saves: dict[tuple[str, int], SaveSpec] = {}
        self._steps = 0
        # M5: per-request block-hash chains, keyed (request_id, block_size).
        # get_num_new_matched_tokens is re-queried every step while lookups
        # are deferred (the None contract), and request_finished hashes the
        # grown token list again; all_token_ids is append-only, so cached
        # chains extend instead of recomputing (research.md §8.6: hash the
        # prefix once per request per lookup-group).
        self._hash_chains: dict[tuple[str, int], tuple[int, list[bytes]]] = {}
        self._hash_steps: dict[str, int] = {}
        # Worker-role completion state: async save tickets keyed by request
        # id, each carrying what to publish when the D2H completes.
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._save_tickets: dict[tuple[str, int], _SaveTicket] = {}

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        adapter = VllmLayoutAdapter()
        descriptors = adapter.describe(kv_caches, self._kv_cache_config)
        if not descriptors:
            raise OmniKVError("no transferable groups")
        namespace = (
            str(getattr(self._vllm_config.model_config, "model", "")),
            str(getattr(self._vllm_config.model_config, "revision", "")),
        )
        self._scale_layers = {}
        self._scale_values = {}
        self._scale_versions = {}
        resolved = {}
        for desc in descriptors:
            scale_layout = None
            if desc.fingerprint.quant == "FP8_PER_TENSOR":
                context = self._vllm_config.compilation_config.static_forward_context
                values = []
                for name in desc.geometry.layer_names:
                    layer = context[name]
                    if getattr(layer, "calculate_kv_scales", False):
                        raise OmniKVError("dynamic per-tensor FP8 scales are unsupported")
                    scales = tuple(
                        float(getattr(layer, attr).item()) for attr in ("_k_scale", "_v_scale")
                    )
                    if not all(math.isfinite(v) and v > 0 for v in scales):
                        raise OmniKVError("invalid per-tensor FP8 scales")
                    self._scale_layers[name] = layer
                    self._scale_values[name] = scales
                    self._scale_versions[name] = self._scale_stamp(layer)
                    values.append((name, scales, self._scale_stamp(layer)[2:]))
                # Captured checkpoint scales are part of compatibility: never
                # overwrite a global scale while other requests use that layer.
                scale_layout = repr(tuple(values))
            fp = replace(
                desc.fingerprint,
                scale_layout=scale_layout,
                namespace=namespace + (desc.group_id, desc.geometry.layer_names),
            )
            resolved[desc.group_id] = replace(desc, fingerprint=fp)
        _RUNTIME.descriptors = resolved
        _RUNTIME.placements = adapter.placements
        _RUNTIME.generation += 1
        for gid, desc in resolved.items():
            self._activate_group(gid)
            _RUNTIME.store_for(desc.fingerprint, self._num_pages)
            if self._spill_dir:
                self._install_spill(desc.fingerprint)
        self._activate_group(next(iter(resolved)))
        if _RUNTIME.engine is None:
            _RUNTIME.engine = RegionTransferEngine(device_index=adapter.device_index)
        if self._preload_store and not self._preloaded:
            load_runtime_stores(self._preload_store)
            self._preloaded = True

    def _activate_group(self, gid: int) -> None:
        _RUNTIME.descriptor = _RUNTIME.descriptors[gid]
        _RUNTIME.placement = _RUNTIME.placements[gid]

    @staticmethod
    def _scale_stamp(layer):
        # Tensor mutation versions avoid a device-to-host synchronization in
        # the transfer hot path. Replacement and in-place updates invalidate
        # the captured checkpoint scale namespace.
        return tuple(
            (id(getattr(layer, a)), getattr(layer, a)._version) for a in ("_k_scale", "_v_scale")
        ) + tuple(getattr(layer, a, None) for a in ("_k_scale_float", "_v_scale_float"))

    def _check_scales(self) -> None:
        for name, layer in getattr(self, "_scale_layers", {}).items():
            if self._scale_stamp(layer) != self._scale_versions[name]:
                raise OmniKVError(f"FP8 scales changed after registration: {name}")

    def _install_spill(self, fingerprint) -> None:
        """Wrap the fingerprint's hot store with a disk spill tier."""
        from omnikv.store.disk import DiskPageStore
        from omnikv.store.spill import SpillStore

        fp_id = fingerprint.identity
        if isinstance(_RUNTIME.stores.get(fp_id), SpillStore):
            return
        hot = _RUNTIME.stores.get(fp_id)
        assert hot is not None
        cold = DiskPageStore(hot.descriptor, self._spill_dir, self._spill_pages)
        _RUNTIME.stores[fp_id] = SpillStore(hot, cold)

    def _check_generation(self, spec_generation: int, what: str) -> None:
        if spec_generation != _RUNTIME.generation:
            raise OmniKVError(
                f"{what} built for engine generation {spec_generation} but "
                f"the runtime is at generation {_RUNTIME.generation}; "
                "sequential engines only — a plan from a superseded engine "
                "must never execute against the current placement"
            )

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, OmniKVMetadata)
        # All groups belong to one synchronous load transaction. Even if
        # the first group's generation/scale/copy check fails, release every
        # reservation, including groups whose copies never started.
        try:
            self._load_batches(meta)
        finally:
            for spec in meta.loads:
                for fp_id, page in spec.conversions:
                    _RUNTIME.stores[fp_id].release_pages([page])

    def _load_batches(self, meta: OmniKVMetadata) -> None:
        runtime = _RUNTIME
        if not runtime.ready:
            raise OmniKVError("worker load before registration")
        engine = runtime.engine
        assert engine is not None
        self._check_scales()
        for spec in meta.loads:
            self._check_generation(spec.generation, "transfer spec")
            self._activate_group(spec.group_id)
            placement = runtime.placements[spec.group_id]
            fingerprint = runtime.descriptors[spec.group_id].fingerprint
            primary = runtime.stores[fingerprint.identity]
            self._check_generation(spec.generation, "load spec")
            ops: list[PageCopyOp] = []
            sources: list[tuple[int, int]] = []  # (host_ptr, nbytes) per op
            if len(spec.conversions) != len(spec.hashes):
                raise OmniKVError("load lacks complete reservation metadata")
            n_converted = sum(f != fingerprint.identity for f, _ in spec.conversions)
            scratch = None
            if n_converted:
                scratch = runtime.pinned_scratch(n_converted * primary.slot_bytes)
            scratch_off = 0
            for block_id, block_hash, (source_id, page) in zip(
                spec.dst_block_ids, spec.hashes, spec.conversions, strict=True
            ):
                store = runtime.stores[source_id]
                key = BlockKey(source_id, block_hash)
                if store.page_of(key) != page:
                    raise OmniKVError("load missed its reserved store page")
                host_base = store.page_ptr(page)
                if source_id != fingerprint.identity:
                    assert scratch is not None
                    self._converted_page(
                        store,
                        page,
                        fingerprint,
                        scratch[scratch_off : scratch_off + primary.slot_bytes],
                    )
                    host_base = scratch.data_ptr() + scratch_off
                    scratch_off += primary.slot_bytes
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
            if not ops:
                continue  # e.g. window=1 has no past state to restore
            ticket = engine.submit("h2d", plan_batch(ops), fence=True)
            ticket.wait()  # synchronous loads (M1-M3)
            if self._verify_loads:
                self._verify_loaded_bytes(ops, sources, engine)

    def _converted_page(
        self,
        src_store: CpuPageStore,
        page: int,
        dst_fp: FormatFingerprint,
        scratch,  # torch.Tensor: the destination region for this block
    ):
        """Repack one foreign-fingerprint store page into ``dst_fp``.

        The codec's unit is ONE layer's page; a store slot holds the
        group's layer-major pages, so each layer's slice is repacked into
        the matching slice of the destination region.
        """
        runtime = _RUNTIME
        src_fp = src_store.descriptor.fingerprint
        codec = runtime.registry.find(src_fp, dst_fp)
        if codec is None:
            raise CompatibilityError(
                f"no registered codec for {src_fp.identity} -> "
                f"{dst_fp.identity}; cross-layout load refused"
            )
        slot = src_store.pool_slice(page)
        # One batched repack for the whole slot (M3: was per-layer).
        codec.convert(slot, scratch, src_fp, dst_fp)
        runtime.stats_codec_hits += 1

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
        """Submit this step's saves asynchronously (M3).

        The D2H copies run on the private save stream; pages are published
        — and the request reported finished_sending — only when the ticket
        completes, polled in `get_transfer_results`. The deferred-free
        contract keeps the source GPU blocks alive until then, so no
        synchronize is needed here. This also removes the M2-era
        "saves need a follow-up engine step" flush workaround ordering:
        the ticket is checked on every subsequent step.
        """
        meta = self._get_connector_metadata()
        assert isinstance(meta, OmniKVMetadata)
        runtime = _RUNTIME
        if not runtime.ready:
            raise OmniKVError("worker save before registration")
        engine = runtime.engine
        assert engine is not None
        self._check_scales()
        for spec in meta.saves:
            self._check_generation(spec.generation, "transfer spec")
            self._activate_group(spec.group_id)
            placement = runtime.placements[spec.group_id]
            fingerprint_id = runtime.descriptors[spec.group_id].fingerprint.identity
            store = runtime.stores[fingerprint_id]
            self._check_generation(spec.generation, "save spec")
            self._save_one(spec, store, placement, engine, fingerprint_id, runtime)

    def _save_one(
        self,
        spec: SaveSpec,
        store,  # CpuPageStore | SpillStore
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
            if not ops:
                self._finished_sending.add(spec.request_id)
                return
            ticket = engine.submit("d2h", plan_batch(ops), fence=True)
            _RUNTIME.pending_saves += 1
            self._save_tickets[(spec.request_id, spec.group_id)] = _SaveTicket(
                ticket=ticket,
                store=store,
                fingerprint_id=fingerprint_id,
                acquired=[(k, p) for k, p, _ in acquired],
            )
        except Exception:  # noqa: BLE001 - best-effort save by design
            # Best-effort save semantics (research.md §7.5): a failed save
            # is a future cache miss, never a request failure. Release the
            # acquired pages and report the request finished sending.
            for _, page, _ in acquired:
                if page in getattr(store, "_inflight", ()):
                    store.release(page)
            runtime.stats_save_failures += 1
            self._finished_sending.add(spec.request_id)

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )

        # Publish completed async saves (readiness-before-visibility:
        # pages become lookup-visible only after the D2H completes).
        for req_id, st in list(self._save_tickets.items()):
            if not st.ticket.ready():
                continue
            del self._save_tickets[req_id]
            _RUNTIME.pending_saves -= 1
            for key, page in st.acquired:
                st.store.publish(key, page)
                _RUNTIME.stats_saved += 1
            self._finished_sending.add(req_id[0])
        inflight = {key[0] for key in self._save_tickets}
        sending = (self._finished_sending & finished_req_ids) - inflight
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
        self._release_ctx(request.request_id)
        self._loaded_groups.pop(request.request_id, None)
        if not runtime.ready or not self._eligible(request):
            return 0, False
        if self._pending_saves or runtime.pending_saves:
            return None, False
        descs = runtime.descriptors
        alignment = lcm(*(d.semantics.block_size for d in descs.values()))
        tokens = list(request.all_token_ids)[: request.num_tokens]
        hashes = self._chains(request.request_id, tokens)
        # Search from the longest boundary: a window group need not contain
        # early pages, so a left-to-right prefix scan is incorrect.
        for boundary in range(
            (len(tokens) - 1) // alignment * alignment, num_computed_tokens, -alignment
        ):
            plan = {}
            complete = True
            for gid, desc in descs.items():
                fp = desc.fingerprint
                entries = []
                for index in required_blocks(desc.semantics, boundary, num_computed_tokens):
                    candidates = [fp]
                    if self._cross_layout:
                        candidates += runtime.registry.source_fingerprints_for(fp)
                    found = None
                    for source in candidates:
                        store = runtime.stores.get(source.identity)
                        if store is None:
                            continue
                        key = BlockKey(source.identity, hashes[gid][index])
                        page = store.page_of(key)
                        if page is not None:
                            # Reserve during probing: disk promotion of a later
                            # page can otherwise evict an earlier candidate.
                            store.reserve_pages([page])
                            store.touch(key)
                            found = (index, source.identity, page, hashes[gid][index])
                            break
                    if found is None:
                        complete = False
                        break
                    entries.append(found)
                plan[gid] = entries
                if not complete:
                    break
            if complete:
                self._group_ctx[request.request_id] = (boundary, plan, self._steps)
                self._loaded_groups[request.request_id] = tuple(descs)
                self._last_match[request.request_id] = (
                    num_computed_tokens,
                    boundary - num_computed_tokens,
                )
                runtime.stats_hits += sum(len(entries) for entries in plan.values())
                return boundary - num_computed_tokens, False
            self._release_plan(plan)
        return 0, False

    @staticmethod
    def _release_plan(plan) -> None:
        for entries in plan.values():
            for _, fp_id, page, _ in entries:
                _RUNTIME.stores[fp_id].release_pages([page])

    def update_state_after_alloc(self, request, blocks, num_external_tokens) -> None:
        if not num_external_tokens:
            self._release_ctx(request.request_id)
            self._loaded_groups.pop(request.request_id, None)
            return
        ctx = self._group_ctx.pop(request.request_id, None)
        if ctx is None:
            raise OmniKVError("allocation without reserved group lookup")
        _boundary, plan, _ = ctx
        expected = self._last_match.pop(request.request_id)
        if num_external_tokens != expected[1]:
            self._release_plan(plan)
            raise OmniKVError("allocation token count differs from reserved lookup")
        ids = blocks.get_block_ids()
        pending = []
        try:
            for gid, entries in plan.items():
                dst = [ids[gid][i] for i, _, _, _ in entries]
                if any(b <= 0 for b in dst):
                    raise OmniKVError("live restore range contains a null block")
                pending.append(
                    LoadSpec(
                        request_id=request.request_id,
                        dst_block_ids=dst,
                        hashes=[entry[3] for entry in entries],
                        conversions=[(f, p) for _, f, p, _ in entries],
                        generation=_RUNTIME.generation,
                        group_id=gid,
                    )
                )
        except Exception:
            self._release_plan(plan)
            raise
        for spec in pending:
            self._pending_loads[(request.request_id, spec.group_id)] = spec

    def _release_ctx(self, request_id: str) -> None:
        ctx = self._group_ctx.pop(request_id, None)
        if ctx is not None:
            self._release_plan(ctx[1])
        self._last_match.pop(request_id, None)

    def _gc_stale_ctx(self) -> None:
        for rid in [rid for rid, ctx in self._group_ctx.items() if ctx[2] <= self._steps - 100]:
            self._release_ctx(rid)
            self._loaded_groups.pop(rid, None)
        for rid in [
            rid for rid, step in self._hash_steps.items() if step <= self._steps - 100
        ]:
            # Abandoned requests (never finished): their chains are stale.
            self._drop_chains(rid)

    def get_loaded_kv_cache_group_ids(self, request) -> tuple[int, ...]:
        return self._loaded_groups.get(request.request_id, ())

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        self._steps += 1
        self._gc_stale_ctx()
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
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        self._release_ctx(request.request_id)
        self._loaded_groups.pop(request.request_id, None)
        # Called exactly once per request: its hash chains die with it.
        try:
            if (
                not self._save_enabled
                or not self._eligible(request)
                or not request.is_finished()
            ):
                return False, None
            reason = request.get_finished_reason()
            if reason is not None and int(reason) >= 2:
                return False, None
            tokens = list(request.all_token_ids)
            # The last sampled token may not have run through the model yet.
            computed = min(len(tokens), request.num_computed_tokens)
            chains = self._chains(request.request_id, tokens)
            queued = False
            for gid, desc in _RUNTIME.descriptors.items():
                bs = desc.semantics.block_size
                hashes = chains[gid]
                boundary = computed // bs * bs
                entries = [
                    (i, block_ids[gid][i])
                    for i in required_blocks(desc.semantics, boundary)
                    if i < len(block_ids[gid]) and block_ids[gid][i] > 0
                ]
                if not entries:
                    continue
                self._pending_saves[(request.request_id, gid)] = SaveSpec(
                    request_id=request.request_id,
                    src_block_ids=[b for _, b in entries],
                    hashes=[hashes[i] for i, _ in entries],
                    generation=_RUNTIME.generation,
                    group_id=gid,
                )
                queued = True
            return queued, None
        finally:
            self._drop_chains(request.request_id)

    def _drop_chains(self, request_id: str) -> None:
        self._hash_steps.pop(request_id, None)
        for key in [k for k in self._hash_chains if k[0] == request_id]:
            del self._hash_chains[key]

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

    def _chains(self, request_id: str, tokens: list[int]) -> dict[int, list[bytes]]:
        """Per-group block-hash chains for one request's token prefix,
        memoized and extended across scheduler re-queries (M5).

        Valid because chained block hashes are prefix-stable and a
        request's ``all_token_ids`` only ever appends.
        """
        out: dict[int, list[bytes]] = {}
        for gid, desc in _RUNTIME.descriptors.items():
            bs = desc.semantics.block_size
            key = (request_id, bs)
            need = len(tokens) // bs
            cached = self._hash_chains.get(key)
            if cached is not None and cached[0] >= need:
                out[gid] = cached[1][:need]
                continue
            if cached is not None:
                hashes = cached[1] + extend_block_hashes(
                    cached[1][-1] if cached[1] else b"",
                    cached[0],
                    tokens,
                    bs,
                )
            else:
                hashes = chain_block_hashes(tokens, bs)
            self._hash_chains[key] = (len(hashes), hashes)
            out[gid] = hashes
        self._hash_steps[request_id] = self._steps
        return out

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
        source host bytes (research.md §10). M3: torch.equal on one
        packed read-back buffer (was a per-op .tolist()). M5: the read-back
        buffer is a runtime pinned scratch slot, not a per-load
        cudaHostAlloc."""
        total = sum(op.nbytes for op in ops)
        scratch = _RUNTIME.pinned_scratch(total, slot="verify")
        base = scratch.data_ptr()
        offset = 0
        readback: list[PageCopyOp] = []
        for op in ops:
            readback.append(PageCopyOp(src_ptr=op.dst_ptr, dst_ptr=base + offset, nbytes=op.nbytes))
            offset += op.nbytes
        engine.submit("d2h", readback, fence=False).wait()
        offset = 0
        for op, (src_ptr, _) in zip(ops, sources):
            got = scratch[offset : offset + op.nbytes]
            want = torch.frombuffer(
                bytearray(ctypes.string_at(src_ptr, op.nbytes)), dtype=torch.uint8
            )
            if not torch.equal(got, want):
                raise OmniKVError(
                    f"load verification failed: GPU bytes at {hex(op.dst_ptr)} "
                    f"do not match the source page bytes at {hex(src_ptr)} "
                    f"({op.nbytes} B)"
                )
            offset += op.nbytes

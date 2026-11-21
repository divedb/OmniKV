"""The vLLM layout adapter: engine caches -> OmniKV descriptors + placement.

Reads vLLM's authoritative metadata (`kv_cache_config` from the connector
constructor + the `kv_caches` dict from `register_kv_caches`) and produces
layout-parametric descriptors and a runtime `GroupPlacement`. Metadata is
the source of truth; the registered views' actual strides are then
**certified** against the expected permutation (detection is never trusted
silently — research.md §4.3).

Dual-form: supports both the v0.34.0 release `KVCacheGroupSpec`
(``layer_names`` + singular ``kv_cache_spec``) and main's post-#61452 form
(``kv_cache_specs`` dict + ``manager_config``). All access is duck-typed so
this module imports (and unit-tests) without vLLM installed.

Supported physical families (M2):
  - block-compact, layer-compact:  LBNHC (NHD), LBHNC (HND)
  - block-compact, block-outermost: BLNHC, BLHNC
  - head-fragmented:               LHBNC, BHLNC (one store fragment per head)
  - per-layer storages (worker/extensible projection), dim-0 page padding
M4: MLA, sliding-window and sparse-index groups; aligned padding, compressed
states and explicit quantized content. Recurrent states fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from omnikv.errors import RegistrationError
from omnikv.types import (
    FORMAT_VERSION,
    LAYOUT_FAMILIES,
    FormatFingerprint,
    GroupDescriptor,
    GroupPlacement,
    LayerAddress,
    PageGeometry,
    StateKind,
    StateSemantics,
)

try:  # torch is optional for the pure logic path; required at runtime
    import torch
except ImportError:  # local unit tests on hosts without a torch wheel
    torch = None  # type: ignore[assignment]


def _element_size(dtype: Any) -> int:
    """Bytes per element of a torch dtype (or a duck-typed test double)."""
    if torch is not None and hasattr(dtype, "is_floating_point"):
        return torch.tensor([], dtype=dtype).element_size()
    size = getattr(dtype, "itemsize", None)
    if isinstance(size, int) and size > 0:
        return size
    raise RegistrationError(f"cannot resolve element size of dtype {dtype!r}")


def _group_names_and_specs(group: Any) -> tuple[list[str], list[Any]]:
    """Dual-form reader for one KV cache group (release vs. main API).

    Handles the worker-side `UniformTypeKVCacheSpecs` container (same-typed
    layers wrapped in one spec object; it does not derive from AttentionSpec
    itself — unwrap to its `first_spec` for kind/geometry questions).
    """
    names = getattr(group, "layer_names", None)
    if names is not None:  # v0.34.0 release: KVCacheGroupSpec
        spec = group.kv_cache_spec
        per_layer = getattr(spec, "kv_cache_specs", None)
        if isinstance(per_layer, Mapping):
            return list(names), [per_layer[n] for n in names]
        return list(names), [_unwrap_uniform(spec)]
    specs = getattr(group, "kv_cache_specs", None)  # main: post-#61452
    if isinstance(specs, Mapping):
        return list(specs), [_unwrap_uniform(s) for s in specs.values()]
    raise RegistrationError(
        f"unrecognized KV cache group form: {type(group).__name__} has "
        "neither layer_names+kv_cache_spec nor kv_cache_specs"
    )


def _unwrap_uniform(spec: Any) -> Any:
    first = getattr(spec, "first_spec", None)
    return first if first is not None else spec


def _heads(spec: Any) -> int:
    return getattr(spec, "num_heads", spec.num_kv_heads)


def _content(spec: Any) -> int:
    value = getattr(spec, "state_content_size_bytes", None)
    if value is not None:
        return value
    return _element_size(spec.dtype) * (spec.head_size + spec.head_size_v)


def _states(spec: Any) -> int:
    count = Fraction(spec.block_size) / Fraction(getattr(spec, "tokens_per_state", 1))
    if count.denominator != 1 or count <= 0:
        raise RegistrationError("block_size must contain a positive integral number of states")
    return int(count)


def _spec_identity(spec: Any) -> tuple:
    """Value-based identity of one spec's physical format (layout-bearing
    fields only); used to detect mixed per-layer formats."""
    quant = getattr(spec, "kv_quant_mode", None)
    return (
        type(spec).__name__,
        getattr(spec, "sliding_window", None),
        getattr(spec, "extra_retained_tokens", 0),
        getattr(spec, "tokens_per_state", 1),
        _content(spec),
        _heads(spec),
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        getattr(spec, "head_size_v", None),
        str(spec.dtype),
        getattr(quant, "name", None),
        spec.page_size_bytes,
        getattr(spec, "kv_head_major_slots", False),
        getattr(spec, "cache_dtype_str", None),
        getattr(spec, "model_version", None),
    )


def _expected_view_strides(
    layout: str,
    spec: Any,
    rows: int,
    num_layers: int,
    tokens: int,
    elem: int,
) -> tuple[int, int, int, int]:
    """Expected per-layer view strides, in ELEMENTS of the view's dtype.

    ``rows`` is the view's block-row count (``num_blocks * ratio`` for
    kernel-block splits) and ``tokens`` the per-row token count (the kernel
    block size). The physical axis order is the layout permutation over
    [L, B, H, N, C] (RFC #42082); within one layer the view exposes
    (rows, H, tokens, C) with the strides that permutation implies. C is
    the full per-(token, head) content width (fused K/V packing included);
    it must divide evenly into whole elements of the view dtype.
    """
    h, c_bytes = _heads(spec), _content(spec)
    if c_bytes % elem != 0:
        raise RegistrationError(
            f"content width {c_bytes} B not divisible by view element size "
            f"{elem}; the engine's page cannot be addressed by this view"
        )
    c = c_bytes // elem
    row = tokens * h * c
    if tokens == _states(spec):
        row = spec.page_size_bytes // elem
    if layout == "LBNHC":
        return (row, c, h * c, 1)
    if layout == "LBHNC":
        return (row, tokens * c, c, 1)
    if layout == "LHBNC":
        return (tokens * c, rows * tokens * c, c, 1)
    if layout == "BLNHC":
        return (num_layers * row, c, h * c, 1)
    if layout == "BLHNC":
        return (num_layers * row, tokens * c, c, 1)
    if layout == "BHLNC":
        # (B, H, L, N, C): heads outermost; a layer's head h pages sit at
        # h * (L * row) with block-row stride L * row * H.
        return (h * num_layers * row, num_layers * row, c, 1)
    raise RegistrationError(f"unsupported layout {layout!r}")


class VllmLayoutAdapter:
    """Builds descriptors + placement from vLLM's resolved KV cache config.

    Groups retain independent semantics, fingerprints and runtime placements.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.last_placement: GroupPlacement | None = None
        self.placements: dict[int, GroupPlacement] = {}

    # -- properties kept from M1 for the connector ------------------------

    @property
    def last_layer_ptrs(self) -> dict[str, int]:
        assert self.last_placement is not None
        return {a.layer_name: a.base_ptr for a in self.last_placement.addresses}

    @property
    def last_block_stride(self) -> int:
        assert self.last_placement is not None
        strides = {s for a in self.last_placement.addresses for s in a.block_strides}
        if len(strides) != 1:
            raise RegistrationError(f"single block_stride unavailable: {sorted(strides)}")
        return strides.pop()

    # -- describe ---------------------------------------------------------

    def describe(
        self,
        kv_caches: Mapping[str, Any],
        kv_cache_config: Any,
    ) -> tuple[GroupDescriptor, ...]:
        groups = list(getattr(kv_cache_config, "kv_cache_groups", []) or [])
        if not groups:
            raise RegistrationError("engine has no KV cache groups")
        self.placements = {}
        descriptors = []
        for gid, group in enumerate(groups):
            if not getattr(group, "enable_kv_transfer", True):
                continue
            descriptor = self._describe_group(kv_caches, kv_cache_config, group, gid)
            descriptors.append(descriptor)
            assert self.last_placement is not None
            self.placements[gid] = self.last_placement
        return tuple(descriptors)

    def _describe_group(self, kv_caches, kv_cache_config, group, gid):
        layer_names, specs = _group_names_and_specs(group)
        supported = {
            "FullAttentionSpec": StateKind.PAGED_KV,
            "MLAAttentionSpec": StateKind.LATENT_KV,
            "SlidingWindowSpec": StateKind.SLIDING_WINDOW_KV,
            "SlidingWindowMLASpec": StateKind.SLIDING_WINDOW_KV,
        }
        if not specs or not layer_names:
            raise RegistrationError("empty cache group")
        kinds = {type(s).__name__ for s in specs}
        if not kinds <= supported.keys():
            raise RegistrationError(
                f"unsupported state specs {sorted(kinds)}; PAGED_KV/MLA/sliding-window only; "
                "RECURRENT_STATE requires validated boundary metadata (M4 no-go)"
            )
        if len({_spec_identity(s) for s in specs}) > 1:
            raise RegistrationError("mixed per-layer formats inside one cache group")
        spec = specs[0]
        kind = supported[type(spec).__name__]
        role = getattr(group, "role", None)
        if (
            getattr(role, "name", role) == "HISPARSE_INDEXER"
            or getattr(getattr(spec, "cache_role", None), "value", None) == "indexer"
        ):
            kind = StateKind.SPARSE_INDEX

        layout = getattr(kv_cache_config, "kv_cache_layout", None)
        if layout is None:
            raise RegistrationError(
                "kv_cache_config.kv_cache_layout is not resolved; refusing "
                "to guess the physical layout (research.md §4.3)"
            )
        if layout not in LAYOUT_FAMILIES:
            raise RegistrationError(f"unknown KV cache layout {layout!r}")

        quant = self._quant_name(spec)
        head_size_v = getattr(spec, "head_size_v", spec.head_size)
        if head_size_v not in (0, spec.head_size):
            raise RegistrationError("diff-KV (head_size_v != head_size) is not supported")

        block_size = spec.block_size
        num_heads = _heads(spec)
        dtype = spec.dtype
        unpadded = _states(spec) * num_heads * _content(spec)
        page_bytes = spec.page_size_bytes
        if page_bytes < unpadded:
            raise RegistrationError(
                f"page_size_bytes={page_bytes} inconsistent with geometry (expected >= {unpadded})"
            )
        head_fragmented = layout in ("LHBNC", "BHLNC")
        if head_fragmented and page_bytes != unpadded:
            raise RegistrationError(
                f"head-fragmented layout {layout} requires an unpadded page "
                f"(page={page_bytes}, unpadded={unpadded})"
            )

        geometry = self._resolve_geometry(
            kv_cache_config, kv_caches, layer_names, page_bytes, layout, spec
        )

        if head_fragmented:
            row = _states(spec) * _content(spec)
            fragments = tuple(row for _ in range(num_heads))
        else:
            fragments = (page_bytes,)

        semantics = StateSemantics(
            kind=kind,
            block_size=block_size,
            tokens_per_state=Fraction(getattr(spec, "tokens_per_state", 1)),
            slots_per_block=_states(spec),
            window_tokens=getattr(spec, "sliding_window", None),
            extra_retained_tokens=getattr(spec, "extra_retained_tokens", 0),
            prefix_cacheable=bool(getattr(spec, "prefix_cacheable", True)),
        )
        fingerprint = FormatFingerprint(
            state_kind=kind,
            layout=layout,
            dtype=(str(dtype).removeprefix("torch."),),
            quant=quant,
            head_geometry=(num_heads, spec.head_size, head_size_v, block_size),
            page_bytes=(page_bytes, page_bytes),
            semantic_geometry=(
                str(semantics.tokens_per_state),
                semantics.window_tokens,
                semantics.extra_retained_tokens,
                _content(spec),
                getattr(spec, "cache_dtype_str", None),
                getattr(spec, "model_version", None),
                getattr(spec, "kv_head_major_slots", False),
                spec.num_kv_heads,
            ),
            scale_layout=None,
            page_fragments=fragments,
            version=FORMAT_VERSION,
        )
        return GroupDescriptor(
            group_id=gid, semantics=semantics, fingerprint=fingerprint, geometry=geometry
        )

    def _quant_name(self, spec: Any) -> str:
        mode = getattr(spec, "kv_quant_mode", None)
        name = getattr(mode, "name", "NONE")
        inline_modes = {
            "FP8_PER_TOKEN_HEAD",
            "INT8_PER_TOKEN_HEAD",
            "INT4_PER_TOKEN_HEAD",
            "NVFP4_DS_MLA",
        }
        if name not in {"NONE", "FP8_PER_TENSOR"} | inline_modes:
            raise RegistrationError(f"quantized mode {name} has no certified scale contract")
        if name in inline_modes and getattr(spec, "state_content_bytes", None) is None:
            raise RegistrationError(
                f"quantized {name} requires explicit inline state_content_bytes"
            )
        return name

    # -- geometry + placement ---------------------------------------------

    def _resolve_geometry(
        self,
        kv_cache_config: Any,
        kv_caches: Mapping[str, Any],
        layer_names: Sequence[str],
        page_bytes: int,
        layout: str,
        spec: Any,
    ) -> PageGeometry:
        """Match the group to its KVCacheTensor(s), certify the registered
        views' strides against the layout permutation, and resolve the
        runtime `GroupPlacement` (also exposed as ``last_placement``).

        Two structural tensor forms occur (verified on vLLM 0.34.0):
        scheduler-side: one tensor covering every layer; worker-side under
        the extensible KV cache: one tensor per layer. Both are matched by
        layer membership.
        """
        tensors = list(getattr(kv_cache_config, "kv_cache_tensors", []) or [])
        owner: dict[str, Any] = {}
        for t in tensors:
            for name in getattr(t, "layers", []) or []:
                if name in owner:
                    raise RegistrationError(f"layer {name!r} appears in multiple KVCacheTensors")
                owner[name] = t
        missing = [n for n in layer_names if n not in owner]
        if missing:
            raise RegistrationError(
                f"{len(missing)} of {len(layer_names)} group layers have no "
                f"KVCacheTensor (first missing: {missing[0]!r})"
            )

        block_strides = {owner[n].block_stride for n in layer_names}
        if len(block_strides) != 1:
            raise RegistrationError(
                f"group layers disagree on block_stride: {sorted(block_strides)}"
            )
        block_stride = block_strides.pop()

        missing_views = [n for n in layer_names if n not in kv_caches]
        if missing_views:
            raise RegistrationError(
                f"kv_caches is missing {len(missing_views)} of "
                f"{len(layer_names)} registered layers (first missing: "
                f"{missing_views[0]!r})"
            )

        num_blocks = kv_cache_config.num_blocks
        head_fragmented = layout in ("LHBNC", "BHLNC")
        row_bytes = page_bytes // _heads(spec)  # one head's page run

        addresses: list[LayerAddress] = []
        for li, name in enumerate(layer_names):
            view = kv_caches[name]
            device_index = getattr(view.device, "index", None)
            if device_index is None or device_index != self.device_index:
                raise RegistrationError(
                    f"layer {name!r} is on {view.device}; adapter targets "
                    f"device {self.device_index}"
                )
            # Kernel-block split: the registered view exposes
            # (num_blocks * ratio, H, block_size // ratio, C). The ratio is
            # metadata-derived (shape vs num_blocks), never guessed.
            shape = tuple(view.shape)
            if len(shape) != 4:
                raise RegistrationError(f"layer {name!r}: expected a rank-4 page view, got {shape}")
            rows = shape[0]
            if rows % num_blocks != 0:
                raise RegistrationError(
                    f"layer {name!r}: view rows {rows} not divisible by num_blocks {num_blocks}"
                )
            ratio = rows // num_blocks
            if ratio < 1 or _states(spec) % ratio != 0:
                raise RegistrationError(
                    f"layer {name!r}: block_size {spec.block_size} not "
                    f"divisible by kernel ratio {ratio}"
                )
            if ratio > 1 and layout.startswith("B"):
                # M5 fail-closed: in block-outermost layouts a manager
                # block's kernel rows sit L*row apart, so its page is NOT
                # one contiguous run and `LayerAddress`'s single-run
                # addressing would copy interleaved bytes silently.
                raise RegistrationError(
                    f"layer {name!r}: block-outermost layout {layout} with "
                    f"kernel-block splits (ratio {ratio}) scatters a manager "
                    "block's page across kernel rows; contiguous per-block "
                    "addressing is not provable, refusing registration"
                )
            tokens = _states(spec) // ratio
            if ratio > 1 and spec.page_size_bytes != _states(spec) * _heads(spec) * _content(spec):
                raise RegistrationError("padded kernel-split pages are not certified")
            if shape[2] != tokens:
                raise RegistrationError(
                    f"layer {name!r}: view tokens {shape[2]} != kernel block size {tokens}"
                )
            # torch strides are in ELEMENTS of the view dtype (the engine
            # registers dtype-typed views). Certification runs in element
            # units; byte addressing (data_ptr + offsets) uses stride*elem.
            elem = _element_size(getattr(view, "dtype", None) or spec.dtype)
            if shape[1] != _heads(spec) or shape[3] * elem != _content(spec):
                raise RegistrationError(f"layer {name!r}: view shape disagrees with page content")
            self._certify_view(
                name,
                tuple(view.stride()),
                elem,
                layout,
                spec,
                rows,
                len(layer_names),
                tokens,
            )
            strides_b = tuple(s * elem for s in view.stride())
            base = view.data_ptr()
            # Manager-block addressing: consecutive kernel rows of one
            # manager block are contiguous (certified), so a manager block
            # spans ratio * stride(0) bytes from its first kernel row.
            manager_stride = ratio * strides_b[0]
            if head_fragmented:
                head_stride = strides_b[1]
                addresses.append(
                    LayerAddress(
                        base_ptr=base,
                        offsets=tuple(h * head_stride for h in range(_heads(spec))),
                        block_strides=tuple(manager_stride for _ in range(_heads(spec))),
                        sizes=tuple(row_bytes for _ in range(_heads(spec))),
                        component="kv",
                        layer_name=name,
                    )
                )
            else:
                addresses.append(
                    LayerAddress(
                        base_ptr=base,
                        offsets=(0,),
                        block_strides=(manager_stride,),
                        sizes=(page_bytes,),
                        component="kv",
                        layer_name=name,
                    )
                )

        offsets: list[int] = []
        acc = 0
        for addr in addresses:
            offsets.append(acc)
            acc += sum(addr.sizes)
        self.last_placement = GroupPlacement(
            addresses=tuple(addresses), page_offsets=tuple(offsets)
        )

        # Cross-check the derived manager-block strides: they must cover
        # the page payload without overlap.
        all_strides = {s for addr in addresses for s in addr.block_strides}
        if len(all_strides) != 1:
            raise RegistrationError(f"inconsistent manager block strides: {sorted(all_strides)}")
        derived_stride = next(iter(all_strides))
        if derived_stride < page_bytes and not head_fragmented:
            raise RegistrationError(
                f"derived manager stride {derived_stride} < page {page_bytes}; "
                "kernel rows would overlap blocks"
            )

        shared = len(tensors) == 1
        first = owner[layer_names[0]]
        offset0 = int(getattr(first, "offset", 0) or 0)
        layer_stride = first.layer_stride if shared else block_stride
        if shared:
            layer_compact = layout.startswith("L")
            if layer_compact and block_stride < page_bytes and not head_fragmented:
                raise RegistrationError(
                    f"layer-compact layout {layout} requires block_stride >= "
                    f"page; got block_stride={block_stride}, page={page_bytes}"
                )
            if not layer_compact and layer_stride != page_bytes and not head_fragmented:
                raise RegistrationError(
                    f"block-outermost layout {layout} requires layer_stride == "
                    f"page; got layer_stride={layer_stride}, page={page_bytes}"
                )
        if block_stride < page_bytes and not head_fragmented:
            raise RegistrationError(
                f"page ({page_bytes} B) exceeds block stride ({block_stride} B); "
                "pages would overlap"
            )

        return PageGeometry(
            num_blocks=num_blocks,
            page_bytes=page_bytes,
            layer_stride=layer_stride,
            block_stride=block_stride,
            storage_id=0,
            offset=offset0,
            layer_names=tuple(layer_names),
            shared_storage=shared,
        )

    def _certify_view(
        self,
        name: str,
        strides_e: tuple[int, ...],
        elem: int,
        layout: str,
        spec: Any,
        rows: int,
        num_layers: int,
        tokens: int,
    ) -> None:
        """Fail-fast check: the view's actual element strides must equal the
        strides the layout permutation implies (at kernel-block
        granularity). This is certification, not detection: a mismatch
        means our model of the engine is wrong and we refuse to touch the
        bytes."""
        expected = _expected_view_strides(layout, spec, rows, num_layers, tokens, elem)
        if tuple(strides_e) != expected:
            raise RegistrationError(
                f"layer {name!r}: view strides {tuple(strides_e)} do not match "
                f"the certified {layout} strides {expected} "
                f"(rows={rows}, tokens={tokens}, elem={elem}); refusing "
                "uncertified layout (research.md §4.3)"
            )

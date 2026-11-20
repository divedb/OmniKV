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
Unsupported (fail-fast): mixed per-layer specs inside one group, quantized
caches, diff-KV heads, non-attention state kinds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def _spec_identity(spec: Any) -> tuple:
    """Value-based identity of one spec's physical format (layout-bearing
    fields only); used to detect mixed per-layer formats."""
    quant = getattr(spec, "kv_quant_mode", None)
    return (
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        getattr(spec, "head_size_v", None),
        str(spec.dtype),
        getattr(quant, "name", None),
        spec.page_size_bytes,
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
    h, c_bytes = spec.num_kv_heads, _element_size(spec.dtype) * (
        spec.head_size + spec.head_size_v
    )
    if c_bytes % elem != 0:
        raise RegistrationError(
            f"content width {c_bytes} B not divisible by view element size "
            f"{elem}; the engine's page cannot be addressed by this view"
        )
    c = c_bytes // elem
    row = tokens * h * c
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

    M1/M2 scope (fail-fast outside it): exactly one cache group of
    ``FullAttentionSpec`` (PAGED_KV), unquantized, ``head_size_v ==
    head_size``.
    """

    # Spec class names (duck-typed) that map onto M1/M2's PAGED_KV semantics.
    _PAGED_KV_SPECS = frozenset({"FullAttentionSpec"})

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.last_placement: GroupPlacement | None = None

    # -- properties kept from M1 for the connector ------------------------

    @property
    def last_layer_ptrs(self) -> dict[str, int]:
        assert self.last_placement is not None
        return {a.layer_name: a.base_ptr for a in self.last_placement.addresses}

    @property
    def last_block_stride(self) -> int:
        assert self.last_placement is not None
        strides = {
            s for a in self.last_placement.addresses for s in a.block_strides
        }
        if len(strides) != 1:
            raise RegistrationError(
                f"single block_stride unavailable: {sorted(strides)}"
            )
        return strides.pop()

    # -- describe ---------------------------------------------------------

    def describe(
        self,
        kv_caches: Mapping[str, Any],
        kv_cache_config: Any,
    ) -> tuple[GroupDescriptor, ...]:
        groups = list(getattr(kv_cache_config, "kv_cache_groups", []) or [])
        if len(groups) != 1:
            raise RegistrationError(
                f"M1/M2 support exactly one KV cache group; engine reports "
                f"{len(groups)} groups. Hybrid engines require M4."
            )
        group = groups[0]
        layer_names, specs = _group_names_and_specs(group)

        kinds = {type(s).__name__ for s in specs}
        if not kinds <= self._PAGED_KV_SPECS:
            # A mixed-format group would show >1 distinct spec kind here.
            raise RegistrationError(
                f"M1/M2 support full-attention PAGED_KV only; group 0 specs "
                f"are {sorted(kinds)}. MLA/sliding-window/Mamba arrive in M4."
            )
        if len({_spec_identity(s) for s in specs}) > 1:
            raise RegistrationError(
                "mixed per-layer formats inside one cache group are not "
                "supported (uniform-type groups only)"
            )
        spec = specs[0]

        layout = getattr(kv_cache_config, "kv_cache_layout", None)
        if layout is None:
            raise RegistrationError(
                "kv_cache_config.kv_cache_layout is not resolved; refusing "
                "to guess the physical layout (research.md §4.3)"
            )
        if layout not in LAYOUT_FAMILIES:
            raise RegistrationError(f"unknown KV cache layout {layout!r}")

        quant = self._quant_name(spec)
        head_size_v = getattr(spec, "head_size_v", None) or spec.head_size
        if head_size_v != spec.head_size:
            raise RegistrationError(
                "diff-KV (head_size_v != head_size) is not supported"
            )

        block_size = spec.block_size
        num_heads = spec.num_kv_heads
        dtype = spec.dtype
        elem = _element_size(dtype)
        unpadded = block_size * num_heads * (spec.head_size + head_size_v) * elem
        page_bytes = spec.page_size_bytes
        if page_bytes < unpadded or page_bytes % unpadded != 0:
            raise RegistrationError(
                f"page_size_bytes={page_bytes} inconsistent with geometry "
                f"(expected >= {unpadded} and divisible)"
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
            row = block_size * elem * (spec.head_size + head_size_v)
            fragments = tuple(row for _ in range(num_heads))
        else:
            fragments = (page_bytes,)

        semantics = StateSemantics(
            kind=StateKind.PAGED_KV,
            block_size=block_size,
            tokens_per_state=1,
            slots_per_block=block_size,
            prefix_cacheable=bool(getattr(spec, "prefix_cacheable", True)),
        )
        fingerprint = FormatFingerprint(
            state_kind=StateKind.PAGED_KV,
            layout=layout,
            dtype=(str(dtype).removeprefix("torch."),),
            quant=quant,
            head_geometry=(num_heads, spec.head_size, head_size_v, block_size),
            page_bytes=(unpadded, page_bytes),
            scale_layout=None,
            page_fragments=fragments,
            version=FORMAT_VERSION,
        )
        return (
            GroupDescriptor(
                group_id=0,
                semantics=semantics,
                fingerprint=fingerprint,
                geometry=geometry,
            ),
        )

    def _quant_name(self, spec: Any) -> str:
        mode = getattr(spec, "kv_quant_mode", None)
        name = getattr(mode, "name", None)
        if name is None:
            return "NONE"
        if name != "NONE":
            raise RegistrationError(
                f"quantized KV cache ({name}) is passthrough-only from M4; "
                "M1/M2 require an unquantized cache"
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
                    raise RegistrationError(
                        f"layer {name!r} appears in multiple KVCacheTensors"
                    )
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
        row_bytes = page_bytes // spec.num_kv_heads  # one head's page run

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
                raise RegistrationError(
                    f"layer {name!r}: expected a rank-4 page view, got {shape}"
                )
            rows = shape[0]
            if rows % num_blocks != 0:
                raise RegistrationError(
                    f"layer {name!r}: view rows {rows} not divisible by "
                    f"num_blocks {num_blocks}"
                )
            ratio = rows // num_blocks
            if spec.block_size % ratio != 0:
                raise RegistrationError(
                    f"layer {name!r}: block_size {spec.block_size} not "
                    f"divisible by kernel ratio {ratio}"
                )
            tokens = spec.block_size // ratio
            if shape[2] != tokens:
                raise RegistrationError(
                    f"layer {name!r}: view tokens {shape[2]} != kernel "
                    f"block size {tokens}"
                )
            # torch strides are in ELEMENTS of the view dtype (the engine
            # registers dtype-typed views). Certification runs in element
            # units; byte addressing (data_ptr + offsets) uses stride*elem.
            elem = _element_size(getattr(view, "dtype", None) or spec.dtype)
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
                        offsets=tuple(
                            h * head_stride for h in range(spec.num_kv_heads)
                        ),
                        block_strides=tuple(
                            manager_stride for _ in range(spec.num_kv_heads)
                        ),
                        sizes=tuple(row_bytes for _ in range(spec.num_kv_heads)),
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
            raise RegistrationError(
                f"inconsistent manager block strides: {sorted(all_strides)}"
            )
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
            if (
                not layer_compact
                and layer_stride != page_bytes
                and not head_fragmented
            ):
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
        expected = _expected_view_strides(
            layout, spec, rows, num_layers, tokens, elem
        )
        if tuple(strides_e) != expected:
            raise RegistrationError(
                f"layer {name!r}: view strides {tuple(strides_e)} do not match "
                f"the certified {layout} strides {expected} "
                f"(rows={rows}, tokens={tokens}, elem={elem}); refusing "
                "uncertified layout (research.md §4.3)"
            )

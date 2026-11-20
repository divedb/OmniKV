"""The vLLM layout adapter: engine caches -> OmniKV GroupDescriptors.

Reads vLLM's authoritative metadata (`kv_cache_config` from the connector
constructor + the `kv_caches` dict from `register_kv_caches`) and produces
layout-parametric descriptors. Tensor-shape *detection* is deliberately NOT
used in M1: metadata-first with fail-fast rejection of anything uncertified
(research.md §4.3 source-of-truth priority).

Dual-form: supports both the v0.34.0 release `KVCacheGroupSpec`
(``layer_names`` + singular ``kv_cache_spec``) and main's post-#61452 form
(``kv_cache_specs`` dict + ``manager_config``). All access is duck-typed so
this module imports (and unit-tests) without vLLM installed.
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

# Layouts whose whole-page passthrough M1 covers: block-compact stride
# permutations addressable as offset + l*layer_stride + b*block_stride with
# the page itself one contiguous run. LHBNC/BHLNC fragment a (layer, block)
# page per head; rejected until run-decomposed regions land (M2).
_M1_SUPPORTED_LAYOUTS = frozenset({"LBNHC", "LBHNC", "BLNHC", "BLHNC"})

# Spec class names (duck-typed) that map onto M1's PAGED_KV semantics.
_M1_PAGED_KV_SPECS = frozenset({"FullAttentionSpec"})


def _class_mro_names(obj: Any) -> set[str]:
    return {cls.__name__ for cls in type(obj).__mro__}


def _group_names_and_specs(group: Any) -> tuple[list[str], list[Any]]:
    """Dual-form reader for one KV cache group (release vs. main API)."""
    names = getattr(group, "layer_names", None)
    if names is not None:  # v0.34.0 release: KVCacheGroupSpec
        return list(names), [group.kv_cache_spec]
    specs = getattr(group, "kv_cache_specs", None)  # main: post-#61452
    if isinstance(specs, Mapping):
        return list(specs), list(specs.values())
    raise RegistrationError(
        f"unrecognized KV cache group form: {type(group).__name__} has "
        "neither layer_names+kv_cache_spec nor kv_cache_specs"
    )


class VllmLayoutAdapter:
    """Builds GroupDescriptors from vLLM's resolved KV cache configuration.

    M1 scope (fail-fast outside it):
      - exactly one cache group;
      - ``FullAttentionSpec`` (PAGED_KV, no sliding window, no MLA);
      - unquantized cache, ``head_size_v == head_size``;
      - a block-compact layout (LBNHC/LBHNC/BLNHC/BLHNC) whose tensor
        strides certify contiguous whole pages.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index

    def describe(
        self,
        kv_caches: Mapping[str, torch.Tensor],
        kv_cache_config: Any,
    ) -> tuple[GroupDescriptor, ...]:
        groups = list(getattr(kv_cache_config, "kv_cache_groups", []) or [])
        if len(groups) != 1:
            raise RegistrationError(
                f"M1 supports exactly one KV cache group; engine reports "
                f"{len(groups)} groups. Hybrid engines require M4."
            )
        group = groups[0]
        layer_names, specs = _group_names_and_specs(group)

        spec = specs[0]
        kinds = {type(s).__name__ for s in specs}
        if not kinds <= _M1_PAGED_KV_SPECS:
            raise RegistrationError(
                f"M1 supports full-attention PAGED_KV only; group 0 specs "
                f"are {sorted(kinds)}. MLA/sliding-window/Mamba arrive in M4."
            )

        layout = getattr(kv_cache_config, "kv_cache_layout", None)
        if layout is None:
            raise RegistrationError(
                "kv_cache_config.kv_cache_layout is not resolved; refusing "
                "to guess the physical layout (research.md §4.3)"
            )
        if layout not in LAYOUT_FAMILIES:
            raise RegistrationError(f"unknown KV cache layout {layout!r}")
        if layout not in _M1_SUPPORTED_LAYOUTS:
            raise RegistrationError(
                f"layout {layout} fragments a (layer, block) page per head; "
                "not supported until run-decomposed regions (M2). Prefer "
                "LBNHC/LBHNC/BLNHC/BLHNC."
            )

        quant = self._quant_name(spec)
        head_size_v = getattr(spec, "head_size_v", None) or spec.head_size
        if head_size_v != spec.head_size:
            raise RegistrationError("diff-KV (head_size_v != head_size) is not supported in M1")

        block_size = spec.block_size
        num_heads = spec.num_kv_heads
        dtype = spec.dtype
        elem = _element_size(dtype)
        unpadded = block_size * num_heads * (spec.head_size + head_size_v) * elem
        page_bytes = spec.page_size_bytes
        if page_bytes < unpadded or page_bytes % unpadded != 0:
            raise RegistrationError(
                f"page_size_bytes={page_bytes} inconsistent with geometry "
                f"(expected >= {unpadded} and divisible; padded pages that "
                "are not whole multiples are rejected)"
            )

        geometry = self._resolve_geometry(
            kv_cache_config, kv_caches, layer_names, page_bytes, layout
        )

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
                "M1 requires an unquantized cache"
            )
        return name

    def _resolve_geometry(
        self,
        kv_cache_config: Any,
        kv_caches: Mapping[str, torch.Tensor],
        layer_names: Sequence[str],
        page_bytes: int,
        layout: str,
    ) -> PageGeometry:
        """Match the group to its KVCacheTensor and certify the strides.

        Also resolves, from the registered per-layer views, the absolute
        device base pointer of each layer's block 0 (``data_ptr`` already
        includes the view's storage offset, i.e. ``l * layer_stride``).
        """
        tensors = list(getattr(kv_cache_config, "kv_cache_tensors", []) or [])
        match = None
        for t in tensors:
            if list(getattr(t, "layers", []) or []) == list(layer_names):
                match = t
                break
        if match is None:
            raise RegistrationError(
                f"no KVCacheTensor matches the group's {len(layer_names)} layers"
            )
        layer_stride, block_stride = match.layer_stride, match.block_stride
        offset = int(getattr(match, "offset", 0) or 0)
        num_blocks = kv_cache_config.num_blocks

        # Certify: a (layer, block) page must be one contiguous run and the
        # stride arithmetic must not overlap distinct blocks' pages.
        if page_bytes > block_stride and block_stride != layer_stride:
            raise RegistrationError(
                f"page ({page_bytes} B) exceeds block stride ({block_stride} B) "
                "in a layer-compact layout; pages would overlap"
            )
        layer_compact = layout.startswith("L")
        if layer_compact and block_stride < page_bytes:
            raise RegistrationError(
                f"layer-compact layout {layout} requires block_stride >= page; "
                f"got block_stride={block_stride}, page={page_bytes}"
            )
        if not layer_compact and layer_stride != page_bytes:
            raise RegistrationError(
                f"block-outermost layout {layout} requires layer_stride == page; "
                f"got layer_stride={layer_stride}, page={page_bytes}"
            )
        if layer_stride * len(layer_names) > int(match.size) + max(
            layer_stride, block_stride
        ):
            raise RegistrationError(
                "layer stride arithmetic exceeds the backing allocation size"
            )

        missing = [n for n in layer_names if n not in kv_caches]
        if missing:
            raise RegistrationError(
                f"kv_caches is missing {len(missing)} of {len(layer_names)} "
                f"registered layers (first missing: {missing[0]!r})"
            )
        layer_ptrs = {}
        for n in layer_names:
            t = kv_caches[n]
            if t.device.index is None or t.device.index != self.device_index:
                raise RegistrationError(
                    f"layer {n!r} is on {t.device}; adapter targets device "
                    f"{self.device_index}"
                )
            layer_ptrs[n] = t.data_ptr()
        # The resolved per-layer base pointers ride along on the descriptor
        # via `geometry.storage_id` bookkeeping in the connector; store them
        # on the adapter result through the returned mapping below.
        self.last_layer_ptrs = layer_ptrs
        self.last_block_stride = block_stride

        return PageGeometry(
            num_blocks=num_blocks,
            page_bytes=page_bytes,
            layer_stride=layer_stride,
            block_stride=block_stride,
            storage_id=0,
            offset=offset,
            layer_names=tuple(layer_names),
            shared_storage=True,
        )

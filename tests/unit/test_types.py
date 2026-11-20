"""Unit tests for core value types (omnikv.types)."""

from __future__ import annotations

import pytest

from omnikv.types import (
    BlockView,
    CopyRun,
    FormatFingerprint,
    GroupDescriptor,
    PageGeometry,
    StateKind,
    StateSemantics,
)


def make_fingerprint(**over) -> FormatFingerprint:
    kwargs = {
        "state_kind": StateKind.PAGED_KV,
        "layout": "LBNHC",
        "dtype": ("bfloat16",),
        "quant": "NONE",
        "head_geometry": (2, 128, 128, 64),
        "page_bytes": (65536, 65536),
        "scale_layout": None,
    }
    kwargs.update(over)
    return FormatFingerprint(**kwargs)


def make_geometry(**over) -> PageGeometry:
    kwargs = {
        "num_blocks": 100,
        "page_bytes": 65536,
        # != num_blocks * page: mirrors the measured extensible-KV-cache
        # padding (111 extra block slots per layer; docs/TESTLOG.md).
        "layer_stride": 111 * 65536,
        "block_stride": 65536,
        "storage_id": 0,
        "offset": 0,
        "layer_names": tuple(f"layer.{i}" for i in range(28)),
    }
    kwargs.update(over)
    return PageGeometry(**kwargs)


def make_descriptor(**over) -> GroupDescriptor:
    fp = over.pop("fingerprint", make_fingerprint())
    geo = over.pop("geometry", make_geometry())
    sem = StateSemantics(
        kind=StateKind.PAGED_KV,
        block_size=64,
        tokens_per_state=1,
        slots_per_block=64,
    )
    return GroupDescriptor(group_id=0, semantics=sem, fingerprint=fp, geometry=geo)


class TestFormatFingerprint:
    def test_identity_stable_and_distinct(self):
        a, b = make_fingerprint(), make_fingerprint()
        assert a.identity == b.identity
        assert a.identity != make_fingerprint(layout="LBHNC").identity
        assert a.identity != make_fingerprint(dtype=("float16",)).identity
        assert a.identity != make_fingerprint(quant="FP8_PER_TENSOR").identity

    def test_unknown_layout_rejected(self):
        with pytest.raises(ValueError, match="unknown layout family"):
            make_fingerprint(layout="NHD")

    def test_version_bump_changes_identity(self):
        assert make_fingerprint().identity != make_fingerprint(version=2).identity


class TestPageGeometry:
    def test_offset_formula_uses_strides_not_num_blocks(self):
        # The empirically observed case: layer_stride != num_blocks * page.
        geo = make_geometry()
        assert geo.layer_block_offset(0, 0) == 0
        assert geo.layer_block_offset(1, 0) == geo.layer_stride
        assert geo.layer_block_offset(0, 1) == geo.page_bytes
        # Deriving from num_blocks would give a WRONG answer here.
        assert geo.layer_stride != geo.num_blocks * geo.page_bytes

    def test_invalid_geometry_rejected(self):
        with pytest.raises(ValueError):
            make_geometry(page_bytes=0)
        with pytest.raises(ValueError):
            make_geometry(layer_stride=-1)
        with pytest.raises(ValueError):
            make_geometry(layer_names=())


class TestCopyRun:
    def test_self_overlap_rejected(self):
        with pytest.raises(ValueError, match="self-overlap"):
            CopyRun(
                local_offset=0,
                remote_offset=0,
                fragment_size=100,
                num_fragments=3,
                local_stride=50,  # < fragment_size
                remote_stride=100,
            )

    def test_total_bytes(self):
        run = CopyRun(0, 0, 100, num_fragments=4, local_stride=100, remote_stride=128)
        assert run.total_bytes == 400


class TestBlockView:
    def test_passthrough_view_is_single_run(self):
        geo = make_geometry()
        view = make_descriptor(geometry=geo).block_view(7)
        assert view.is_single_run
        regions = view.gpu_regions(device_index=0)
        assert len(regions) == 28
        assert all(r.size == 65536 for r in regions)
        assert regions[3].offset == geo.layer_block_offset(3, 7)

    def test_run_decomposed_not_supported_yet(self):
        desc = make_descriptor()
        run = CopyRun(0, 0, 32, num_fragments=64, local_stride=32, remote_stride=64)
        view = BlockView(
            group_id=0,
            block_id=0,
            semantics=desc.semantics,
            fingerprint=desc.fingerprint,
            geometry=desc.geometry,
            runs=(run,),
        )
        assert not view.is_single_run
        with pytest.raises(NotImplementedError):
            view.gpu_regions()

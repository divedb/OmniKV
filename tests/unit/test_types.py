"""Unit tests for core value types (omnikv.types) — M2 model."""

from __future__ import annotations

import pytest

from omnikv.types import (
    CopyRun,
    FormatFingerprint,
    GroupDescriptor,
    GroupPlacement,
    LayerAddress,
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
        "page_fragments": (65536,),
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
        assert make_fingerprint().identity != make_fingerprint(version=999).identity

    def test_fragments_change_identity(self):
        frag = make_fingerprint(
            layout="LHBNC", page_fragments=(32768, 32768)
        )
        assert frag.identity != make_fingerprint(layout="LHBNC").identity

    def test_fragments_must_sum_to_payload(self):
        with pytest.raises(ValueError, match="do not sum"):
            make_fingerprint(page_fragments=(1024, 2048))


class TestPageGeometry:
    def test_offset_formula_uses_strides_not_num_blocks(self):
        # The empirically observed case: layer_stride != num_blocks * page.
        geo = make_geometry()
        assert geo.layer_block_offset(0, 0) == 0
        assert geo.layer_block_offset(1, 0) == geo.layer_stride
        assert geo.layer_block_offset(0, 1) == geo.block_stride
        assert geo.layer_stride != geo.num_blocks * geo.page_bytes

    def test_invalid_geometry_rejected(self):
        with pytest.raises(ValueError):
            make_geometry(page_bytes=0)
        with pytest.raises(ValueError):
            make_geometry(layer_stride=-1)
        with pytest.raises(ValueError):
            make_geometry(block_stride=0)
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
    def test_single_fragment_page_is_single_run(self):
        view = make_descriptor().block_view(7)
        assert view.is_single_run

    def test_fragmented_page_is_not_single_run(self):
        fp = make_fingerprint(layout="LHBNC", page_fragments=(32768, 32768))
        view = make_descriptor(fingerprint=fp).block_view(0)
        assert not view.is_single_run

    def test_requires_conversion(self):
        fp_a, fp_b = make_fingerprint(), make_fingerprint(layout="LBHNC")
        view = make_descriptor(fingerprint=fp_a).block_view(0)
        assert view.requires_conversion_to(fp_b)
        assert not view.requires_conversion_to(fp_a)


class TestGroupPlacement:
    def test_single_fragment_placement_ops(self):
        base = 0x10000
        placement = GroupPlacement(
            addresses=tuple(
                LayerAddress(
                    base_ptr=base + i * 65536,
                    offsets=(0,),
                    block_strides=(65536,),
                    sizes=(65536,),
                    layer_name=f"layer.{i}",
                )
                for i in range(2)
            ),
            page_offsets=(0, 65536),
        )
        ops = placement.gpu_ops(block_id=3)
        assert len(ops) == 2
        assert ops[0] == (base + 3 * 65536, 65536)
        assert ops[1] == (base + 65536 + 3 * 65536, 65536)
        assert placement.store_page_bytes == 2 * 65536

    def test_fragmented_placement_ops(self):
        # LHBNC-like: 2 heads, fragment stride = page/heads per block.
        head_stride = 100 * 64 * 512  # num_blocks * N * C
        placement = GroupPlacement(
            addresses=(
                LayerAddress(
                    base_ptr=0x20000,
                    offsets=(0, head_stride),
                    block_strides=(64 * 512, 64 * 512),
                    sizes=(64 * 512, 64 * 512),
                    layer_name="layer.0",
                ),
            ),
            page_offsets=(0,),
        )
        ops = placement.gpu_ops(block_id=5)
        assert len(ops) == 2
        assert ops[0] == (0x20000 + 5 * 64 * 512, 64 * 512)
        assert ops[1] == (0x20000 + head_stride + 5 * 64 * 512, 64 * 512)
        assert placement.store_page_bytes == 65536

    def test_page_offsets_must_be_dense(self):
        with pytest.raises(ValueError, match="dense"):
            GroupPlacement(
                addresses=(
                    LayerAddress(0, (0,), (128,), (128,)),
                    LayerAddress(0, (0,), (128,), (128,)),
                ),
                page_offsets=(0, 256),
            )

    def test_layer_address_validation(self):
        with pytest.raises(ValueError, match="equal length"):
            LayerAddress(0, (0, 1), (128,), (128, 128))
        with pytest.raises(ValueError, match="invalid fragment"):
            LayerAddress(0, (0,), (64,), (128,))  # stride < size

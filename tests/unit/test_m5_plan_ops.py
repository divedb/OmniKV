"""M5 unit tests: pitched 2-D op planning (`plan_batch`).

Torch-free pure-logic tests. The load-bearing property — a planned batch
moves exactly the same bytes as the op list it was built from — is checked
by interval-coverage comparison on both sides of every copy.
"""

from __future__ import annotations

import pytest

from omnikv.transfer.engine import PageCopy2DOp, PageCopyOp, coalesce_ops, plan_batch


def expand(ops):
    """(ptr, size) intervals of every copy, per side, sorted."""
    src, dst = [], []
    for op in ops:
        if isinstance(op, PageCopy2DOp):
            for i in range(op.height):
                src.append((op.src_ptr + i * op.src_pitch, op.width))
                dst.append((op.dst_ptr + i * op.dst_pitch, op.width))
        else:
            src.append((op.src_ptr, op.nbytes))
            dst.append((op.dst_ptr, op.nbytes))
    return sorted(src), sorted(dst)


def assert_same_coverage(original: list[PageCopyOp], planned: list) -> None:
    assert expand(original) == expand(planned)


class TestPlanBatchFolding:
    def test_save_pattern_folds_per_block(self):
        """The connector's save shape: block-major store slots, layer-
        strided GPU (layer-compact shared storage). B*L memcpys -> B
        pitched copies of height L."""
        L, B, page, layer_stride = 28, 64, 65536, 235_864_064
        gpu = 0x1_0000_0000
        host = 0x1000_0000
        ops = []
        for b in range(B):
            for li in range(L):
                ops.append(
                    PageCopyOp(
                        src_ptr=gpu + li * layer_stride + b * page,
                        dst_ptr=host + b * L * page + li * page,
                        nbytes=page,
                    )
                )
        planned = plan_batch(ops)
        assert all(isinstance(op, PageCopy2DOp) for op in planned)
        assert len(planned) == B
        first = planned[0]
        assert (first.width, first.height) == (page, L)
        assert first.src_pitch == layer_stride
        assert first.dst_pitch == page
        assert first.src_ptr == gpu
        assert first.dst_ptr == host
        assert_same_coverage(ops, planned)

    def test_load_pattern_folds_per_layer(self):
        """The connector's load shape with consecutive GPU blocks and
        consecutive store pages: layer-major GPU, slot-strided host.
        B*L memcpys -> L pitched copies of height B."""
        L, B, page = 12, 96, 196_608
        gpu = 0x2_0000_0000
        host = 0x2000_0000
        ops = []
        for b in range(B):
            for li in range(L):
                ops.append(
                    PageCopyOp(
                        src_ptr=host + b * L * page + li * page,
                        dst_ptr=gpu + li * B * page + b * page,
                        nbytes=page,
                    )
                )
        planned = plan_batch(ops)
        assert len(planned) == L
        first = planned[0]
        assert (first.width, first.height) == (page, B)
        assert first.src_pitch == L * page  # host slots are block-major
        assert first.dst_pitch == page
        assert_same_coverage(ops, planned)

    def test_fully_contiguous_stays_single_1d(self):
        ops = [PageCopyOp(0x1000 + i * 4096, 0x9000 + i * 4096, 4096) for i in range(8)]
        planned = plan_batch(ops)
        assert len(planned) == 1
        assert isinstance(planned[0], PageCopyOp)
        assert planned[0].nbytes == 8 * 4096

    def test_scattered_sources_barely_fold(self):
        """Loads whose store pages map in strictly decreasing order:
        planning sorts by destination (layer-major on the GPU); within a
        layer every source step is negative (no chain), and only the
        layer-boundary pairs fold incidentally (height 2). Coverage is
        preserved either way."""
        L, B, page = 4, 8, 1024
        pages = list(range(B - 1, -1, -1))  # b -> host slot B-1-b
        gpu = 0x4000_0000
        host = 0x8000_0000
        ops = []
        for b, p in enumerate(pages):
            for li in range(L):
                ops.append(
                    PageCopyOp(
                        src_ptr=host + p * L * page + li * page,
                        dst_ptr=gpu + li * B * page + b * page,
                        nbytes=page,
                    )
                )
        planned = plan_batch(ops)
        heights = [
            op.height for op in planned if isinstance(op, PageCopy2DOp)
        ]
        assert max(heights, default=1) <= 2  # no cross-block folds
        assert len(planned) == L * B - (L - 1)  # only boundary pairs fold
        assert_same_coverage(ops, planned)

    def test_unequal_sizes_break_the_chain(self):
        ops = [
            PageCopyOp(0x1000, 0x9000, 4096),
            PageCopyOp(0x1000 + 8192, 0x9000 + 4096, 2048),  # different width
            PageCopyOp(0x1000 + 8192 + 6144, 0x9000 + 4096 + 4096, 2048),
        ]
        planned = plan_batch(ops)
        assert sum(1 for op in planned if isinstance(op, PageCopy2DOp)) == 1

    def test_single_and_empty_passthrough(self):
        assert plan_batch([]) == []
        one = [PageCopyOp(1, 2, 3)]
        assert plan_batch(one) == one

    def test_dst_contiguous_src_strided_folds(self):
        """Half-contiguous patterns (dst pitch == width) are legal folds."""
        ops = [
            PageCopyOp(0x1000 + i * 8192, 0x9000 + i * 4096, 4096) for i in range(4)
        ]
        assert len(coalesce_ops(list(ops))) == 4  # 1-D coalescing cannot act
        planned = plan_batch(ops)
        assert len(planned) == 1
        first = planned[0]
        assert isinstance(first, PageCopy2DOp)
        assert first.height == 4 and first.dst_pitch == 4096 and first.src_pitch == 8192
        assert_same_coverage(ops, planned)


class TestPlanBatchInvariants:
    """Property-style checks over randomized op lists (fixed seeds)."""

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_coverage_always_preserved(self, seed):
        import random

        rng = random.Random(seed)
        for _ in range(50):
            n = rng.randint(0, 40)
            ops = []
            for _ in range(n):
                size = rng.choice([512, 1024, 4096])
                ops.append(
                    PageCopyOp(
                        src_ptr=rng.randrange(0, 1 << 28) & ~(size - 1),
                        dst_ptr=rng.randrange(0, 1 << 28) & ~(size - 1),
                        nbytes=size,
                    )
                )
            assert_same_coverage(ops, plan_batch(list(ops)))

    def test_never_more_ops_than_input(self):
        import random

        rng = random.Random(7)
        for _ in range(50):
            ops = [
                PageCopyOp(rng.randrange(1 << 24), rng.randrange(1 << 24), 1024)
                for _ in range(rng.randint(0, 30))
            ]
            assert len(plan_batch(list(ops))) <= len(ops)


class TestPageCopy2DOp:
    def test_rejects_overlapping_rows(self):
        with pytest.raises(ValueError, match="pitches must be"):
            PageCopy2DOp(src_ptr=0, dst_ptr=0, width=4096, height=3, src_pitch=1024, dst_pitch=8192)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            PageCopy2DOp(0, 0, 0, 3, 4096, 4096)
        with pytest.raises(ValueError):
            PageCopy2DOp(0, 0, 4096, 0, 4096, 4096)

    def test_nbytes_is_width_times_height(self):
        assert PageCopy2DOp(0, 0, 100, 7, 128, 128).nbytes == 700

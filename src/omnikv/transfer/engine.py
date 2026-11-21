"""Batched page-copy transfer engine (M1: stream-based, wait-before-use).

Operates purely on absolute pointers + byte sizes — it never sees tensors,
shapes, or layouts (research.md §4.5). Copies are enqueued on a dedicated
CUDA stream and completed with a single event. M1 waits on the ticket
before results become visible (publish / forward), which keeps the
correctness rules trivially satisfied while the async machinery (streams,
events, batching) is already in place for M3.

M5 adds pitched 2-D copies (`PageCopy2DOp` via `cudaMemcpy2DAsync`) and
`plan_batch`, which folds regularly-strided op sequences into single 2-D
copies — the cross-block merging the M3 audit showed plain coalescing
cannot do (store slots are block-major while layer-compact engine pools
are layer-major, so a B-block x L-layer batch submits as ~B pitched copies
instead of B*L memcpys).
"""

from __future__ import annotations

import ctypes
import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass

try:  # CUDA torch required at runtime; the coalescing logic below is
    # torch-free and unit-tested locally.
    import torch
except ImportError:  # pragma: no cover - local test hosts only
    torch = None  # type: ignore[assignment]

# cudaMemcpyKind enum values (CUDA runtime).
_MEMCPY_KIND = {"h2d": 1, "d2h": 2, "d2d": 3}


@dataclass(frozen=True)
class PageCopyOp:
    """One contiguous byte copy: (src_ptr, dst_ptr, nbytes)."""

    src_ptr: int
    dst_ptr: int
    nbytes: int


@dataclass(frozen=True)
class PageCopy2DOp:
    """One pitched 2-D copy: ``height`` rows of ``width`` bytes each.

    Row ``i`` spans ``[src_ptr + i*src_pitch, +width)`` at the source and
    ``[dst_ptr + i*dst_pitch, +width)`` at the destination. Backed by
    ``cudaMemcpy2DAsync``; pitches must be >= width (rows may not overlap).
    """

    src_ptr: int
    dst_ptr: int
    width: int
    height: int
    src_pitch: int
    dst_pitch: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.src_pitch < self.width or self.dst_pitch < self.width:
            raise ValueError("pitches must be >= width (rows must not overlap)")

    @property
    def nbytes(self) -> int:
        return self.width * self.height


def coalesce_ops(ops: list[PageCopyOp]) -> list[PageCopyOp]:
    """Merge adjacent runs into fewer, larger copies.

    Two ops merge when both their source and destination ranges are
    contiguous (the whole-block case: consecutive blocks of one layer are
    contiguous in both the store slots and the engine pool). Mirrors
    vLLM's offloading `_coalesce_runs` (research.md §2.1.6): a 64-block
    load collapses from ~layers*blocks ops to ~layers ops.
    """
    if len(ops) < 2:
        return ops
    ordered = sorted(ops, key=lambda op: op.dst_ptr)
    out: list[PageCopyOp] = []
    for op in ordered:
        prev = out[-1] if out else None
        if (
            prev is not None
            and op.dst_ptr == prev.dst_ptr + prev.nbytes
            and op.src_ptr == prev.src_ptr + prev.nbytes
        ):
            out[-1] = PageCopyOp(prev.src_ptr, prev.dst_ptr, prev.nbytes + op.nbytes)
        else:
            out.append(op)
    return out


def plan_batch(ops: list[PageCopyOp]) -> list[PageCopyOp | PageCopy2DOp]:
    """Coalesce contiguous runs, then fold regular stride patterns into
    pitched 2-D copies (research.md §8.5 "pinned copy2d").

    A maximal chain of equal-sized ops whose consecutive source and
    destination pointers advance by constant strides becomes ONE
    `PageCopy2DOp`. Fully-contiguous chains were already merged by
    `coalesce_ops`, so every fold here is a genuine submission-count win:

    - block-major store saves (layer-compact engine): per block, the L
      layer pages are contiguous in the store slot but strided by
      ``layer_stride`` on the GPU -> one 2-D op of height L per block;
    - loads with consecutive GPU blocks and consecutive store pages: per
      layer, B block pages are contiguous on the GPU but one slot apart
      in the store -> one 2-D op of height B per layer.
    """
    coalesced = coalesce_ops(ops)
    if len(coalesced) < 2:
        # list-invariance workaround: the values are valid union members.
        import typing

        return typing.cast("list[PageCopyOp | PageCopy2DOp]", list(coalesced))
    ordered = sorted(coalesced, key=lambda op: op.dst_ptr)
    out: list[PageCopyOp | PageCopy2DOp] = []
    i = 0
    while i < len(ordered):
        first = ordered[i]
        j = i + 1
        src_pitch = dst_pitch = 0
        height = 1
        while j < len(ordered):
            cand = ordered[j]
            if cand.nbytes != first.nbytes:
                break
            step_src = cand.src_ptr - ordered[j - 1].src_ptr
            step_dst = cand.dst_ptr - ordered[j - 1].dst_ptr
            if step_src < first.nbytes or step_dst < first.nbytes:
                break  # overlapping or backward rows: not a legal pitch
            if src_pitch == 0:
                src_pitch, dst_pitch = step_src, step_dst
            elif (step_src, step_dst) != (src_pitch, dst_pitch):
                break
            height += 1
            j += 1
        if height > 1:
            out.append(
                PageCopy2DOp(
                    src_ptr=first.src_ptr,
                    dst_ptr=first.dst_ptr,
                    width=first.nbytes,
                    height=height,
                    src_pitch=src_pitch,
                    dst_pitch=dst_pitch,
                )
            )
        else:
            out.append(first)
        i = j
    return out


class TransferTicket:
    """Completion handle for one submitted batch."""

    def __init__(self, event: torch.cuda.Event):
        self._event = event
        self._done = False

    def ready(self) -> bool:
        if self._done:
            return True
        self._done = self._event.query()
        return self._done

    def wait(self) -> None:
        if not self._done:
            self._event.synchronize()
            self._done = True


class RegionTransferEngine:
    """Submits batches of contiguous page copies on private streams.

    One stream per direction. Loads fence against the caller's current
    stream (they must be ordered after nothing but must complete before the
    forward reads the blocks); saves fence against the current stream so
    they are ordered after the model's producing kernels (research.md §7.4).
    """

    def __init__(self, device_index: int = 0):
        self.device = torch.device("cuda", device_index)
        self._streams = {
            "h2d": torch.cuda.Stream(device=self.device),
            "d2h": torch.cuda.Stream(device=self.device),
        }
        self._cudart = _load_cudart()
        self._memcpy = self._cudart.cudaMemcpyAsync
        self._memcpy.restype = ctypes.c_int
        self._memcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._memcpy2d = self._cudart.cudaMemcpy2DAsync
        self._memcpy2d.restype = ctypes.c_int
        self._memcpy2d.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._errstr = self._cudart.cudaGetErrorString
        self._errstr.restype = ctypes.c_char_p
        self._copies_submitted = 0
        self._bytes_submitted = 0

    def submit(
        self,
        direction: str,
        ops: Sequence[PageCopyOp | PageCopy2DOp],
        *,
        fence: bool = True,
    ) -> TransferTicket:
        """Enqueue one batch; the returned ticket must be waited on.

        Args:
            direction: "h2d" (host -> GPU) or "d2h" (GPU -> host).
            ops: contiguous `PageCopyOp` and/or pitched `PageCopy2DOp`
                copies; pointers must remain valid until wait().
            fence: order the batch after the caller's current stream.
        """
        if direction not in _MEMCPY_KIND or direction == "d2d":
            raise ValueError(f"direction must be 'h2d' or 'd2h', got {direction!r}")
        if not ops:
            raise ValueError("empty copy batch")
        stream = self._streams[direction]
        current = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            if fence:
                stream.wait_stream(current)
            for op in ops:
                if isinstance(op, PageCopy2DOp):
                    rc = self._memcpy2d(
                        ctypes.c_void_p(op.dst_ptr),
                        ctypes.c_size_t(op.dst_pitch),
                        ctypes.c_void_p(op.src_ptr),
                        ctypes.c_size_t(op.src_pitch),
                        ctypes.c_size_t(op.width),
                        ctypes.c_size_t(op.height),
                        _MEMCPY_KIND[direction],
                        ctypes.c_void_p(stream.cuda_stream),
                    )
                else:
                    rc = self._memcpy(
                        ctypes.c_void_p(op.dst_ptr),
                        ctypes.c_void_p(op.src_ptr),
                        ctypes.c_size_t(op.nbytes),
                        _MEMCPY_KIND[direction],
                        ctypes.c_void_p(stream.cuda_stream),
                    )
                if rc != 0:
                    raise RuntimeError(
                        f"cudaMemcpyAsync failed ({rc}): "
                        f"{self._errstr(rc).decode()}"
                    )
            event = torch.cuda.Event()
            event.record(stream)
        self._copies_submitted += len(ops)
        self._bytes_submitted += sum(op.nbytes for op in ops)
        return TransferTicket(event)

    @property
    def stats(self) -> tuple[int, int]:
        return self._copies_submitted, self._bytes_submitted


_CUDART: ctypes.CDLL | None = None


def _load_cudart() -> ctypes.CDLL:
    """Locate the CUDA runtime already bundled with torch."""
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    import sys

    candidates: list[str] = []
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    candidates += sorted(glob.glob(os.path.join(torch_lib, "libcudart.so*")))
    for sp in sys.path:
        if os.path.isdir(sp):
            # Wheel layouts: nvidia/cu13/lib (cu13 line) and
            # nvidia/cuda_runtime/lib (older naming).
            candidates += sorted(
                glob.glob(
                    os.path.join(sp, "nvidia", "**", "libcudart.so*"),
                    recursive=True,
                )
            )
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            if not hasattr(lib, "cudaMemcpyAsync"):
                continue
            _CUDART = lib
            return _CUDART
        except OSError:
            continue
    raise RuntimeError(
        "libcudart with cudaMemcpyAsync not found; looked in torch/lib and "
        "nvidia.cuda_runtime. The transfer engine requires a CUDA-enabled "
        "torch (run on the remote GPU host)."
    )

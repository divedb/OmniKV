"""Batched page-copy transfer engine (M1: stream-based, wait-before-use).

Operates purely on absolute pointers + byte sizes — it never sees tensors,
shapes, or layouts (research.md §4.5). Copies are enqueued on a dedicated
CUDA stream and completed with a single event. M1 waits on the ticket
before results become visible (publish / forward), which keeps the
correctness rules trivially satisfied while the async machinery (streams,
events, batching) is already in place for M3.
"""

from __future__ import annotations

import ctypes
import glob
import os
from dataclasses import dataclass

import torch

# cudaMemcpyKind enum values (CUDA runtime).
_MEMCPY_KIND = {"h2d": 1, "d2h": 2, "d2d": 3}


@dataclass(frozen=True)
class PageCopyOp:
    """One contiguous byte copy: (src_ptr, dst_ptr, nbytes)."""

    src_ptr: int
    dst_ptr: int
    nbytes: int


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
        self._errstr = self._cudart.cudaGetErrorString
        self._errstr.restype = ctypes.c_char_p
        self._copies_submitted = 0
        self._bytes_submitted = 0

    def submit(
        self, direction: str, ops: list[PageCopyOp], *, fence: bool = True
    ) -> TransferTicket:
        """Enqueue one batch; the returned ticket must be waited on.

        Args:
            direction: "h2d" (host -> GPU) or "d2h" (GPU -> host).
            ops: contiguous copies; pointers must remain valid until wait().
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
            candidates += sorted(
                glob.glob(
                    os.path.join(
                        sp, "nvidia", "cuda_runtime", "lib", "libcudart.so*"
                    )
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

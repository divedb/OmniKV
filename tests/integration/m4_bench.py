"""Per-kind transfer smoke matrix (whole-group slots, no codec or readback).

Uses synthetic 8-layer pages and reports both small and large batches. This
measures the current Python submission path, not a hardware bandwidth claim.
"""

import json
import time

import torch

from omnikv.transfer.engine import PageCopyOp, RegionTransferEngine


def main():
    engine = RegionTransferEngine()
    result = []
    for kind, page in [
        ("latent_padded", 64 * 576 * 2 + 256),
        ("sliding_window", 64 * 4 * 256 * 2 * 2),
        ("sparse_index", 64 * 132),
        ("fp8_inline", 64 * 4 * (256 + 4) * 2),
    ]:
        for blocks in (1, 64):
            size = page * 8 * blocks
            gpu = torch.empty(size, dtype=torch.uint8, device="cuda")
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            gpu.fill_(7)
            for direction in ("d2h", "h2d"):
                src, dst = (
                    (gpu.data_ptr(), host.data_ptr())
                    if direction == "d2h"
                    else (host.data_ptr(), gpu.data_ptr())
                )
                ops = [PageCopyOp(src + i * page, dst + i * page, page) for i in range(8 * blocks)]
                engine.submit(direction, ops).wait()
                start = time.perf_counter()
                for _ in range(10):
                    engine.submit(direction, ops).wait()
                elapsed = (time.perf_counter() - start) / 10
                result.append(
                    {
                        "kind": kind,
                        "blocks": blocks,
                        "direction": direction,
                        "bytes": size,
                        "ms": elapsed * 1000,
                        "GBps": size / elapsed / 1e9,
                    }
                )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

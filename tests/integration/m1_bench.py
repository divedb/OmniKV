"""M1 microbenchmark: page-copy bandwidth and per-batch overhead.

Measures the transfer engine in isolation (no model): batches of whole
block save/load copies at the geometry used by the M1 integration tests
(Qwen2.5-1.5B: 28 layers x 64 KiB pages per block; gpt2: 12 x 192 KiB).

Run on the remote GPU host:
    .venv/bin/python tests/integration/m1_bench.py
"""

from __future__ import annotations

import time

import torch

from omnikv.transfer.engine import PageCopyOp, RegionTransferEngine


def bench(name: str, layers: int, page: int, blocks: int, reps: int = 5) -> None:
    torch.cuda.init()
    engine = RegionTransferEngine(device_index=0)
    gpu = torch.empty(layers * blocks * page, dtype=torch.uint8, device="cuda")
    host = torch.empty(layers * blocks * page, dtype=torch.uint8, pin_memory=True)
    host.copy_(torch.arange(layers * blocks * page, dtype=torch.int32).to(torch.uint8) % 251)
    ops = []
    for b in range(blocks):
        for li in range(layers):
            gpu_off = (li * blocks + b) * page
            host_off = (b * layers + li) * page  # store page layout: block-major
            ops.append(
                PageCopyOp(
                    src_ptr=gpu.data_ptr() + gpu_off,
                    dst_ptr=host.data_ptr() + host_off,
                    nbytes=page,
                )
            )
    total = layers * blocks * page
    # warmup
    engine.submit("d2h", ops).wait()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        engine.submit("d2h", ops).wait()
        times.append(time.perf_counter() - t0)
    d2h = min(times)
    h2d_ops = [PageCopyOp(src_ptr=op.dst_ptr, dst_ptr=op.src_ptr, nbytes=op.nbytes) for op in ops]
    engine.submit("h2d", h2d_ops).wait()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        engine.submit("h2d", h2d_ops).wait()
        times.append(time.perf_counter() - t0)
    h2d = min(times)
    copies, bytes_moved = engine.stats
    print(
        f"{name}: {blocks} blocks x {layers} layers x {page//1024} KiB "
        f"= {total/2**20:.1f} MiB | d2h {d2h*1e3:.2f} ms "
        f"({total/d2h/2**30:.2f} GiB/s) | h2d {h2d*1e3:.2f} ms "
        f"({total/h2d/2**30:.2f} GiB/s) | {copies} copies, "
        f"{bytes_moved/2**20:.0f} MiB submitted"
    )


if __name__ == "__main__":
    bench("qwen2.5-1.5b (28x64KiB)", layers=28, page=65536, blocks=64)
    bench("qwen2.5-1.5b (28x64KiB)", layers=28, page=65536, blocks=256)
    bench("gpt2 (12x192KiB)", layers=12, page=196608, blocks=64)
    bench("gpt2 (12x192KiB)", layers=12, page=196608, blocks=256)

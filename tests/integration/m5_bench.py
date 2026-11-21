"""M5 microbenchmark: op-planning (2-D pitched copies), the hash path,
and pinned-scratch allocation cost.

Measures what M5 changed (research.md §9 M5 performance hardening):

1. save/load-shaped batches at the Qwen2.5-1.5B geometry: raw per-page
   ops vs `coalesce_ops` (M3) vs `plan_batch` (M5 pitched copies) — op
   counts, wall time, bandwidth. A byte-equality gate proves the 2-D
   copies land exactly the same bytes as the 1-D ops on this stack.
2. `chain_block_hashes` on long prompts (the scheduler lookup path).
3. `cudaHostAlloc` (per-call pinned alloc) vs the runtime's cached
   scratch — the cost the M5 scratch slots remove from every
   conversion/verify call.

Run on the remote GPU host:
    .venv/bin/python tests/integration/m5_bench.py [results.json]
"""

from __future__ import annotations

import json
import sys
import time

import torch

from omnikv.keys import chain_block_hashes
from omnikv.transfer.engine import (
    PageCopyOp,
    RegionTransferEngine,
    coalesce_ops,
    plan_batch,
)


def bench_direction(engine, direction, ops, reps=10) -> float:
    engine.submit(direction, ops).wait()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        engine.submit(direction, ops).wait()
        times.append(time.perf_counter() - t0)
    return min(times)


def transfer_bench(engine, layers: int, page: int, blocks: int) -> dict:
    """Save-shaped (block-major host) and load-shaped (layer-major GPU +
    consecutive host pages) batches at one geometry."""
    gpu = torch.empty(layers * blocks * page, dtype=torch.uint8, device="cuda")
    host = torch.empty(layers * blocks * page, dtype=torch.uint8, pin_memory=True)
    gpu.copy_(torch.arange(gpu.numel(), dtype=torch.int64).to(torch.uint8) % 251)
    total = layers * blocks * page

    save_ops = [
        PageCopyOp(
            src_ptr=gpu.data_ptr() + (li * blocks + b) * page,
            dst_ptr=host.data_ptr() + (b * layers + li) * page,
            nbytes=page,
        )
        for b in range(blocks)
        for li in range(layers)
    ]
    load_ops = [
        PageCopyOp(
            src_ptr=host.data_ptr() + (b * layers + li) * page,
            dst_ptr=gpu.data_ptr() + (li * blocks + b) * page,
            nbytes=page,
        )
        for b in range(blocks)
        for li in range(layers)
    ]
    out = {"layers": layers, "page": page, "blocks": blocks, "bytes": total}
    for label, ops, direction in (
        ("save", save_ops, "d2h"),
        ("load", load_ops, "h2d"),
    ):
        planned = plan_batch(list(ops))
        coalesced = coalesce_ops(list(ops))
        # Byte-equality gate: pitched copies must land identical bytes.
        dst = host if direction == "d2h" else gpu
        dst.zero_()
        engine.submit(direction, ops).wait()
        reference = dst.clone()
        dst.zero_()
        engine.submit(direction, planned).wait()
        assert torch.equal(dst, reference), f"{label}: planned batch moved different bytes"

        for name, batch in (("raw", ops), ("coalesced", coalesced), ("planned", planned)):
            dt = bench_direction(engine, direction, batch)
            out[f"{label}_{name}"] = {
                "ops": len(batch),
                "ms": round(dt * 1e3, 3),
                "GBps": round(total / dt / 1e9, 2),
            }
    return out


def hash_bench() -> dict:
    out = {}
    for n in (1024, 4096, 16384):
        tokens = (list(range(151_000, 151_000 + n)))
        chain_block_hashes(tokens, 64)  # warm
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            chain_block_hashes(tokens, 64)
        dt = (time.perf_counter() - t0) / reps
        out[f"tokens_{n}"] = {
            "us_per_call": round(dt * 1e6, 1),
            "us_per_token": round(dt * 1e6 / n, 3),
        }
    return out


def scratch_bench() -> dict:
    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.empty(28 * 65536, dtype=torch.uint8, pin_memory=True)
    per_call = (time.perf_counter() - t0) / reps

    class _RT:  # the runtime's cached-scratch pattern
        _scratch = None

        def pinned_scratch(self, nbytes):
            buf = _RT._scratch
            if buf is None or buf.numel() < nbytes:
                buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                _RT._scratch = buf
            return buf[:nbytes]

    rt = _RT()
    rt.pinned_scratch(1024)  # force one alloc
    t0 = time.perf_counter()
    for _ in range(reps):
        rt.pinned_scratch(28 * 65536)
    cached = (time.perf_counter() - t0) / reps
    return {
        "alloc_ms_per_call": round(per_call * 1e3, 3),
        "cached_us_per_call": round(cached * 1e6, 1),
    }


def main() -> None:
    torch.cuda.init()
    engine = RegionTransferEngine(device_index=0)
    results = {
        "transfers": [
            transfer_bench(engine, layers=28, page=65536, blocks=64),
            transfer_bench(engine, layers=28, page=65536, blocks=256),
        ],
        "hash": hash_bench(),
        "scratch": scratch_bench(),
    }
    text = json.dumps(results, indent=2)
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text + "\n")
        print(f"WROTE {sys.argv[1]}")


if __name__ == "__main__":
    main()

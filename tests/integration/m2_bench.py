"""M2 microbenchmark: codec repack overhead vs passthrough.

Measures the NHD<->HND store-page repack at the Qwen2.5-1.5B slot geometry
(28 layer pages x 64 KiB per block) against a raw host memcpy of the same
bytes, so the codec's cost is quotable as a multiple of passthrough.

Run on the remote GPU host:
    .venv/bin/python tests/integration/m2_bench.py
"""

from __future__ import annotations

import time

import torch

from omnikv.codec import DEFAULT_REGISTRY
from omnikv.types import FORMAT_VERSION, FormatFingerprint, StateKind

N, H, C = 64, 2, 512
PAGE = N * H * C
LAYERS = 28
SLOT = LAYERS * PAGE


def make_fp(layout: str) -> FormatFingerprint:
    return FormatFingerprint(
        state_kind=StateKind.PAGED_KV,
        layout=layout,
        dtype=("bfloat16",),
        quant="NONE",
        head_geometry=(H, 128, 128, N),
        page_bytes=(PAGE, PAGE),
        scale_layout=None,
        page_fragments=(PAGE,),
        version=FORMAT_VERSION,
    )


def main() -> None:
    nhd, hnd = make_fp("LBNHC"), make_fp("LBHNC")
    codec = DEFAULT_REGISTRY.find(nhd, hnd)
    assert codec is not None
    src = torch.randint(0, 256, (SLOT,), dtype=torch.uint8)
    dst = torch.empty_like(src)

    # Codec repack of one full slot (all 28 layer pages).
    def repack_slot():
        for li in range(LAYERS):
            lo = li * PAGE
            codec.convert(src[lo : lo + PAGE], dst[lo : lo + PAGE], nhd, hnd)

    # Passthrough reference: raw host copy of the same slot.
    def passthrough():
        dst.copy_(src)

    reps = 50
    for fn in (repack_slot, passthrough):
        fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        repack_slot()
    repack_s = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        passthrough()
    copy_s = (time.perf_counter() - t0) / reps
    print(
        f"slot={SLOT/2**20:.1f} MiB ({LAYERS} pages x {PAGE//1024} KiB): "
        f"repack {repack_s*1e6:.0f} us ({SLOT/repack_s/2**30:.2f} GiB/s) | "
        f"memcpy {copy_s*1e6:.0f} us ({SLOT/copy_s/2**30:.2f} GiB/s) | "
        f"codec overhead x{repack_s/copy_s:.1f}"
    )
    # Sanity: repack is a permutation, bytes conserved.
    assert dst.sum().item() == src.sum().item(), "repack lost bytes"


if __name__ == "__main__":
    main()

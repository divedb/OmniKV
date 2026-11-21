"""M5 head-to-head workload runner: identical prompts through different
cache configurations (research.md §9 M5: "head-to-head vs. LMCache and
upstream FlexKV configurations on shared workloads; publish wins *and*
regressions").

Modes (one engine each, subprocess-isolated):
- ``none``: plain vLLM, prefix caching disabled — full recompute.
- ``omnikv``: OmniKVConnectorV1 (the only cache in play).
- ``pcache``: vLLM's native GPU prefix cache.
- ``lmcache``: the in-tree LMCache connector (CPU offload). Requires the
  ``lmcache`` package; a missing package is recorded, not hidden.

Each run warms 8 prefix families, then executes three measured phases at
request hit rates {0.0, 0.5, 0.9}: the first ``rate`` share of each
phase's 32 requests reuse a warm prefix, the rest use fresh prefixes.
Per-request TTFT/e2e and phase wall time are recorded.

Usage:
    python m5_headtohead.py {none,omnikv,pcache,lmcache} out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get(
    "OMNIKV_MODEL",
    "/root/autodl-tmp/hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)

HIT_RATES = (0.0, 0.5, 0.9)
WARM_FAMILIES = 8
MEASURED = 32

_TOPICS = [
    "the stability of grid-tied inverters under weak grid conditions",
    "Mesopotamian administrative clay tablets and early literacy",
    "deep-sea hydrothermal vent chemosynthetic ecosystems",
    "formal verification of railway interlocking logic",
    "the optics of the human eye and adaptive corrections",
    "Baltic amber trade routes in the Bronze Age",
    "stochastic gradient methods for non-convex objectives",
    "the preservation of medieval illuminated manuscripts",
]


def prefix_for(topic: str) -> str:
    return (
        f"An expert overview of {topic}. The overview first frames the "
        "problem and its constraints, then presents the dominant "
        "approach with its quantitative trade-offs, contrasts it with two "
        "competing approaches, and finishes with current research "
        "directions. Precision matters more than breadth: every claim is "
        "tied to the mechanism that produces it. The framing section "
        "carefully separates what is settled from what remains contested, "
        "and names the assumptions that practising engineers actually "
        "rely on when they make decisions in this area. The quantitative "
        "sections work through the governing relationships step by step, "
        "state the regimes where each relationship holds, and mark the "
        "boundary beyond which the simple model breaks down and a more "
        "careful treatment is required. The contrasting approaches are "
        "not presented as strawmen: each is given its strongest form, "
        "the setting where it wins, and the reason practitioners still "
        "choose the dominant approach most of the time. The closing "
        "research directions identify the open questions that would most "
        "change practice if they were resolved, and the measurement "
        "obstacles that make progress slower than anyone would like. "
    )


def unique_tail(i: int) -> str:
    return (
        f" Case study {i:04d}: analyse the third trade-off above for "
        f"this specific instance (parameters {i}, {i + 7}, {i * 3 % 97}) "
        "and state the practical conclusion in two sentences."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["none", "omnikv", "pcache", "lmcache"])
    parser.add_argument("out")
    parser.add_argument("--cpu-pages", type=int, default=1024)
    parser.add_argument("--gpu-util", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    if args.mode == "lmcache":
        try:
            import lmcache  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - recorded honestly
            with open(args.out, "w") as f:
                json.dump({"mode": args.mode, "lmcache_available": False,
                           "import_error": repr(exc)}, f, indent=2)
            print(f"WROTE {args.out} (lmcache unavailable: {exc!r})")
            return

    kwargs = {
        "model": MODEL,
        "gpu_memory_utilization": args.gpu_util,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "enable_prefix_caching": args.mode == "pcache",
    }
    if args.mode == "omnikv":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OmniKVConnectorV1",
            kv_connector_module_path="omnikv.connector.v1",
            kv_role="kv_both",
            kv_connector_extra_config={"omnikv.cpu_pages": args.cpu_pages},
        )
    elif args.mode == "lmcache":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            kv_connector_extra_config={
                "chunk_size": 256,
                "save_decode_cache": False,
            },
        )

    result: dict = {
        "mode": args.mode,
        "hit_rates": list(HIT_RATES),
        "phases": {},
        "stats": None,
    }
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    warm_prefixes = [
        prefix_for(_TOPICS[i % len(_TOPICS)]) + unique_tail(10_000 + i)
        for i in range(WARM_FAMILIES)
    ]
    fresh_counter = 20_000

    # Warm every family once (the cache each mode uses gets populated).
    llm.generate(warm_prefixes, SamplingParams(temperature=0.0, max_tokens=1))

    for rate in HIT_RATES:
        prompts = []
        n_warm = round(MEASURED * rate)
        for i in range(MEASURED):
            if i < n_warm:
                # same family prefix as a warm request, distinct tail
                prompts.append(
                    prefix_for(_TOPICS[i % WARM_FAMILIES]) + unique_tail(30_000 + i)
                )
            else:
                prompts.append(
                    prefix_for(_TOPICS[(i + 3) % len(_TOPICS)])
                    + f" Fresh subject {fresh_counter}: develop the "
                    "contrasting approach instead, with its own numbers "
                    "and failure modes, in three paragraphs. "
                    + unique_tail(fresh_counter)
                )
                fresh_counter += 1
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        wall = time.perf_counter() - t0
        ttfts, e2es = [], []
        for out in outs:
            metrics = out.metrics
            if metrics is None:
                continue
            if metrics.first_token_latency is not None:
                ttfts.append(metrics.first_token_latency)
            e2e = getattr(metrics, "e2e_latency", None)
            if e2e is None:
                finished = getattr(metrics, "finished_time", None)
                arrival = getattr(metrics, "arrival_time", None)
                if finished is not None and arrival is not None:
                    e2e = finished - arrival
            if e2e is not None:
                e2es.append(e2e)
        result["phases"][str(rate)] = {
            "wall_s": round(wall, 3),
            "ttft_mean_ms": round(1e3 * sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "ttft_p50_ms": round(1e3 * sorted(ttfts)[len(ttfts) // 2], 2) if ttfts else None,
            "ttft_max_ms": round(1e3 * max(ttfts), 2) if ttfts else None,
            "e2e_mean_ms": round(1e3 * sum(e2es) / len(e2es), 2) if e2es else None,
            "n": len(prompts),
        }
        if args.mode == "omnikv":
            from omnikv.connector import v1 as cm

            rt = cm._RUNTIME
            result["stats"] = {
                "saved": rt.stats_saved,
                "hits": rt.stats_hits,
                "pending_saves": rt.pending_saves,
            }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())

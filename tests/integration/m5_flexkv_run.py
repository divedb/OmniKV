"""M5 head-to-head control: the audited FlexKV v1.0.0 checkout (the
legacy baseline environment, vLLM 0.10.1.1 — see docs/ENVIRONMENT.md).

Runs the same shared workload shape as ``m5_headtohead.py`` inside the
FlexKV virtualenv so the M5 report can publish an honest configuration-
level comparison (different vLLM version — recorded, not hidden).

The FlexKV server is configured through its own JSON config file; pass
``--flexkv-config`` (see the checkout's own tests for the schema — the
final file is filled in on the remote host where the checkout lives).

Usage (inside the FlexKV venv):
    python m5_flexkv_run.py out.json --flexkv-config /tmp/flexkv.json
    python m5_flexkv_run.py out.json --mode baseline
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

# Same workload generator as m5_headtohead.py (duplicated deliberately:
# this file must run standalone inside the FlexKV venv, which has no
# access to the OmniKV test package layout beyond this directory).
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
    parser.add_argument("out")
    parser.add_argument("--mode", choices=["flexkv", "baseline"], default="flexkv")
    parser.add_argument("--flexkv-config", default="")
    parser.add_argument("--gpu-util", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kwargs = {
        "model": MODEL,
        "gpu_memory_utilization": args.gpu_util,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }
    if args.mode == "flexkv":
        if not args.flexkv_config:
            sys.exit("--flexkv-config is required for mode=flexkv")
        os.environ.setdefault("FLEXKV_CONFIG_PATH", args.flexkv_config)
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="FlexKVConnector",
            kv_connector_module_path="flexkv.integration.vllm",
            kv_role="kv_both",
        )

    result: dict = {
        "mode": args.mode,
        "vllm_note": "FlexKV control env (vLLM 0.10.1.1, torch 2.7.1+cu128)",
        "hit_rates": list(HIT_RATES),
        "phases": {},
    }
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    warm_prefixes = [
        prefix_for(_TOPICS[i % len(_TOPICS)]) + unique_tail(10_000 + i)
        for i in range(WARM_FAMILIES)
    ]
    llm.generate(warm_prefixes, SamplingParams(temperature=0.0, max_tokens=1))

    fresh_counter = 20_000
    for rate in HIT_RATES:
        prompts = []
        n_warm = round(MEASURED * rate)
        for i in range(MEASURED):
            if i < n_warm:
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
        ttfts = []
        for out in outs:
            metrics = out.metrics
            if metrics is None:
                continue
            latency = getattr(metrics, "first_token_latency", None)
            if latency is None:
                latency = getattr(metrics, "first_token_time", None)  # 0.10.x name
            if latency is not None:
                ttfts.append(float(latency))
        result["phases"][str(rate)] = {
            "wall_s": round(wall, 3),
            "ttft_mean_ms": round(1e3 * sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "ttft_p50_ms": round(1e3 * sorted(ttfts)[len(ttfts) // 2], 2) if ttfts else None,
            "ttft_max_ms": round(1e3 * max(ttfts), 2) if ttfts else None,
            "n": len(prompts),
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())

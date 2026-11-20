"""M2 integration runner: cross-layout KV reuse through the codec.

Two sequential vLLM engines in ONE process (so the runtime store persists
between them — the M2 validation topology):

- engine 1 (native layout, LBNHC on this stack): connector run of prompt A;
  its blocks are saved under the LBNHC fingerprint.
- engine 2 (``VLLM_KV_CACHE_LAYOUT=LBHNC`` forced via env before engine 1
  in a subprocess variant — here both layouts are exercised in-process by
  constructing engine 2 with the env var set at import time is NOT
  possible; instead use the two-subprocess mode below).

Modes:
- ``save``:  run engine (native layout) + connector; generate A; dump the
             runtime stores to a file on exit.
- ``load``:  run engine (alternative layout, set VLLM_KV_CACHE_LAYOUT in the
             environment) + connector with cross_layout + preload_store;
             generate B; report codec hits and outputs.

Usage:
    python m2_run.py save out.json store.pt
    VLLM_KV_CACHE_LAYOUT=LBHNC python m2_run.py load out.json store.pt
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get(
    "OMNIKV_MODEL",
    "/root/autodl-tmp/hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)

PREFIX = (
    "The OmniKV project builds a layout-agnostic key-value cache system "
    "for large language model inference engines. Its design separates the "
    "logical identity of cached tokens from the physical layout of the "
    "tensors that store them, so that many different attention backends "
    "can share one cache implementation without hardcoding shapes. "
    "The core abstraction is a block view that describes, for every cache "
    "group, layer and block, exactly which bytes hold the state and how "
    "to move them between devices. A fingerprint of the physical format "
    "gates every transfer: pages cross between the engine and the store "
    "only when their layouts, dtypes, quantization modes and geometries "
    "match exactly, and anything unverifiable is rejected with an "
    "explicit error instead of a silent fallback. "
    "The system keeps a per-group semantic contract that says what a "
    "block means and what a correct restore requires, because recurrent "
    "states and latent attention caches cannot be treated as ordinary "
    "token-key-value pages. Ownership epochs pin blocks against reuse "
    "while transfers are in flight, and stored pages become visible to "
    "lookups only after the copy has completed, which prevents both "
    "premature eviction and torn reads. "
    "Scheduling metadata flows from the scheduler side to the worker side "
    "as block identifiers alone; the worker resolves them into physical "
    "byte regions through layout adapters built once at registration, so "
    "the hot path never parses tensor shapes. "
)
SUFFIX_A = "First, summarize the goals of the project in three short bullet points."
SUFFIX_B = "Second, explain the main risk of opaque byte copying between engines."


def serialize(out) -> dict:
    def _lp(value):
        v = getattr(value, "logprob", None)  # vLLM Logprob objects
        return float(v if v is not None else value)

    return {
        "token_ids": list(out.outputs[0].token_ids),
        "text": out.outputs[0].text,
        "logprobs": [
            {str(tok): _lp(lp) for tok, lp in (pos or {}).items()}
            for pos in (out.outputs[0].logprobs or [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save", "load"])
    parser.add_argument("out")
    parser.add_argument("store")
    parser.add_argument("--cpu-pages", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    extra = {
        "omnikv.cpu_pages": args.cpu_pages,
        "omnikv.verify_load_bytes": True,  # L1 bitwise gate
    }
    if args.mode == "load":
        extra["omnikv.cross_layout"] = True
        extra["omnikv.preload_store"] = args.store

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OmniKVConnectorV1",
            kv_connector_module_path="omnikv.connector.v1",
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        ),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32, logprobs=5)

    from omnikv.connector import v1 as connector_module

    if args.mode == "save":
        atexit.register(connector_module.dump_runtime_stores, args.store)
        out = llm.generate([PREFIX + SUFFIX_A], sp)[0]
        result = {"mode": "save", "outputs": {"a": serialize(out)}}
    else:
        out = llm.generate([PREFIX + SUFFIX_B], sp)[0]
        runtime = connector_module._RUNTIME
        result = {
            "mode": "load",
            "outputs": {"b": serialize(out)},
            "stats": {
                "hits": runtime.stats_hits,
                "codec_hits": runtime.stats_codec_hits,
                "stores": sorted(runtime.stores),
            },
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())

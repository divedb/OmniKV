"""M1 integration runner: generation with/without the OmniKV connector.

Runs as a subprocess so each vLLM instance fully releases the GPU before
the next one starts. Modes:

- ``baseline``: plain vLLM, prefix caching DISABLED — every prompt is fully
  recomputed. This is the ground truth.
- ``connector``: vLLM + OmniKVConnectorV1, prefix caching disabled (the
  only cache in play is OmniKV's). Phase A populates the store; phase B
  shares A's prefix and must hit it.

Writes a JSON result file: outputs (token ids + text) and, for the
connector mode, store statistics proving hits happened.

Usage:
    python m1_run.py baseline out.json
    python m1_run.py connector out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# In-process engine (scheduler + worker share the process) is an M1
# deployment requirement: the store runtime is a process-level singleton.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get(
    "OMNIKV_MODEL",
    "/root/autodl-tmp/hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)

# A shared ~340-token prefix (5+ blocks of 64) + distinct suffixes. Long
# enough that several *whole* blocks are identical across the two prompts.
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
PROMPTS = {
    "a": PREFIX + SUFFIX_A,
    "b": PREFIX + SUFFIX_B,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "connector", "pcache"])
    parser.add_argument("out")
    parser.add_argument("--cpu-pages", type=int, default=64)
    parser.add_argument(
        "--model",
        default=None,
        help="local model path (default: the cached Qwen2.5-1.5B snapshot)",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    model = args.model or MODEL
    kwargs = {
        "model": model,
        "gpu_memory_utilization": 0.85,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        # baseline: full recompute. connector: the ONLY cache is OmniKV's.
        # pcache: vLLM's native prefix cache (the control).
        "enable_prefix_caching": (args.mode == "pcache"),
    }
    if args.mode == "connector":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OmniKVConnectorV1",
            kv_connector_module_path="omnikv.connector.v1",
            kv_role="kv_both",
            kv_connector_extra_config={
                "omnikv.cpu_pages": args.cpu_pages,
                "omnikv.verify_load_bytes": True,  # L1 bitwise gate
            },
        )
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=32, logprobs=5)

    result: dict = {"mode": args.mode, "outputs": {}, "stats": None}

    def serialize(out):
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

    if args.mode == "baseline":
        for key, prompt in PROMPTS.items():
            out = llm.generate([prompt], sp)[0]
            result["outputs"][key] = serialize(out)
    else:
        # "connector" and "pcache" share the two-phase A/B structure: A
        # populates the cache (OmniKV store or vLLM's native prefix cache),
        # B shares the prefix and must reuse it.
        connector_module = None
        if args.mode == "connector":
            from omnikv.connector import v1 as _cm

            connector_module = _cm

        # Phase A: populate the cache.
        out = llm.generate([PROMPTS["a"]], sp)[0]
        result["outputs"]["a"] = serialize(out)
        stats_after_save = None
        if connector_module is not None:
            stats_after_save = {
                "saved": connector_module._RUNTIME.stats_saved,
                "hits": connector_module._RUNTIME.stats_hits,
            }
        # Phase B: shares the prefix; must reuse it from the cache.
        out = llm.generate([PROMPTS["b"]], sp)[0]
        result["outputs"]["b"] = serialize(out)
        if connector_module is not None:
            runtime = connector_module._RUNTIME
            stores = runtime.stores
            result["stats"] = {
                "after_save": stats_after_save,
                "final": {
                    "saved": runtime.stats_saved,
                    "hits": runtime.stats_hits,
                    "store_len": sum(len(s) for s in stores.values()),
                    "free_pages": sum(s.free_pages for s in stores.values()),
                },
            }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())

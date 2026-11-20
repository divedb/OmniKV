"""M3 integration runner: async saves, concurrency, spill, failure injection.

Modes (subprocess each, one engine per mode):

- ``concurrent``: engine + connector; phase A populates two prefix
  families; phase B issues 8 concurrent requests (4 per family) sharing
  those prefixes. Gates: per-family hits > 0, bitwise verify on, outputs
  must match a ``pcache`` control run with the same concurrent pattern.
- ``soak``: engine + connector with a TINY hot store + spill tier; 30
  alternating generate rounds cycling 6 distinct prefixes. After each
  round the store audit must pass (zero page leaks); hot evictions must
  spill and later lookups must promote. Also injects one save failure
  (monkeypatched submit) and requires the engine to survive with correct
  output for that round (best-effort save contract).
- ``pcache_concurrent``: native prefix-cache control for ``concurrent``.

Usage:
    python m3_run.py concurrent out.json
    python m3_run.py pcache_concurrent out.json
    python m3_run.py soak out.json --spill-dir DIR
"""

from __future__ import annotations

import argparse
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

P1 = (
    "Photovoltaic energy conversion in residential districts depends on "
    "inverter efficiency curves, shading analysis, and the regulatory "
    "framework for feed-in tariffs. Engineers modelling whole "
    "neighbourhoods must account for panel degradation over decades, "
    "seasonal irradiance variation, and battery storage cycling. "
    "A careful model separates capital expenditure from operating costs "
    "and treats uncertainty in future electricity prices explicitly. "
)
P2 = (
    "The history of typeface design reveals a constant tension between "
    "readability at small sizes and expressive character at display "
    "sizes. Punchcutters working in metal solved it with optical sizes; "
    "digital designers inherit the problem as hinting and subpixel "
    "rendering. A modern revival must decide which compromises to keep. "
    "Typography is engineering under constraint, not decoration. "
    "Letters carry the voice of documents. "
)
FAMILIES = {"p1": P1, "p2": P2}


def _suffix(family: str, i: int) -> str:
    return f" Write one short paragraph about application {i} of this, numbered."


def serialize(out) -> dict:
    def _lp(value):
        v = getattr(value, "logprob", None)
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
    parser.add_argument("mode", choices=["concurrent", "pcache_concurrent", "soak"])
    parser.add_argument("out")
    parser.add_argument("--cpu-pages", type=int, default=64)
    parser.add_argument("--spill-dir", default="")
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    connector = args.mode in ("concurrent", "soak")
    extra = {
        "omnikv.cpu_pages": args.cpu_pages,
        "omnikv.verify_load_bytes": True,
    }
    if args.spill_dir:
        extra["omnikv.spill_dir"] = args.spill_dir
        extra["omnikv.spill_pages"] = 8
    kwargs = {
        "model": MODEL,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 1024,
        "enforce_eager": True,
        "enable_prefix_caching": args.mode == "pcache_concurrent",
    }
    if connector:
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OmniKVConnectorV1",
            kv_connector_module_path="omnikv.connector.v1",
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        )
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=24, logprobs=5)

    result: dict = {"mode": args.mode, "outputs": {}, "stats": None}

    if args.mode in ("concurrent", "pcache_concurrent"):
        # Phase A: populate both prefix families sequentially.
        outs = llm.generate([P1 + _suffix("p1", 0), P2 + _suffix("p2", 0)], sp)
        result["outputs"]["seed_p1"] = serialize(outs[0])
        result["outputs"]["seed_p2"] = serialize(outs[1])
        # Phase B: 8 concurrent requests, 4 per family, distinct suffixes.
        prompts = [
            FAMILIES[f] + _suffix(f, i) for f in ("p1", "p2") for i in range(1, 5)
        ]
        outs = llm.generate(prompts, sp)
        for idx, out in enumerate(outs):
            result["outputs"][f"b{idx}"] = serialize(out)
    else:  # soak
        import gc

        from omnikv.connector import v1 as connector_module

        prefixes = [P1, P2, P1 + "Extra sentence one for variety. ", P2 + "Extra two. "]
        audits_ok = 0
        engine_patch = None  # (engine, orig_submit) while the injection is armed
        for r in range(args.rounds):
            if engine_patch is not None and r > args.rounds // 2 + 1:
                # The save rides the steps of the round AFTER its request
                # finishes (the M2 flush finding): stay armed through that
                # entire round, disarm at the one after it.
                engine_ref, orig_submit = engine_patch
                engine_ref.submit = orig_submit
                engine_patch = None
            # A per-round unique tail of well over one block so distinct
            # full blocks accumulate across rounds — this is what forces
            # hot-tier eviction (and thus spilling) under the tiny capacity.
            prompt = (
                prefixes[r % len(prefixes)]
                + f" Round {r}: summarize briefly. The tracking code for this "
                f"round is {r:04d} and the auditor signature is Q{r}."
                + f" Diagnostic note {r}: sensors {r}, {r + 1}, and {r + 2} "
                "reported nominal values throughout the interval with no "
                "anomalies detected. The system recalibrated itself twice "
                "during the window and both events completed cleanly. "
            )
            if r == args.rounds // 2:
                # Failure injection: the round's SAVE submission fails once
                # (d2h + fence=True is exactly the save path; loads are h2d
                # and verify read-backs use fence=False); the engine must
                # survive (best-effort save) and the audit must hold.
                engine_ref = connector_module._RUNTIME.engine
                orig_submit = engine_ref.submit
                fired = [False]

                def boom(direction, ops, _orig=orig_submit, _fired=fired,
                         *, fence=True, **k):
                    if direction == "d2h" and fence and not _fired[0]:
                        _fired[0] = True
                        raise RuntimeError("injected transfer failure")
                    return _orig(direction, ops, fence=fence, **k)

                engine_ref.submit = boom
                engine_patch = (engine_ref, orig_submit)
                llm.generate([prompt], sp)
            else:
                llm.generate([prompt], sp)
            gc.collect()
            for store in connector_module._RUNTIME.stores.values():
                store.audit()
            audits_ok += 1
        runtime = connector_module._RUNTIME

        def _cold_len(s) -> int:
            cold = getattr(s, "cold", None)
            return len(cold) if cold is not None else 0

        result["stats"] = {
            "rounds": args.rounds,
            "audits_ok": audits_ok,
            "saved": runtime.stats_saved,
            "hits": runtime.stats_hits,
            "save_failures": runtime.stats_save_failures,
            "pending_saves": runtime.pending_saves,
            "stores": sorted(runtime.stores),
            "tiers": {
                fp_id: {
                    "hot_len": len(getattr(s, "hot", s)),
                    "cold_len": _cold_len(s),
                }
                for fp_id, s in runtime.stores.items()
            },
        }
        # One final reuse round: the last prefix must hit (hot or promoted).
        last = prefixes[(args.rounds - 1) % len(prefixes)]
        out = llm.generate([last + " Final reuse check."], sp)[0]
        result["outputs"]["final"] = serialize(out)

    if connector:
        from omnikv.connector import v1 as connector_module

        runtime = connector_module._RUNTIME
        if result["stats"] is None:
            result["stats"] = {
                "saved": runtime.stats_saved,
                "hits": runtime.stats_hits,
                "save_failures": runtime.stats_save_failures,
                "pending_saves": runtime.pending_saves,
            }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())

"""M5 integration runner: T6 concurrency/overhead sweep and T9 preemption
stress (remote GPU only).

Modes (subprocess each, one engine per mode):

- ``sweep``: connector engine; phase A seeds 16 prefix families, phase B
  issues ``--requests`` (default 128) concurrent requests sharing those
  prefixes. Scheduler-side connector hooks are timed per engine step
  (the research.md §8 match-phase budget); gates: hits > 0, per-step
  overhead reported against the 1 ms target.
- ``pcache_sweep``: identical workload on vLLM's native prefix cache
  (the correctness control for ``sweep``).
- ``preempt``: KV-starved engine (low ``--gpu-util``) forced into
  preemptive recomputation under a concurrent long-prompt batch; store
  audits must stay clean and a follow-up round must still hit the store.

Usage:
    python m5_run.py sweep out.json [--requests 128]
    python m5_run.py pcache_sweep out.json
    python m5_run.py preempt out.json [--gpu-util 0.30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL = os.environ.get(
    "OMNIKV_MODEL",
    "/root/autodl-tmp/hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)

_TOPICS = [
    "distributed consensus under partial network partitions",
    "the metallurgy of high-pressure turbine blades",
    "wetland hydrology and seasonal bird migration patterns",
    "the evolution of maritime insurance law in northern Europe",
    "numerical methods for stiff ordinary differential equations",
    "the chemistry of alkaline glazes in Song dynasty ceramics",
    "urban heat islands and nocturnal temperature inversion",
    "the transmission history of Ptolemy's Almagest",
    "photonic crystal fibers and dispersion engineering",
    "crop rotation records in medieval English manorial accounts",
    "the acoustics of Renaissance Venetian churches",
    "protein folding energy landscapes and kinetic traps",
    "railway signalling interlocking logic and fail-safe design",
    "the cartography of the East African rift lakes",
    "quantum error correction thresholds for surface codes",
    "the economics of container shipping alliances",
]


def prefix_for(topic: str) -> str:
    return (
        f"A detailed technical essay on {topic}. The essay develops the "
        "subject systematically: it begins with the historical context, "
        "then identifies the central mechanism, works through two or "
        "three concrete examples with numbers, and closes with the "
        "present-day frontier and its open problems. Throughout, the "
        "essay is precise about which claims are established results and "
        "which are working hypotheses held by practitioners in the field. "
    )


def sweep_prompts(n_requests: int):
    """(phase_a_prompts, phase_b_prompts): 16 seeded families; phase B
    shares each family's prefix with distinct suffixes."""
    families = [_prefix_seed(i) for i in range(len(_TOPICS))]
    phase_a = [f + f" Summary number {i:03d}." for i, f in enumerate(families)]
    per_family = n_requests // len(families)
    phase_b = []
    for i, f in enumerate(families):
        for j in range(per_family):
            phase_b.append(f + f" Continuation {i:02d}-{j:02d}: one short paragraph.")
    return phase_a, phase_b


def _prefix_seed(i: int) -> str:
    return prefix_for(_TOPICS[i])


class _Instrument:
    """Class-level timing shims around the connector's scheduler/worker
    hooks. Per-step scheduler time is accumulated between
    `build_connector_meta` calls (the engine-step boundary)."""

    def __init__(self):
        self.per_step_ms: list[float] = []
        self._current_ms = 0.0
        self.worker_ms = 0.0
        self.hooks: Counter = Counter()
        self._patched: list[tuple[type, str, object]] = []

    def install(self) -> None:
        from omnikv.connector import v1 as cm

        cls = cm.OmniKVConnectorV1
        sched_hooks = [
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "request_finished",
            "request_finished_all_groups",
            "get_loaded_kv_cache_group_ids",
        ]
        worker_hooks = [
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
            "get_transfer_results",
        ]
        for name in sched_hooks + ["build_connector_meta"] + worker_hooks:
            orig = getattr(cls, name)
            is_step_boundary = name == "build_connector_meta"
            is_worker = name in worker_hooks
            inst = self

            def timed(self_conn, *a, _orig=orig, _name=name,
                      _boundary=is_step_boundary, _worker=is_worker,
                      _inst=inst, **k):
                t0 = time.perf_counter()
                try:
                    return _orig(self_conn, *a, **k)
                finally:
                    dt = (time.perf_counter() - t0) * 1e3
                    _inst.hooks[_name] += 1
                    if _worker:
                        _inst.worker_ms += dt
                    else:
                        _inst._current_ms += dt
                        if _boundary:
                            _inst.per_step_ms.append(_inst._current_ms)
                            _inst._current_ms = 0.0

            setattr(cls, name, timed)
            self._patched.append((cls, name, orig))

    def uninstall(self) -> None:
        for cls, name, orig in self._patched:
            setattr(cls, name, orig)
        self._patched.clear()
        if self._current_ms > 0:  # trailing partial step
            self.per_step_ms.append(self._current_ms)
            self._current_ms = 0.0

    def summary(self) -> dict:
        steps = self.per_step_ms
        return {
            "steps": len(steps),
            "sched_total_ms": round(sum(steps), 3),
            "sched_mean_step_ms": round(sum(steps) / len(steps), 4) if steps else None,
            "sched_max_step_ms": round(max(steps), 4) if steps else None,
            "worker_total_ms": round(self.worker_ms, 3),
            "hook_calls": dict(self.hooks),
        }


class _PreemptCounter:
    """Counts scheduler preemption log records (evidence that the stress
    actually preempted requests). vLLM V1 logs preemptions from the
    scheduler logger; the handler matches 'preempt' in any vllm logger."""

    def __init__(self):
        import logging

        self.count = 0
        handler = logging.Handler()
        handler.emit = lambda record: self._on_record(record)  # type: ignore[method-assign]
        self.handler = handler

    def _on_record(self, record) -> None:
        if "preempt" in record.getMessage().lower():
            self.count += 1

    def install(self) -> None:
        import logging

        self._loggers = [logging.getLogger("vllm")]
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("vllm"):
                self._loggers.append(logging.getLogger(name))
        for lg in self._loggers:
            lg.addHandler(self.handler)

    def uninstall(self) -> None:
        for lg in getattr(self, "_loggers", []):
            lg.removeHandler(self.handler)


def serialize(out) -> dict:
    def _lp(value):
        v = getattr(value, "logprob", None)
        return float(v if v is not None else value)

    metrics = out.metrics
    return {
        "token_ids": list(out.outputs[0].token_ids),
        "ttft": metrics.first_token_latency if metrics else None,
        "logprobs": [
            {str(tok): _lp(lp) for tok, lp in (pos or {}).items()}
            for pos in (out.outputs[0].logprobs or [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["sweep", "pcache_sweep", "preempt"]
    )
    parser.add_argument("out")
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--cpu-pages", type=int, default=1024)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--families", type=int, default=16)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    connector = args.mode in ("sweep", "preempt")
    extra = {
        "omnikv.cpu_pages": args.cpu_pages,
    }
    if args.verify:
        extra["omnikv.verify_load_bytes"] = True
    kwargs = {
        "model": MODEL,
        "gpu_memory_utilization": args.gpu_util,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "enable_prefix_caching": args.mode == "pcache_sweep",
    }
    if connector:
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OmniKVConnectorV1",
            kv_connector_module_path="omnikv.connector.v1",
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        )
    instrument = None
    preempt_counter = None
    if args.mode == "sweep":
        instrument = _Instrument()
        instrument.install()
    if args.mode == "preempt":
        preempt_counter = _PreemptCounter()
        preempt_counter.install()

    result: dict = {"mode": args.mode, "outputs": {}, "stats": None}
    try:
        llm = LLM(**kwargs)
        if preempt_counter is not None:
            # Preemption records are debug-level in some vLLM builds;
            # raise just the scheduler logger (the engine stays at
            # WARNING, so this does not flood the run).
            import logging

            for name in (
                "vllm.v1.core.sched.scheduler",
                "vllm.v1.engine.core",
            ):
                logging.getLogger(name).setLevel(logging.DEBUG)
        sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)

        if args.mode in ("sweep", "pcache_sweep"):
            phase_a, phase_b = sweep_prompts(args.requests)
            for i, out in enumerate(llm.generate(phase_a, sp)):
                result["outputs"][f"seed{i}"] = serialize(out)
            t0 = time.perf_counter()
            for i, out in enumerate(llm.generate(phase_b, sp)):
                result["outputs"][f"b{i}"] = serialize(out)
            result["phase_b_wall_s"] = round(time.perf_counter() - t0, 3)
            ttfts = [
                o["ttft"] for o in result["outputs"].values() if o["ttft"] is not None
            ]
            result["ttft"] = {
                "n": len(ttfts),
                "mean_ms": round(1e3 * sum(ttfts) / len(ttfts), 2) if ttfts else None,
                "p50_ms": round(1e3 * sorted(ttfts)[len(ttfts) // 2], 2)
                if ttfts
                else None,
                "max_ms": round(1e3 * max(ttfts), 2) if ttfts else None,
            }
        else:  # preempt
            # Long shared prefixes x unique tails: the concurrent running
            # sequence demand (~34k tokens across 80 requests) exceeds the
            # starved KV budget, forcing preemptive recomputation.
            families = [_prefix_seed(i) for i in range(4)]
            filler = (
                " The discussion must also cover the interactions with "
                "adjacent subsystems, the measurement difficulties that "
                "practitioners report, and at least one documented incident "
                "where the naive approach failed expensively, including the "
                "post-incident analysis and the three corrective actions "
                "that were adopted afterwards by the whole industry. "
            )
            prompts = []
            for i, f in enumerate(families):
                for j in range(20):
                    prompts.append(
                        f + filler * 2 + f" Load test {i:02d}-{j:02d}."
                    )
            sp_preempt = SamplingParams(temperature=0.0, max_tokens=32)
            outs = llm.generate(prompts, sp_preempt)
            for i, out in enumerate(outs):
                result["outputs"][f"p{i}"] = serialize(out)
            result["preemption_log_records"] = preempt_counter.count
            # Best-effort KV-budget evidence from the engine config.
            try:
                cache_config = llm.llm_engine.cache_config
                result["kv_budget"] = {
                    "num_gpu_blocks": getattr(cache_config, "num_gpu_blocks", None),
                    "block_size": getattr(cache_config, "block_size", None),
                    "prompt_tokens_total": sum(
                        len(p) // 3 for p in prompts
                    ),  # rough token estimate, evidence only
                }
            except Exception:  # noqa: BLE001 - evidence field only
                result["kv_budget"] = None
            # Follow-up: the same prompts must now hit the store (the
            # preemption path must not have corrupted or leaked it).
            outs2 = llm.generate(prompts[:16], sp_preempt)
            for i, out in enumerate(outs2):
                result["outputs"][f"again{i}"] = serialize(out)

        if connector:
            from omnikv.connector import v1 as cm

            runtime = cm._RUNTIME
            result["stats"] = {
                "saved": runtime.stats_saved,
                "hits": runtime.stats_hits,
                "save_failures": runtime.stats_save_failures,
                "pending_saves": runtime.pending_saves,
            }
            for store in runtime.stores.values():
                store.audit()
            result["stats"]["audits_ok"] = True
        if instrument is not None:
            result["overhead"] = instrument.summary()
    finally:
        if instrument is not None:
            instrument.uninstall()
        if preempt_counter is not None:
            preempt_counter.uninstall()

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    sys.exit(main())

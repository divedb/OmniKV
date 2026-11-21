"""M5 end-to-end matrix: T6 concurrency/overhead, T9 preemption stress,
extended soak (remote GPU only). Corresponds to research.md §9 M5
correctness scope ("full matrix incl. soak + fuzz") on top of the M1-M4
suites, which remain the regression gates.

Run on the remote host:
    .venv/bin/pytest tests/integration/test_m5_matrix.py -v -s
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable

MODEL_ENV = {
    "HF_HOME": "/root/autodl-tmp/hf",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _run(cmd: list[str], timeout: int = 3600) -> None:
    proc = subprocess.run(  # noqa: PLW1510 - rc checked below
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=timeout,
        env={**os.environ, **MODEL_ENV},
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        pytest.fail(f"run failed (rc={proc.returncode}): {' '.join(cmd)}")


def _load(name: str) -> dict:
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


@pytest.mark.order(1)
def test_sweep_128_concurrent_matches_native_control():
    """T6 at the M1 target scale: 128 concurrent requests (16 prefix
    families) restore through OmniKV while saves drain asynchronously;
    outputs must match the native-cache control, the stores must audit
    clean, and the measured scheduler-side connector overhead is reported
    against the 1 ms/step budget (research.md §9 M1 benchmark criterion,
    discharged here at M5 scale)."""
    _run([PY, os.path.join(HERE, "m5_run.py"), "sweep",
          os.path.join(HERE, "m5_sweep.json"), "--requests", "128"])
    _run([PY, os.path.join(HERE, "m5_run.py"), "pcache_sweep",
          os.path.join(HERE, "m5_sweep_pc.json"), "--requests", "128"])

    conn = _load("m5_sweep.json")
    pc = _load("m5_sweep_pc.json")

    stats = conn["stats"]
    assert stats["hits"] >= 16, (
        f"the 128 concurrent requests must restore seeded family prefixes, "
        f"got {stats['hits']} hit blocks"
    )
    assert stats["pending_saves"] == 0, "async saves must drain"
    assert stats["audits_ok"] is True

    mismatches = [
        key
        for key in conn["outputs"]
        if conn["outputs"][key]["token_ids"] != pc["outputs"][key]["token_ids"]
    ]
    assert not mismatches, (
        f"outputs diverged from the native-cache control for {mismatches[:5]} "
        f"(of {len(mismatches)})"
    )

    overhead = conn["overhead"]
    mean_ms = overhead["sched_mean_step_ms"]
    print(
        f"\n[M5 T6] 128 concurrent requests == native-cache control; "
        f"hits={stats['hits']}; connector scheduler overhead "
        f"mean={mean_ms} ms/step, max={overhead['sched_max_step_ms']} ms "
        f"over {overhead['steps']} steps; worker hooks "
        f"{overhead['worker_total_ms']} ms total; TTFT(mean) "
        f"{conn['ttft']['mean_ms']} ms vs pcache "
        f"{pc['ttft']['mean_ms']} ms."
    )
    assert mean_ms <= 1.0, (
        f"per-step scheduler overhead {mean_ms} ms exceeds the 1 ms budget"
    )


@pytest.mark.order(2)
def test_preemption_stress_survives_and_reuses():
    """T9: KV-starved engine preempting requests by recomputation while
    the connector saves/restores. Gates: every request completes, store
    audits clean, saves drain, and a follow-up round of the same prompts
    still hits the store (preemption must not corrupt or leak it).
    Preemption evidence (log records / KV budget) is reported, not
    gated on — the correctness contract is what matters."""
    _run([PY, os.path.join(HERE, "m5_run.py"), "preempt",
          os.path.join(HERE, "m5_preempt.json"), "--gpu-util", "0.30"])

    res = _load("m5_preempt.json")
    stats = res["stats"]
    assert stats["audits_ok"] is True
    assert stats["pending_saves"] == 0
    assert stats["hits"] > 0, "the follow-up round must reuse saved pages"
    short = [
        key
        for key, out in res["outputs"].items()
        if len(out["token_ids"]) < 32
    ]
    assert not short, f"requests did not complete: {short[:5]}"
    print(
        f"\n[M5 T9] preemption stress: {len(res['outputs'])} requests "
        f"completed, preemption log records="
        f"{res.get('preemption_log_records')}, kv budget="
        f"{res.get('kv_budget')}, hits={stats['hits']}, "
        f"saved={stats['saved']}."
    )


@pytest.mark.order(3)
def test_soak_100_rounds_clean():
    """The M3 soak at M5 scale: 100 rounds on a 4-page hot tier with the
    spill active, one injected save failure, per-round audits."""
    spill_dir = os.path.join(HERE, "m5_spill")
    os.makedirs(spill_dir, exist_ok=True)
    _run(
        [PY, os.path.join(HERE, "m3_run.py"), "soak",
         os.path.join(HERE, "m5_soak.json"),
         "--cpu-pages", "4", "--spill-dir", spill_dir, "--rounds", "100"]
    )
    soak = _load("m5_soak.json")
    stats = soak["stats"]
    assert stats["audits_ok"] == 100, "every round must pass the store audit"
    assert stats["save_failures"] >= 1, "the injected failure must be counted"
    assert stats["pending_saves"] == 0
    hot_lens = [t["hot_len"] for t in stats["tiers"].values()]
    assert max(hot_lens) <= 4, "hot tier must respect its capacity"
    assert stats["hits"] > 0, "later rounds must reuse (hot or promoted)"
    print(
        f"\n[M5 soak] 100/100 audits clean, hits={stats['hits']}, "
        f"saved={stats['saved']}, tiers={stats['tiers']}."
    )

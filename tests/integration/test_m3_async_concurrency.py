"""M3 end-to-end: async saves, concurrency, spill tier, failure injection
(remote GPU only). Corresponds to research.md §10 suites T6-T9.

Run on the remote host:
    .venv/bin/python -m pytest tests/integration/test_m3_async_concurrency.py -v -s
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


def _run(cmd: list[str], timeout: int = 900) -> None:
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


def _first_div(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


@pytest.mark.order(1)
def test_concurrent_reuse_matches_native_control():
    """T6 + T7: concurrent shared-prefix requests restore through OmniKV
    with async saves in flight; outputs match the native-cache control
    that runs the identical concurrent pattern."""
    _run([PY, os.path.join(HERE, "m3_run.py"), "concurrent",
          os.path.join(HERE, "m3_conc.json")])
    _run([PY, os.path.join(HERE, "m3_run.py"), "pcache_concurrent",
          os.path.join(HERE, "m3_conc_pc.json")])

    conn = _load("m3_conc.json")
    pc = _load("m3_conc_pc.json")

    stats = conn["stats"]
    assert stats["hits"] >= 8, (
        f"the 8 concurrent requests must restore shared prefixes, got "
        f"{stats['hits']} hit blocks"
    )
    assert stats["pending_saves"] == 0, "async saves must drain"
    # Seeds are the first computation in both engines.
    for seed in ("seed_p1", "seed_p2"):
        assert (
            conn["outputs"][seed]["token_ids"] == pc["outputs"][seed]["token_ids"]
        ), f"{seed} diverged between engines"
    # Concurrent requests restore the prefix in both engines; token
    # identity expected (same engine config, deterministic kernels).
    mismatches = []
    for i in range(8):
        key = f"b{i}"
        if conn["outputs"][key]["token_ids"] != pc["outputs"][key]["token_ids"]:
            mismatches.append(key)
    assert not mismatches, (
        f"concurrent outputs diverged from the native-cache control for "
        f"{mismatches}; first divergence at "
        f"{_first_div(conn['outputs'][mismatches[0]]['token_ids'], pc['outputs'][mismatches[0]]['token_ids'])}"
    )
    print(f"\n[M3] 8 concurrent requests == native-cache control; hits={stats['hits']}.")


@pytest.mark.order(2)
def test_soak_spill_failure_injection_no_leaks():
    """T9 + spill + failure injection: 30 alternating rounds on a tiny hot
    store (8 pages) with a disk spill tier; store audits every round; one
    save submission fails mid-run and the engine survives (best-effort
    saves); the final round still reuses cache."""
    spill_dir = os.path.join(HERE, "m3_spill")
    os.makedirs(spill_dir, exist_ok=True)
    _run(
        [PY, os.path.join(HERE, "m3_run.py"), "soak",
         os.path.join(HERE, "m3_soak.json"),
         "--cpu-pages", "4", "--spill-dir", spill_dir, "--rounds", "30"]
    )
    soak = _load("m3_soak.json")
    stats = soak["stats"]
    assert stats["audits_ok"] == 30, "every round must pass the store audit"
    assert stats["save_failures"] >= 1, "the injected failure must be counted"
    assert stats["pending_saves"] == 0
    hot_lens = [t["hot_len"] for t in stats["tiers"].values()]
    cold_lens = [t["cold_len"] for t in stats["tiers"].values()]
    assert max(hot_lens) <= 4, "hot tier must respect its capacity"
    assert sum(cold_lens) > 0, "pressure must have spilled pages to disk"
    assert stats["hits"] > 0, "later rounds must reuse (hot or promoted)"
    print(
        f"\n[M3] soak: 30/30 audits clean, save_failure survived, "
        f"hot={hot_lens} cold={cold_lens}, hits={stats['hits']}."
    )

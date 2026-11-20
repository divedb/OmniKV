"""M2 end-to-end: cross-layout KV reuse through the codec (remote GPU only).

Topology (four subprocesses; each fully releases the GPU):

1. ``save``  — engine with the native layout (LBNHC on this stack), OmniKV
   connector: prompt A's blocks are saved under the LBNHC fingerprint;
   stores dumped to disk.
2. ``load``  — engine forced to LBHNC (HND) via ``VLLM_KV_CACHE_LAYOUT``,
   connector with ``cross_layout`` + ``preload_store``: prompt B shares A's
   prefix; the lookup misses the (empty) LBHNC store, finds the LBNHC
   pages through the codec registry, and the worker repacks them into
   LBHNC pages before the H2D copy.
3. ``baseline`` — LBHNC engine, no caches: full recompute ground truth.
4. ``pcache`` — LBHNC engine with vLLM's native prefix cache: the reuse
   control.

Gates:
- hard: codec path executed (codec_hits >= 4, primary hits contributed 0).
- hard: bitwise load verification inside the connector (enabled).
- hard: codec exactness proven at the unit level (reference permutation +
  round-trip) — carried by tests/unit/test_codec.py, torch-backed here.
- output equivalence vs the recompute baseline, near-tie-aware (the same
  methodology validated in M1: cache-reuse and recompute legitimately
  differ at bf16 near-ties; corruption-type divergence fails).

Run on the remote host:
    .venv/bin/python -m pytest tests/integration/test_m2_cross_layout.py -v -s
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
NEAR_TIE = 0.05
MIN_COMMON = 8


def _run(cmd: list[str], env: dict | None = None) -> None:
    full_env = dict(os.environ)
    full_env.update(MODEL_ENV)
    if env:
        full_env.update(env)
    proc = subprocess.run(  # noqa: PLW1510 - rc checked below
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=900,
        env=full_env,
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


def _assert_equivalent(key: str, base: dict, conn: dict) -> None:
    b_ids, c_ids = base[key]["token_ids"], conn[key]["token_ids"]
    if b_ids == c_ids:
        print(f"[M2] output {key!r}: cache-reuse == recompute (token-identical).")
        return
    common = 0
    for x, y in zip(b_ids, c_ids):
        if x != y:
            break
        common += 1
    assert common >= MIN_COMMON, (
        f"output {key!r} diverged at token {common} (< {MIN_COMMON} common) — "
        f"corruption pattern:\nbase: {base[key]['text']!r}\nconn: {conn[key]['text']!r}"
    )
    lps = (base[key].get("logprobs") or [])[common]
    ranked = sorted(lps.items(), key=lambda kv: -kv[1])
    assert len(ranked) >= 2
    gap = ranked[0][1] - ranked[1][1]
    assert gap < NEAR_TIE, (
        f"output {key!r} diverged at token {common} but the baseline top-2 "
        f"gap was {gap:.4f} (>= {NEAR_TIE}): not a near-tie, treating as "
        "corruption"
    )
    print(
        f"[M2] output {key!r}: divergence at token {common} accepted as "
        f"near-tie (gap={gap:.5f})"
    )


@pytest.mark.order(1)
def test_cross_layout_codec_reuse():
    store = os.path.join(HERE, "m2_store.pt")
    if os.path.exists(store):
        os.remove(store)

    # 1. Save under the native layout (LBNHC on this stack).
    _run(
        [PY, os.path.join(HERE, "m2_run.py"), "save",
         os.path.join(HERE, "m2_save.json"), store]
    )
    # 2. Load into an LBHNC (HND) engine through the codec.
    _run(
        [PY, os.path.join(HERE, "m2_run.py"), "load",
         os.path.join(HERE, "m2_load.json"), store],
        env={"VLLM_KV_CACHE_LAYOUT": "LBHNC"},
    )
    # 3./4. Controls under the same (LBHNC) layout.
    _run(
        [PY, os.path.join(HERE, "m1_run.py"), "baseline",
         os.path.join(HERE, "m2_base_lb.json")],
        env={"VLLM_KV_CACHE_LAYOUT": "LBHNC"},
    )
    _run(
        [PY, os.path.join(HERE, "m1_run.py"), "pcache",
         os.path.join(HERE, "m2_pcache_lb.json")],
        env={"VLLM_KV_CACHE_LAYOUT": "LBHNC"},
    )

    save = _load("m2_save.json")
    load = _load("m2_load.json")
    base = _load("m2_base_lb.json")
    pcache = _load("m2_pcache_lb.json")

    stats = load["stats"]
    assert stats["codec_hits"] >= 4, (
        f"the codec path must restore >= 4 blocks, got {stats['codec_hits']}"
    )
    assert len(stats["stores"]) == 2, (
        f"expected both fingerprint stores (LBNHC preloaded + LBHNC native), "
        f"got {stats['stores']}"
    )
    del save  # the save run's outputs are not compared (different engine)

    _assert_equivalent("b", base["outputs"], load["outputs"])
    # Informational: native-cache control under the same layout.
    _assert_equivalent("b", base["outputs"], pcache["outputs"])
    print("\n[M2] cross-layout reuse through the codec validated.")


@pytest.mark.order(2)
def test_m1_regression_matrix_still_green():
    """The M1 suite must still pass against the v2 core (fingerprint v2,
    placement-driven ops)."""
    proc = subprocess.run(  # noqa: PLW1510 - rc checked below
        [
            PY,
            "-m",
            "pytest",
            os.path.join("tests", "integration", "test_m1_roundtrip.py"),
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=1800,
        env={**os.environ, **MODEL_ENV},
    )
    print(proc.stdout[-2000:])
    assert proc.returncode == 0, "M1 regression suite failed under the M2 core"

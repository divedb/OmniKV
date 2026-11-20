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
- hard: the codec path executed (codec_hits >= 4, two fingerprint stores).
- hard: bitwise load verification inside the connector (enabled).
- hard: codec exactness at the unit level (tests/unit/test_codec.py).
- hard: the codec-restored continuation matches the native prefix-cache
  control token-for-token (identical engine config + deterministic kernels
  make producer bytes equal modulo the layout permutation). A divergence
  fails outright.
- drift gate: cache-vs-recompute divergence (inherent bf16 prefix reuse,
  established in M1) must hit both cache paths at the SAME position;
  asymmetric divergence indicates an OmniKV defect.

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
    del save  # the save run's outputs are not compared (different engine)

    stats = load["stats"]
    assert stats["codec_hits"] >= 4, (
        f"the codec path must restore >= 4 blocks, got {stats['codec_hits']}"
    )
    assert len(stats["stores"]) == 2, (
        f"expected both fingerprint stores (LBNHC preloaded + LBHNC native), "
        f"got {stats['stores']}"
    )

    # Gate 1 (decisive): the codec-restored continuation matches vLLM's own
    # native prefix-cache reuse under the destination layout.
    cross_ids = load["outputs"]["b"]["token_ids"]
    pcache_ids = pcache["outputs"]["b"]["token_ids"]
    base_ids = base["outputs"]["b"]["token_ids"]
    assert cross_ids == pcache_ids, (
        f"codec-restored output differs from the native-cache control at "
        f"token {_first_div(cross_ids, pcache_ids)}:\n"
        f"cross:  {load['outputs']['b']['text']!r}\n"
        f"pcache: {pcache['outputs']['b']['text']!r}"
    )
    print("\n[M2] cross-layout restore == native LBHNC prefix cache (token-identical).")

    # Gate 2: cache-vs-recompute drift must affect both cache paths at the
    # same position (asymmetric divergence indicates an OmniKV defect).
    cross_div = _first_div(cross_ids, base_ids)
    pcache_div = _first_div(pcache_ids, base_ids)
    assert cross_div == pcache_div, (
        f"asymmetric divergence from recompute: cross at {cross_div}, "
        f"native cache at {pcache_div} — indicates an OmniKV defect"
    )
    if cross_div is not None:
        print(
            f"[M2] both cache paths diverge from recompute at token "
            f"{cross_div} (identical positions; inherent prefix-reuse drift)."
        )
    else:
        print("[M2] both cache paths match recompute token-for-token.")
    print("[M2] cross-layout reuse through the codec validated.")


@pytest.mark.order(2)
def test_m1_regression_matrix_still_green():
    """The M1 suite must still pass against the v2 core (fingerprint v2,
    placement-driven ops). Run the M1 file directly in a subprocess."""
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

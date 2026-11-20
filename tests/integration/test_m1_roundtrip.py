"""M1 end-to-end correctness test (remote GPU only).

Three-run design (research.md §10, L1–L3 adapted to M1):

- ``baseline``: plain vLLM, prefix caching disabled — full recompute. The
  ground truth.
- ``connector``: vLLM + OmniKVConnectorV1, prefix caching disabled. Phase A
  populates the store; phase B shares the prefix and restores it from
  OmniKV (bitwise GPU read-back verification enabled in the connector).
- ``pcache``: plain vLLM with its NATIVE prefix cache, same A/B phases.
  The control: it reuses A's block bytes exactly like OmniKV does.

Gates:
1. hard: connector-B is token-identical to pcache-B — OmniKV's restore
   reproduces vLLM's own cache-reuse semantics exactly.
2. hard: connector-A is token-identical to baseline-A (no restore involved).
3. informational/soft: divergence between cache-reuse runs (connector-B or
   pcache-B) and the recompute baseline is inherent to prefix reuse under
   bf16 (reused bytes were produced under a different chunk shape); it must
   affect BOTH cache paths identically, else it is corruption.

Run on the remote host:
    .venv/bin/python -m pytest tests/integration/test_m1_roundtrip.py -v -s
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


def _run(mode: str, out: str, *extra: str) -> None:
    proc = subprocess.run(
        [PY, os.path.join(HERE, "m1_run.py"), mode, os.path.join(HERE, out), *extra],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=900,
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        pytest.fail(f"{mode} run failed (rc={proc.returncode})")


def _load(name: str) -> dict:
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


@pytest.mark.order(1)
def test_m1_cache_reuse_matches_native_prefix_cache():
    _run("baseline", "m1_baseline.json")
    _run("pcache", "m1_pcache.json")
    _run("connector", "m1_connector.json")

    base = _load("m1_baseline.json")
    pcache = _load("m1_pcache.json")
    conn = _load("m1_connector.json")

    # The connector run actually used the store.
    stats = conn["stats"]
    prefix_blocks = 4  # >= 4 whole blocks of the prefix must be reusable
    assert stats["final"]["saved"] >= prefix_blocks, (
        f"expected >= {prefix_blocks} saved pages, got {stats['final']['saved']}"
    )
    assert stats["final"]["hits"] >= prefix_blocks, (
        f"phase B must restore the shared prefix from OmniKV, got "
        f"{stats['final']['hits']} hit blocks"
    )

    # Gate 2: phase A recomputes from scratch in both engine flavors.
    assert (
        conn["outputs"]["a"]["token_ids"] == base["outputs"]["a"]["token_ids"]
    ), "phase A (no restore) diverged from baseline"
    assert (
        pcache["outputs"]["a"]["token_ids"] == base["outputs"]["a"]["token_ids"]
    ), "pcache phase A (first computation) diverged from baseline"

    # Gate 1 (the decisive one): OmniKV's restored-prefix continuation is
    # token-identical to vLLM's native prefix-cache continuation. Both reuse
    # A's block bytes and compute the same suffix; any difference here means
    # OmniKV restored wrong bytes or wrong block placement.
    assert (
        conn["outputs"]["b"]["token_ids"] == pcache["outputs"]["b"]["token_ids"]
    ), (
        "OmniKV-restored continuation differs from vLLM native prefix-cache "
        f"continuation:\nconnector: {conn['outputs']['b']['text']!r}\n"
        f"pcache:    {pcache['outputs']['b']['text']!r}"
    )
    print("\n[M1] OmniKV restore == vLLM native prefix-cache reuse (token-identical).")

    # Gate 3: cache-vs-recompute divergence is allowed only when both cache
    # paths diverge from the baseline at the same position (inherent bf16
    # prefix-reuse drift), never when only one does.
    for key in ("b",):
        c_div = _first_div(conn["outputs"][key], base["outputs"][key])
        p_div = _first_div(pcache["outputs"][key], base["outputs"][key])
        if c_div is None:
            print(f"[M1] output {key!r}: cache-reuse == recompute (token-identical).")
            continue
        assert p_div == c_div, (
            f"output {key!r}: connector diverges from baseline at {c_div} but "
            f"native prefix cache diverges at {p_div} — asymmetric divergence "
            "indicates an OmniKV defect, not reuse drift"
        )
        print(
            f"[M1] output {key!r}: both cache paths diverge from recompute at "
            f"token {c_div} (identical positions; inherent prefix-reuse drift)."
        )


def _first_div(a: dict, b: dict) -> int | None:
    for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])):
        if x != y:
            return i
    return None if len(a["token_ids"]) == len(b["token_ids"]) else min(
        len(a["token_ids"]), len(b["token_ids"])
    )


@pytest.mark.order(2)
def test_eviction_under_pressure_still_correct():
    """Store of 4 pages: the shared prefix barely fits; correctness must hold
    via LRU + reservations (output compared against the pcache control)."""
    _run(
        "connector",
        "m1_tiny.json",
        "--cpu-pages",
        "4",
    )
    conn = _load("m1_tiny.json")
    pcache = _load("m1_pcache.json")
    assert conn["stats"]["final"]["store_len"] <= 4
    assert (
        conn["outputs"]["b"]["token_ids"] == pcache["outputs"]["b"]["token_ids"]
    ), "output 'b' diverged from the native-cache control under eviction pressure"
    print("\n[M1] outputs remain correct under store pressure.")


GPT2_DIR = "/root/autodl-tmp/models/models/AI-ModelScope--gpt2/snapshots/master"


@pytest.mark.order(3)
def test_second_model_mha_geometry():
    """M1 'done when' matrix: a second model with MHA geometry (GPT-2: 12
    layers, 12/12 KV heads, head 64, 192 KiB pages). The core must be
    geometry-agnostic: fingerprint + strides carry everything."""
    if not os.path.isdir(GPT2_DIR):
        pytest.skip(f"gpt2 not downloaded at {GPT2_DIR}")
    _run("baseline", "m1_gpt2_base.json", "--model", GPT2_DIR)
    _run("pcache", "m1_gpt2_pcache.json", "--model", GPT2_DIR)
    _run("connector", "m1_gpt2_conn.json", "--model", GPT2_DIR)
    base, pc, co = (
        _load("m1_gpt2_base.json"),
        _load("m1_gpt2_pcache.json"),
        _load("m1_gpt2_conn.json"),
    )
    assert co["stats"]["final"]["saved"] >= 4
    assert co["stats"]["final"]["hits"] >= 4
    assert (
        co["outputs"]["a"]["token_ids"] == base["outputs"]["a"]["token_ids"]
    ), "gpt2 phase A diverged from baseline"
    assert (
        co["outputs"]["b"]["token_ids"] == pc["outputs"]["b"]["token_ids"]
    ), "gpt2: OmniKV restore differs from native prefix-cache reuse"
    print("\n[M1] MHA model (gpt2): OmniKV restore == native prefix cache.")

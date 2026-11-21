#!/usr/bin/env bash
# M5 remote validation runbook: sync + the full matrix in one command.
#
# Usage:
#   scripts/m5_remote_matrix.sh            # core matrix (regression + M5)
#   scripts/m5_remote_matrix.sh --with-shim-venv   # also build a second
#        vLLM venv (M5_SECOND_VLLM, default 0.33.0) and run the unit
#        suite + M1 roundtrip there (the version-shim matrix leg)
#
# Stages (each logs to logs/m5/<stage>.log on the remote):
#   1. rsync workspace
#   2. unit suite (torch+vllm host)
#   3. M5 GPU fuzz (golden offsets)
#   4. M1-M4 integration regression
#   5. M5 matrix: 128-concurrency + preemption + 100-round soak
#   6. m5_bench (2-D copies, hash path, scratch)
#   7. head-to-head: none / omnikv / pcache (+ lmcache if installed)
#   8. optional: second-vLLM shim venv
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/remote_env.sh

WITH_SHIM=0
for arg in "$@"; do
  [[ "$arg" == "--with-shim-venv" ]] && WITH_SHIM=1
done

echo "== 1/8 sync =="
scripts/sync_remote.sh

run_remote() {
  local stage="$1"; shift
  echo "== $stage =="
  ssh -p "$OMNIKV_PORT" "$OMNIKV_HOST" \
    "cd $OMNIKV_DIR && mkdir -p logs/m5 && $* 2>&1 | tee logs/m5/${stage}.log"
}

echo "== 2/8 remote unit suite =="
run_remote unit ".venv/bin/pytest tests/unit -q"

echo "== 3/8 M5 GPU fuzz =="
run_remote fuzz ".venv/bin/pytest tests/integration/test_m5_fuzz_gpu.py -v"

echo "== 4/8 M1-M4 regression =="
run_remote regression \
  "HF_HUB_OFFLINE=1 .venv/bin/pytest tests/integration/test_m1_roundtrip.py tests/integration/test_m2_cross_layout.py tests/integration/test_m3_async_concurrency.py tests/integration/test_m4_gpu.py -q"

echo "== 5/8 M5 matrix (T6/T9/soak) =="
run_remote matrix "HF_HUB_OFFLINE=1 .venv/bin/pytest tests/integration/test_m5_matrix.py -v -s"

echo "== 6/8 M5 microbenchmark =="
run_remote bench ".venv/bin/python tests/integration/m5_bench.py docs/m5_bench_results.json"

echo "== 7/8 head-to-head =="
run_remote h2h_none   "HF_HUB_OFFLINE=1 .venv/bin/python tests/integration/m5_headtohead.py none   /tmp/m5_h2h_none.json"
run_remote h2h_omnikv "HF_HUB_OFFLINE=1 .venv/bin/python tests/integration/m5_headtohead.py omnikv /tmp/m5_h2h_omnikv.json"
run_remote h2h_pcache "HF_HUB_OFFLINE=1 .venv/bin/python tests/integration/m5_headtohead.py pcache /tmp/m5_h2h_pcache.json"
run_remote h2h_lmcache "HF_HUB_OFFLINE=1 .venv/bin/python tests/integration/m5_headtohead.py lmcache /tmp/m5_h2h_lmcache.json || true"
ssh -p "$OMNIKV_PORT" "$OMNIKV_HOST" "cd $OMNIKV_DIR && cat /tmp/m5_h2h_*.json" || true

if [[ "$WITH_SHIM" == 1 ]]; then
  echo "== 8/8 second-vLLM shim venv =="
  run_remote shim_env ".venv/bin/python - <<'PY'
import subprocess, sys, os
version = os.environ.get('M5_SECOND_VLLM', '0.33.0')
venv = f'/root/autodl-tmp/omnikv-shim-{version}'
if not os.path.exists(venv):
    subprocess.run([os.path.expanduser('~/.local/bin/uv'), 'venv', venv], check=True)
    subprocess.run([os.path.expanduser('~/.local/bin/uv'), 'pip', 'install',
                    '--python', f'{venv}/bin/python', f'vllm=={version}'],
                   check=True)
print(venv)
PY"
  SHIM_PY="/root/autodl-tmp/omnikv-shim-${M5_SECOND_VLLM:-0.33.0}/bin/python"
  run_remote shim_unit "$SHIM_PY -m pytest tests/unit -q"
  run_remote shim_m1 "HF_HUB_OFFLINE=1 $SHIM_PY - <<'PY'
import sys
sys.argv = ['m1', 'connector', '/tmp/m5_shim_m1.json']
exec(open('tests/integration/m1_run.py').read())
PY"
fi

echo "== M5 remote matrix complete; logs in $OMNIKV_DIR/logs/m5/ =="

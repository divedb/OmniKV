# Development and Test Environment

Recorded 2026-11-20. Connection details live in `scripts/remote_env.sh`
(defaults) and `scripts/remote.env` (gitignored overrides) — key-based SSH
only; **no credentials are stored in this repository**.

## Topology

- **Local (macOS, source of truth):** editing, unit tests that do not need
  CUDA (`uv` + `.venv`), `rsync` push via `scripts/sync_remote.sh`.
- **Remote (AutoDL GPU container, Ubuntu 22.04.5, kernel 5.15):** all
  GPU-dependent tests, integration tests, benchmarks, via
  `scripts/remote.sh '<cmd>'`.

## Remote hardware / platform (verified 2026-11-20)

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 3080 Ti — sm_86 (Ampere), 12 288 MiB VRAM |
| Driver | 595.71.05 |
| CUDA toolkit (system) | 12.8 (`/usr/local/cuda-12.8`) |
| CPU / RAM | 80 cores / 125 GiB |
| Data disk `/root/autodl-tmp` | 50 GB |
| Python (system conda) | 3.12.3 (`/root/miniconda3`) |
| Network | China-region host: use Tsinghua PyPI mirror + ModelScope |

> Endpoint note: the instance was replaced on 2026-11-20 (new SSH port,
> see `scripts/remote_env.sh`). The replacement started with an **empty
> data disk** — the environment (venv, vLLM stack, models) was rebuilt
> from this repository's scripts and the recorded install commands in
> `docs/TESTLOG.md`. Connection details live only in `scripts/remote_env.sh`
> (defaults) and the gitignored `scripts/remote.env`; key-based SSH only.

## Existing remote artifacts (do not disturb)

- `/root/autodl-tmp/FlexKV` — audited FlexKV checkout (v1.0.0) with its own
  `.venv` (vllm 0.10.1.1, torch 2.7.1+cu128). Serves as the legacy baseline
  "control" environment for M5 comparisons.
- `/root/autodl-tmp/hf` — HF cache containing Qwen2.5-1.5B-Instruct (reusable
  for smoke tests / M1 correctness runs; use `HF_HOME=/root/autodl-tmp/hf`).
- `/root/autodl-tmp/FlexKV/ssd_cache{,2}` — **deleted 2026-11-20** (owner
  -approved): 32 GB of regenerable FlexKV SSD cache artifacts, freeing the
  data disk to 35 GB available.
- `/root/kvloom-venv`, `/root/miniconda3` — unrelated; untouched.

## Environment layout

- Remote project dir: `/root/autodl-tmp/omnikv` (created and synced by
  `scripts/sync_remote.sh`).
- Remote `uv`: `~/.local/bin/uv` (0.13.9, standalone install).
- Remote venv: `/root/autodl-tmp/omnikv/.venv` (Python 3.12, managed by `uv`),
  GPU dependency set `vllm==0.34.0` installed from the Tsinghua mirror.
- Local venv: `.venv` managed by `uv` (0.10.2) with CPU-only dependency
  groups (`dev`, `local-cpu`).

## Constraints and notes

1. **Disk pressure: resolved 2026-11-20.** 32 GB of regenerable FlexKV SSD
   cache artifacts were deleted (owner-approved); 35 GB now free. Remaining
   budget still requires care: one vLLM GPU env (~10–15 GB) + model weights.
2. **12 GB VRAM:** test models must be small (Qwen2.5-0.5B/1.5B class);
   `gpu_memory_utilization` budgets must respect the FlexKV control env
   not running concurrently.
3. **Python PATH:** non-interactive SSH has no conda on PATH; all remote
   commands must use absolute venv paths (the project `.venv` handles this).
4. **CUDA wheels: validated.** torch 2.14.0+cu130 + vllm 0.34.0 run correctly
   on the sm_86 GPU with driver 595.x (smoke test 2026-11-20, see
   `docs/TESTLOG.md`).
5. **HF offline usage:** `huggingface_hub` 2.4.0 cannot resolve revisions
   offline from the existing cache; pass the snapshot directory path as
   `model=` instead (see `scripts/smoke_vllm.py`).
6. **vLLM API skew:** the 0.34.0 *release* still ships the pre-#61452
   `KVCacheGroupSpec` (singular `kv_cache_spec`), unlike `main` researched in
   `research.md`. The adapter must dual-form-read both (precedent: LMCache
   `group_*` helpers).
7. **KVCacheTensor strides ≠ derived values:** measured padding of ~111
   block slots per layer on this stack (extensible-KV-cache headroom).
   Never compute `layer_stride` from `num_blocks`; always use reported
   strides (see TESTLOG for the measured numbers).

## Workflow (per project rules)

1. Edit locally → `uv run pytest` (CPU tests) → lint (`ruff`, `mypy`).
2. `scripts/sync_remote.sh` to push (venvs/caches/models excluded).
3. `scripts/remote.sh '<test command>'` to validate on GPU.
4. Record commands/results/failures in `docs/TESTLOG.md`.
5. A feature is complete only after remote validation passes.

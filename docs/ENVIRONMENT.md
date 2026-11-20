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
| CPU / RAM | 80 cores / 125 GiB (96 GiB available) |
| Data disk `/root/autodl-tmp` | 50 GB (see constraints below) |
| Python (system conda) | 3.12.3 (`/root/miniconda3`) |
| Network | China-region host: use Tsinghua PyPI mirror + ModelScope |

## Existing remote artifacts (do not disturb)

- `/root/autodl-tmp/FlexKV` — audited FlexKV checkout (v1.0.0) with its own
  `.venv` (vllm 0.10.1.1, torch 2.7.1+cu128). Serves as the legacy baseline
  "control" environment for M5 comparisons.
- `/root/autodl-tmp/hf` — HF cache containing Qwen2.5-1.5B-Instruct (reusable
  for smoke tests / M1 correctness runs).
- `/root/autodl-tmp/FlexKV/ssd_cache{,2}` — 32 GB of generated FlexKV SSD
  cache data from audit runs (regenerable artifacts; candidate for cleanup
  when disk is needed — requires owner approval).
- `/root/kvloom-venv`, `/root/miniconda3` — unrelated; untouched.

## Environment layout (planned)

- Remote project dir: `/root/autodl-tmp/omnikv` (created by
  `scripts/sync_remote.sh`).
- Remote venv: `/root/autodl-tmp/omnikv/.venv` managed by `uv` with GPU
  dependency set (vllm pinned per milestone; see `research.md` §9).
- Local venv: `.venv` managed by `uv` with CPU-only dependency groups
  (`dev`, `local-cpu`).

## Constraints and notes

1. **Disk pressure:** the 50 GB data disk had only ~3.5 GB free at audit
   time (FlexKV artifacts + HF cache). GPU-stack installs (~10–15 GB)
   require freeing space first (see cleanup candidate above).
2. **12 GB VRAM:** test models must be small (Qwen2.5-0.5B/1.5B class);
   `gpu_memory_utilization` budgets must respect the FlexKV control env
   not running concurrently.
3. **Python PATH:** non-interactive SSH has no conda on PATH; all remote
   commands must use absolute venv paths (the project `.venv` handles this).
4. **CUDA wheels:** system toolkit is 12.8 but the driver (595.x) supports
   CUDA 13.x runtimes; current vLLM wheels (CUDA 13.0 line) are expected to
   work — must be validated by an on-GPU smoke test before any dependency on
   it (research.md open question Q6 discipline).

## Workflow (per project rules)

1. Edit locally → `uv run pytest` (CPU tests) → lint (`ruff`, `mypy`).
2. `scripts/sync_remote.sh` to push (venvs/caches/models excluded).
3. `scripts/remote.sh '<test command>'` to validate on GPU.
4. Record commands/results/failures in `docs/TESTLOG.md`.
5. A feature is complete only after remote validation passes.

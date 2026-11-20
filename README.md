# OmniKV

A layout-agnostic KV cache system for vLLM. **Status: research complete,
implementation not started.** See [`research.md`](research.md) for the full
findings and architecture design.

## Repository layout

- `research.md` — research and architecture design (Phase 1 deliverable)
- `docs/ENVIRONMENT.md` — development/test environment and constraints
- `docs/TESTLOG.md` — append-only record of remote validation runs
- `scripts/` — remote sync/execution helpers (`sync_remote.sh`, `remote.sh`)
- `src/`, `tests/` — implementation (empty until M1 begins)

## Development workflow

Local (source of truth) → `scripts/sync_remote.sh` → remote GPU validation
via `scripts/remote.sh`. See `docs/ENVIRONMENT.md` for details and
`scripts/remote.env.example` for connection configuration.

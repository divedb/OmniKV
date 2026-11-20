# OmniKV

A layout-agnostic KV cache system for vLLM. **Status: M1 complete —
single-group full-attention PAGED_KV passthrough validated on GPU.** See
[`research.md`](research.md) for the architecture design and
[`docs/TESTLOG.md`](docs/TESTLOG.md) for validation evidence.

**M1 validated** (vLLM 0.34.0, RTX 3080 Ti): block save/restore via a CPU
page store, with restores **token-identical to vLLM's native prefix-cache
reuse** across GQA (Qwen2.5-1.5B) and MHA (gpt2) models × NHD/HND layouts,
bitwise GPU read-back verification, and LRU eviction with ownership
reservations. Next: M2 (layout abstraction hardening: run-decomposed
regions, codecs) per the roadmap in `research.md` §9.

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

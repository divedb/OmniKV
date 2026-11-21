# OmniKV

A layout-agnostic KV cache system for vLLM. **Status: M5 implemented — full-matrix validation
(fuzz/soak/concurrency/preemption), transfer + hash-path optimization, and a machine-readable
support matrix; local validation green, remote GPU run staged (see
[`docs/M5.md`](docs/M5.md)).** See [`research.md`](research.md) for the architecture design and
[`docs/TESTLOG.md`](docs/TESTLOG.md) for validation evidence.

**M1** (vLLM 0.34.0, RTX 3080 Ti): block save/restore via a CPU page store,
restores **token-identical to vLLM's native prefix-cache reuse** across GQA
(Qwen2.5-1.5B) and MHA (gpt2) models × NHD/HND layouts, bitwise GPU
read-back verification, LRU eviction with ownership reservations.

**M2**: fragment-decomposed placement (per-head runs for LHBNC/BHLNC),
kernel-block-split and dim-0-padded views, element-unit stride
certification for all six layout families, the NHD↔HND codec with
cross-fingerprint store routing, and store persistence (dump/preload).
Flagship result: **KV saved under LBNHC, restored into an LBHNC engine
through the codec — token-identical to that engine's native prefix-cache
reuse.**

**M3**: asynchronous ticket-driven saves (publish-on-completion), engine
generation guards, reservation GC, a **disk spill tier** (hot pinned +
cold file LRU with promotion), failure-tolerant best-effort saves, and
batched codecs. Validated: 8-way concurrent reuse matches the native
control; a 30-round soak under a 4-page hot tier holds zero-leak audits
with spilling active; an injected save failure survives. Transfers run at
77%/68% (d2h/h2d) of the measured pinned-copy ceiling; cross-block
coalescing needs request-scoped layer-major staging (documented gap).
**M4**: per-group HMA planning and completion, trailing-window restoration,
padded/compressed MLA, opaque sparse-index pages, and captured per-tensor
FP8 scales. Gemma-2-2B hybrid and a small DeepSeek-V2 MLA fixture match
native prefix reuse in continuation tokens, hidden states and logprobs.
Recurrent caching remains disabled after a no-go semantic spike. See
[`docs/M4.md`](docs/M4.md) for the support boundaries and reproducible tests.
**M5**: pitched 2-D copy planning (the M3 cross-block coalescing gap),
persistent pinned scratch slots, a packed/byte-identical hash path with
per-request chain memoization, oracle-checked store/epoch fuzzing, a
golden-offset GPU adapter fuzz that **found and closed a silent-corruption
path** (block-outermost layouts with kernel-block splits now fail closed),
128-request concurrency + preemption + 100-round soak harnesses, the
LMCache/FlexKV head-to-head workload, and
[`docs/support_matrix.json`](docs/support_matrix.json) whose accuracy is a
unit-tested invariant. See [`docs/M5.md`](docs/M5.md).

## Repository layout

- `research.md` — research and architecture design (Phase 1 deliverable)
- `docs/ENVIRONMENT.md` — development/test environment and constraints
- `docs/TESTLOG.md` — append-only record of remote validation runs
- `docs/M5.md` — M5 scope, matrix and runbook
- `docs/support_matrix.json` — machine-readable support matrix (unit-tested)
- `scripts/` — remote sync/execution helpers (`sync_remote.sh`, `remote.sh`)
- `src/`, `tests/` — implementation and validation suites

## Development workflow

Local (source of truth) → `scripts/sync_remote.sh` → remote GPU validation
via `scripts/remote.sh`. See `docs/ENVIRONMENT.md` for details and
`scripts/remote.env.example` for connection configuration.

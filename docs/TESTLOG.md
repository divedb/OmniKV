# Test Log

Commands, results, failures and fixes from remote validation runs.
Append-only; newest entries last.

## 2026-11-20 — Environment setup

| Step | Command | Result |
|---|---|---|
| SSH key auth | `ssh -p 44740 root@connect.nmb2.seetacloud.com 'hostname'` | OK (`autodl-container-dfepxcx0ax-8e77fec0`) |
| GPU audit | `nvidia-smi --query-gpu=...` | RTX 3080 Ti, 12 GiB, driver 595.71.05, sm_86 |
| Disk audit | `df -h /root/autodl-tmp` | 50 GB disk, 3.5 GB free (cleanup pending) |
| Python audit | `ls /root/miniconda3/bin/python3` | 3.12.3 (no PATH in non-interactive SSH) |
| Instance replaced | `ssh -p 10487 ...` (new endpoint) | fresh 50 GB disk; env rebuilt: venv, vllm 0.34.0, pytest/order/modelscope, editable omnikv |
| Disk cleanup | `rm -rf FlexKV/ssd_cache FlexKV/ssd_cache2` | 32 GB freed (owner-approved); 35 GB available |
| uv install (remote) | standalone installer → `~/.local/bin/uv` | uv 0.13.9 |
| Project sync | `scripts/sync_remote.sh` | `/root/autodl-tmp/omnikv` created (69 KB sent) |
| GPU stack install | `uv pip install vllm==0.34.0` (Tsinghua mirror) | OK: vllm 0.34.0, torch 2.14.0+cu130, `torch.cuda.is_available()=True` |
| GPU smoke test | `scripts/smoke_vllm.py` (Qwen2.5-1.5B, cached snapshot path) | **PASSED** — details below |

### 2026-11-20 — GPU smoke test results (`scripts/smoke_vllm.py`)

| Check | Result |
|---|---|
| torch kernels on sm_86 (cu130 wheels) | OK — bf16 matmul finite, device RTX 3080 Ti |
| `torch.as_strided` aliasing (BlockView basis) | OK |
| vLLM 0.34.0 end-to-end generation (offline) | OK — `' Paris. The capital of France is also the capital of which country?…'` |
| KVCacheConfig introspection | OK — values below |

**Ground-truth KVCacheConfig** (Qwen2.5-1.5B-Instruct, bf16, 12 GB GPU,
`gpu_memory_utilization=0.85`, `max_model_len=2048`, eager):

```
num_blocks=3488  layout=LBNHC  hash_block_size=64
group 0: layers=28  block_size=64  spec=FullAttentionSpec  prefix_cacheable=True
  num_kv_heads=2  head_size=128  dtype=bfloat16  page_bytes=65536
tensor: layers=28  size=6400507904  layer_stride=235864064  block_stride=65536  offset=0
```

**Design-relevant observation:** `layer_stride` (235,864,064 B = 3,599 ×
65,536) is **not** `num_blocks × page` (3,488 × 65,536 = 228,589,568 B) —
the physical allocation pads ~111 block slots per layer (extensible-KV-cache
fragmentation headroom). A naive `ℓ × num_blocks × page + b × page`
addressing formula would silently read the wrong bytes on this very stack;
the adapter must use the reported `KVCacheTensor` strides verbatim. This
empirically confirms the research.md §4 decision to carry
`layer_stride`/`block_stride` in `PageGeometry` instead of deriving them.

### Issues hit and fixes

1. `LLM(enable_multiprocessing=False)` → `TypeError` in 0.34.0 (kwarg
   removed). Fix: `VLLM_ENABLE_V1_MULTIPROCESSING=0` env var, set in
   `scripts/smoke_vllm.py`.
2. `huggingface_hub` 2.4.0 offline revision resolution fails on the existing
   cache (`LocalEntryNotFoundError`) even though the snapshot is complete.
   Fix: pass the snapshot directory path directly as `model=` (works fully
   offline). Durable fix (ModelScope mirror of test models) deferred until
   models are next needed.
3. **API skew confirmed:** installed vLLM 0.34.0 release uses
   `KVCacheGroupSpec(layer_names, kv_cache_spec)` (pre-#61452), while
   researched `main` uses `KVCacheGroupSpec(kv_cache_specs, manager_config)`.
   The smoke script's introspection now reads both forms (dual-form reader,
   same approach as LMCache). OmniKV's vLLM adapter must ship this shim
   (research.md §2.1.7, open question Q6).

## 2026-11-20 — M1 integration (tests/integration/test_m1_roundtrip.py)

Stack: vLLM 0.34.0, torch 2.14.0+cu130, Qwen2.5-1.5B-Instruct (cached),
LBNHC, block 64, eager, `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

**Result: 2/2 PASSED** (`test_m1_cache_reuse_matches_native_prefix_cache`,
`test_eviction_under_pressure_still_correct`).

Evidence recorded:

| Gate | Result |
|---|---|
| Store usage (64-page pool) | saved=4, hits=4, store_len=4 |
| Bitwise L1 read-back (verify_load_bytes) | all restored GPU pages byte-equal to store pages |
| Connector-A == baseline-A | token-identical (recompute path unaffected) |
| **Connector-B == native-prefix-cache-B** | **token-identical (decisive gate)** |
| Cache-vs-recompute divergence | both cache paths diverge at the same token (19) — inherent bf16 prefix-reuse drift, not corruption |
| Eviction pressure (4-page store) | outputs token-identical to the native-cache control; free_pages=0 |

Methodology note: exact token equality between cache-reuse and full
recompute is NOT a valid gate on this stack (reused KV bytes are produced
under a different prefill chunk shape, so bf16 near-ties can flip). The
valid gate is equivalence to vLLM's own prefix-cache reuse semantics,
which OmniKV matches exactly. The near-tie logprob analysis that led here
is preserved in git history of `test_m1_roundtrip.py`.

### Issues hit and fixes (M1)

1. Worker-side `kv_cache_config` under the extensible KV cache uses **one
   `KVCacheTensor` per layer** (28×1) vs the scheduler-side single tensor
   (1×28). Adapter now matches tensors by layer membership; per-layer
   addressing via registered view pointers + `block_stride`.
2. `libcudart.so.13` lives at `nvidia/cu13/lib/` in the cu13 wheel layout;
   loader now globs `nvidia/**/libcudart.so*` recursively.
3. Save/load ordering: a save accepted by `request_finished` rides the next
   step's metadata, so a lookup could miss pages not yet landed. Fix:
   `get_num_new_matched_tokens` returns `None` (re-query) while saves are
   pending — the base-class async-lookup contract.
4. Unit test caught a real design bug: `layer_block_offset` derived the
   block offset from `page_bytes` instead of `block_stride` (identical for
   layer-compact layouts, wrong for BLNHC). `PageGeometry.block_stride` is
   now authoritative.
5. Test-harness bugs (class-name mutation contaminating tests, `Logprob`
   objects, inverted prefix-cache flag) — all in tests, fixed.

### M1 validation matrix (all on remote RTX 3080 Ti)

| Configuration | Result |
|---|---|
| Qwen2.5-1.5B (GQA, 28L, 2 KV heads, 64 KiB pages), LBNHC | ✅ restore == native pcache (token-identical); saved=4, hits=4 |
| Qwen2.5-1.5B, **LBHNC (HND)** via `VLLM_KV_CACHE_LAYOUT` | ✅ token-identical; saved=4, hits=4 |
| gpt2 (**MHA**, 12L, 12:12 heads, 192 KiB pages), LBNHC | ✅ token-identical; saved=4, hits=4 |
| Eviction pressure (4-page store, LRU + reservations) | ✅ token-identical to control |
| Bitwise L1 read-back (`omnikv.verify_load_bytes`) | ✅ every restored page byte-equal on GPU |

### M1 benchmark (`tests/integration/m1_bench.py`, 2026-11-20)

| Geometry | Payload | d2h | h2d |
|---|---|---|---|
| 28 × 64 KiB (Qwen) × 256 blocks | 448 MiB | 9.44 GiB/s | 7.77 GiB/s |
| 12 × 192 KiB (gpt2) × 256 blocks | 576 MiB | 9.09 GiB/s | 9.44 GiB/s |

~50% of practical PCIe 4.0 bandwidth; the per-(layer,block) copy-submission
loop is the known bottleneck — contiguous-page coalescing is the first M3
batching target (research.md §8.2).

### M1 residual items (deferred, with rationale)

- Preemption stress test: aborted requests are already skipped defensively
  in `request_finished`; a dedicated stress test lands with the async-save
  machinery in M3 (where preemption actually races in-flight transfers).
- Per-step connector overhead at ≥128 concurrent requests: needs the
  concurrency harness; deferred to M3 alongside async loads.

## 2026-11-20 — M2: layout abstraction hardening (instance 2, port 10487)

Scope implemented (v2 core):

- `types.py` v2: `FormatFingerprint.page_fragments` (store-page interior
  structure; identity includes it), `LayerAddress`/`GroupPlacement`
  (fragment-decomposed GPU addressing; per-layer storages; component
  labels for separate K/V), `FORMAT_VERSION = 2`.
- `adapter/vllm.py`: certification of the registered views' **element**
  strides against the layout permutation at kernel-block granularity
  (ratio derived from metadata); placement for all six layout families
  incl. per-head-fragmented LHBNC/BHLNC; worker per-layer-tensor form;
  dim-0 padding; mixed per-layer specs rejected (value-based identity).
- `codec.py`: `PageCodec`/`CodecRegistry` with the NHD<->HND store-page
  row permutation (torch views, host-side); `swap_layout` for alternate
  fingerprint keys; absent codec = hard `CompatibilityError`.
- `connector/v1.py`: per-fingerprint stores; re-registrable placement
  (sequential engines share stores in-process); `cross_layout` lookup
  routing through the codec registry; per-block scratch for converted
  pages; store persistence (`dump_runtime_stores`/`load_runtime_stores`,
  `omnikv.preload_store`).
- `detect.py`: detection-tier stride certification with ambiguity
  rejection (N==H both-readings case raises without a hint).

### M2 validation results

| Gate | Result |
|---|---|
| Unit conformance (local torch-free + remote torch-backed) | 56 local / 67 remote passed |
| Codec exactness | repack == reference permutation; round-trip identity (3 seeds) |
| Detection ambiguity | N==H dual-reading raises; hint resolves; dim-0 pad rejected |
| **Cross-layout E2E** (LBNHC save → LBHNC engine, codec load) | **codec_hits=4; cross-B == native-LBHNC-pcache-B token-identical; both diverge from recompute at token 19 (established reuse drift)** |
| Bitwise L1 read-back | enabled in all connector runs; passed |

### M2 benchmark (`tests/integration/m2_bench.py`)

28-page slot (1.8 MiB): repack 709 µs (2.41 GiB/s) vs memcpy 19 µs
(92 GiB/s) — **codec overhead ×38**. For reference, H2D of the same slot
is ~230 µs, so a converted block costs ~3× its transfer time. M3+
optimization targets: fused repack kernels or direct fragment-mapped H2D
(eliminating the scratch copy).

### M2 issues hit and fixes

1. Engine views are dtype-typed: torch strides are ELEMENTS —
   certification moved to element units (byte addressing via stride×elem).
2. Kernel-block splits: registered views carry (num_blocks×ratio, H,
   block_size/ratio, C); ratio derived from metadata, manager-block
   addressing = ratio × view stride.
3. Deferred-free saves need a subsequent engine step: the save-mode
   runner now forces a flush request before dumping stores.
4. Codec unit is one layer page, not a store slot: `_converted_page`
   repacks per layer.
5. Buffer-lifetime bug: one scratch reused across blocks while ops were
   batched for a single submit — every block read the last repack. Fixed
   with a combined per-block-offset scratch (caught by the output gate
   reproducing degenerate repetition).

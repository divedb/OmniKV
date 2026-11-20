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

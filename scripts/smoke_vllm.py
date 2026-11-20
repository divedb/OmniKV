"""GPU stack smoke test for the remote validation environment.

Run on the remote GPU server:
    scripts/remote.sh 'HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 \
        .venv/bin/python scripts/smoke_vllm.py'

Validates:
1. torch cu130 kernels actually execute on the sm_86 GPU (import alone is
   not sufficient evidence).
2. `torch.as_strided` view semantics used by the BlockView design.
3. vLLM 0.34.0 end-to-end generation with the locally cached
   Qwen2.5-1.5B-Instruct (no network).
4. Best-effort introspection of the engine's resolved KVCacheConfig
   (groups / layout) to validate research.md assumptions.
"""

from __future__ import annotations

import os
import traceback

# Run the engine core in-process so the KVCacheConfig can be introspected.
# vLLM 0.34.0 removed the LLM(enable_multiprocessing=...) kwarg in favor of
# this environment variable.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def check_torch() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    free, total = torch.cuda.mem_get_info()
    print(f"[torch] device={name} capability=sm_{cap[0]}{cap[1]} "
          f"cuda={torch.version.cuda} vram={total >> 20}MiB free={free >> 20}MiB")

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = a @ a
    assert torch.isfinite(b).all(), "bf16 matmul produced non-finite values"
    print("[torch] bf16 matmul kernel OK")

    # Strided-view addressing exercised by the future BlockView code: an
    # inner-dim-contiguous page view over a flat buffer must alias storage.
    buf = torch.zeros(256, dtype=torch.uint8, device="cuda")
    view = torch.as_strided(buf, (4, 8, 8), (64, 8, 1))
    view[1, 2, 3] = 7
    assert buf[64 + 2 * 8 + 3] == 7, "as_strided view does not alias storage"
    (buf.zero_(), view[3, 0, 0])  # stream-order sanity, no host sync needed
    torch.cuda.synchronize()
    print("[torch] as_strided view semantics OK")


def check_vllm_generate() -> None:
    from vllm import LLM, SamplingParams

    # NOTE: huggingface_hub 2.x offline revision resolution fails on this
    # cache layout, but the snapshot path itself loads fine offline.
    model = os.environ.get(
        "OMNIKV_MODEL",
        "/root/autodl-tmp/hf/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,  # keep the smoke test fast; graphs validated in M1
    )
    outs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    text = outs[0].outputs[0].text
    print(f"[vllm] generated: {text!r}")
    assert text.strip(), "empty generation"

    _introspect_kv_cache_config(llm)


def _introspect_kv_cache_config(llm) -> None:
    """Best-effort dump of the resolved KVCacheConfig (never fails the test)."""
    try:
        # vLLM 0.34.0 in-process path:
        # LLM -> LLMEngine.engine_core (InprocClient) -> EngineCore.scheduler
        engine = getattr(llm, "llm_engine", None)
        client = getattr(engine, "engine_core", None)
        core = getattr(client, "engine_core", client)
        scheduler = getattr(core, "scheduler", None)
        cfg = getattr(scheduler, "kv_cache_config", None)
        if cfg is None:
            print("[kvconf] kv_cache_config not reachable on this path")
            return
        print(f"[kvconf] num_blocks={cfg.num_blocks} "
              f"layout={cfg.kv_cache_layout} hash_bs={cfg.hash_block_size}")
        for i, g in enumerate(cfg.kv_cache_groups):
            # Dual-form: vLLM release (<=0.34) uses KVCacheGroupSpec
            # (singular kv_cache_spec + layer_names); main (post-#61452) uses
            # kv_cache_specs dict + manager_config.
            names = getattr(g, "layer_names", None) or list(
                getattr(g, "kv_cache_specs", {})
            )
            specs = list(getattr(g, "kv_cache_specs", {None: g.kv_cache_spec}).values())
            kinds = sorted({type(s).__name__ for s in specs})
            s0 = specs[0]
            print(f"[kvconf] group {i}: layers={len(names)} "
                  f"block_size={s0.block_size} specs={kinds} "
                  f"prefix_cacheable={s0.prefix_cacheable}")
            if hasattr(s0, "num_kv_heads"):
                print(f"[kvconf]   geometry: num_kv_heads={s0.num_kv_heads} "
                      f"head_size={s0.head_size} dtype={s0.dtype} "
                      f"page_bytes={s0.page_size_bytes}")
        for t in cfg.kv_cache_tensors:
            print(f"[kvconf] tensor: layers={len(t.layers)} "
                  f"size={t.size} layer_stride={t.layer_stride} "
                  f"block_stride={t.block_stride} offset={t.offset}")
    except Exception:
        print("[kvconf] introspection failed (non-fatal):")
        traceback.print_exc()


if __name__ == "__main__":
    check_torch()
    check_vllm_generate()
    print("SMOKE TEST PASSED")

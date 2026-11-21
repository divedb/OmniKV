"""M4 semantic acceptance runner, one fresh vLLM engine per invocation.

Gemma-2-2B exercises real full/SW hybrid weights. Tiny DeepSeek-V2 uses
seeded dummy weights to exercise MLA kernels within the 12 GB test budget.
Compare recompute, native prefix reuse, and OmniKV with poison/readback gates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

GEMMA = "/root/autodl-tmp/models/models/LLM-Research--gemma-2-2b-it/snapshots/master"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["hybrid", "mla", "fp8"])
    parser.add_argument("mode", choices=["baseline", "connector", "pcache"])
    parser.add_argument("out")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kwargs = {
        "model": GEMMA,
        "max_model_len": 4608,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.8,
        "seed": 42,
        "disable_log_stats": False,
        "enable_prefix_caching": args.mode == "pcache",
    }
    if args.kind == "mla":
        from transformers import DeepseekV2Config

        config = DeepseekV2Config(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=512,
            q_lora_rank=128,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=2,
            max_position_embeddings=1024,
            architectures=["DeepseekV2ForCausalLM"],
            torch_dtype="bfloat16",
        )
        model_dir = Path("/tmp/omnikv-m4-tiny-mla")
        model_dir.mkdir(exist_ok=True)
        config.save_pretrained(model_dir)
        kwargs.update(
            model=str(model_dir),
            load_format="dummy",
            skip_tokenizer_init=True,
            max_model_len=1024,
            dtype="bfloat16",
        )
    elif args.kind == "fp8":
        kwargs.update(kv_cache_dtype="fp8")
    if args.mode == "connector":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OmniKVConnectorV1",
            kv_connector_module_path="omnikv.connector.v1",
            kv_role="kv_both",
            kv_connector_extra_config={
                "omnikv.cpu_pages": 128,
                "omnikv.verify_load_bytes": not args.no_verify,
            },
        )
    llm = LLM(**kwargs)
    hidden = []

    def attach(model):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            hidden.append(value[-1].detach().float().cpu().tolist())

        model.model.norm.register_forward_hook(hook)

    llm.apply_model(attach)
    # Token prompts avoid tokenizer differences and keep shared boundaries exact.
    # Gemma exceeds its real 4096-token window; MLA uses six full blocks.
    prefix = [20 + (i * 17) % 300 for i in range(4352 if args.kind != "mla" else 384)]
    prompts = [prefix + [55, 56, 57], prefix + [71, 72, 73]]
    sp = SamplingParams(temperature=0, max_tokens=8, logprobs=5, ignore_eos=True)
    result = {
        "kind": args.kind,
        "mode": args.mode,
        "outputs": [],
        "hidden": [],
        "seconds": [],
        "ttft_seconds": [],
    }
    for prompt in prompts:
        hidden.clear()
        start = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": prompt}], sp, use_tqdm=False)[0]
        result["seconds"].append(time.perf_counter() - start)
        metrics = out.metrics
        result["ttft_seconds"].append(metrics.first_token_latency if metrics else None)
        result["outputs"].append(
            {
                "tokens": list(out.outputs[0].token_ids),
                "logprobs": [
                    {str(k): v.logprob for k, v in pos.items()} for pos in out.outputs[0].logprobs
                ],
            }
        )
        result["hidden"].append(list(hidden))
    if args.mode == "connector":
        from omnikv.connector.v1 import _RUNTIME

        result["stats"] = {
            "hits": _RUNTIME.stats_hits,
            "saved": _RUNTIME.stats_saved,
            "groups": {str(g): d.semantics.kind.value for g, d in _RUNTIME.descriptors.items()},
        }
        assert _RUNTIME.stats_hits > 0, "test must exercise external reuse"
        assert all(s.reserved_pages == 0 for s in _RUNTIME.stores.values())
    Path(args.out).write_text(json.dumps(result))
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()

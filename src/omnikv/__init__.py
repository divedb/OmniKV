"""OmniKV: a layout-agnostic KV cache system for vLLM.

M1 scope (see research.md §9): single-group full-attention PAGED_KV
passthrough between vLLM GPU blocks and a CPU store, same layout on both
ends. The core value types in `omnikv.types` are layout-parametric from
day one so M2 can extend coverage without core surgery.
"""

__version__ = "0.1.0"

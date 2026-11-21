"""Position-based restore plans, independent of layouts and physical block IDs."""

from omnikv.types import StateSemantics


def required_blocks(semantics: StateSemantics, boundary: int, computed: int = 0) -> range:
    """Whole pages needed to resume at boundary, matching SlidingWindowManager.

    A window hit is allowed even when older prefix pages have been evicted.
    Hashes still cover the complete prefix, including those evicted tokens.
    """
    bs = semantics.block_size
    if boundary % bs:
        raise ValueError("restore boundary must align to the group block size")
    start = computed // bs
    if semantics.window_tokens is not None:
        retained = semantics.window_tokens - 1 + semantics.extra_retained_tokens
        start = max(start, max(0, boundary - retained) // bs)
    return range(start, boundary // bs)

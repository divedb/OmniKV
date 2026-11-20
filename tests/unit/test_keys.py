"""Unit tests for chained block hashing (omnikv.keys)."""

from __future__ import annotations

from omnikv.keys import BlockKey, chain_block_hashes


class TestChainBlockHashes:
    def test_chain_length_full_blocks_only(self):
        tokens = list(range(130))
        hashes = chain_block_hashes(tokens, tokens_per_block=64)
        assert len(hashes) == 2  # floor(130 / 64)

    def test_chaining_prefix_dependence(self):
        # Same block content under a different prefix -> different hashes:
        # the chain includes the parent block's hash.
        a = chain_block_hashes([0] * 64 + [5] * 64, 64)
        b = chain_block_hashes([1] * 64 + [5] * 64, 64)
        assert a[0] != b[0]
        assert a[1] != b[1]

    def test_same_content_same_parent_same_hash(self):
        # Identical (parent, block content) -> identical hash: block-level
        # dedup by content chain, matching vLLM prefix-cache semantics.
        a = chain_block_hashes(list(range(128)), 64)
        b = chain_block_hashes(list(range(256)), 64)
        assert a[0] == b[0]
        assert a[1] == b[1]

    def test_identical_inputs_identical_hashes(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        assert chain_block_hashes(tokens, 4) == chain_block_hashes(tokens, 4)

    def test_empty(self):
        assert chain_block_hashes([], 64) == []
        assert chain_block_hashes([1, 2, 3], 64) == []


class TestBlockKey:
    def test_namespacing_by_fingerprint(self):
        h = chain_block_hashes([1, 2, 3, 4], 4)[0]
        assert BlockKey("aaaa", h) != BlockKey("bbbb", h)

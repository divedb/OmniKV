"""Unit tests for chained block hashing (omnikv.keys)."""

from __future__ import annotations

import hashlib

import pytest

from omnikv.keys import BlockKey, chain_block_hashes, extend_block_hashes

# The M1-M4 per-token encoding, kept as the reference for the M5 packed
# implementation: hashes must be byte-identical or persisted store keys
# would silently miss.
_HASH_PERSON = b"omnikv-block-hash-v1"


def reference_chain(token_ids, tokens_per_block, salt=b""):
    num_blocks = len(token_ids) // tokens_per_block
    hashes = []
    prev = b""
    for i in range(num_blocks):
        h = hashlib.blake2b(_HASH_PERSON + salt, digest_size=16)
        h.update(prev)
        for t in token_ids[i * tokens_per_block : (i + 1) * tokens_per_block]:
            h.update(int(t).to_bytes(8, "little"))
        prev = h.digest()
        hashes.append(prev)
    return hashes


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

    def test_packed_encoding_matches_per_token_reference(self):
        # M5 micro-opt: one struct.pack of the whole token stream must
        # reproduce the M1-M4 per-token byte stream exactly.
        for tokens, bs in [
            (list(range(300)), 64),
            ([151643] * 100 + list(range(77)), 16),
            (list(range(2**31, 2**31 + 40)), 8),
        ]:
            assert chain_block_hashes(tokens, bs) == reference_chain(tokens, bs)

    def test_salt_changes_chain(self):
        tokens = list(range(128))
        assert chain_block_hashes(tokens, 64, salt=b"x") != chain_block_hashes(tokens, 64)


class TestExtendBlockHashes:
    def test_extension_equals_full_chain(self):
        tokens = list(range(1000))
        full = chain_block_hashes(tokens, 64)
        for done in (0, 1, 5, len(full)):
            head = full[:done]
            extended = head + extend_block_hashes(
                head[-1] if head else b"", done, tokens, 64
            )
            assert extended == full

    def test_appended_tokens_extend(self):
        # The scheduler-side memoization contract: a chain over a prefix
        # extends to the appended token list without recomputation.
        head_tokens = list(range(256))
        head = chain_block_hashes(head_tokens, 64)
        grown = head_tokens + list(range(1000, 1512))
        full = chain_block_hashes(grown, 64)
        extended = head + extend_block_hashes(head[-1], len(head), grown, 64)
        assert extended == full

    def test_extend_past_tokens_rejected(self):
        with pytest.raises(ValueError, match="past its tokens"):
            extend_block_hashes(b"", 5, [1] * 64, 64)


class TestBlockKey:
    def test_namespacing_by_fingerprint(self):
        h = chain_block_hashes([1, 2, 3, 4], 4)[0]
        assert BlockKey("aaaa", h) != BlockKey("bbbb", h)

    def test_equal_keys_hash_equal(self):
        h = chain_block_hashes([1, 2, 3, 4], 4)[0]
        a, b = BlockKey("aaaa", h), BlockKey("aaaa", h)
        assert a == b and hash(a) == hash(b) and {a: 1}[b] == 1

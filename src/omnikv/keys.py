"""Block keys: chained per-block hashes over prompt tokens (research.md §7.1).

M1 keys are chained (each block's hash includes the previous block's hash),
mirroring vLLM's prefix-cache hashing: block ``i``'s key therefore identifies
the entire token prefix ending at block ``i``. The store namespaces keys by
the producing `FormatFingerprint.identity`, so pages from different layouts,
dtypes or quant modes never collide.

The engine's own `Request.block_hashes` are preferred when available (they
include vLLM's extra-key handling); this module provides the fallback path
used when the engine has not hashed the request.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

_HASH_PERSON = b"omnikv-block-hash-v1"


def chain_block_hashes(
    token_ids: Sequence[int],
    tokens_per_block: int,
    salt: bytes = b"",
) -> list[bytes]:
    """Chained per-block hashes over ``token_ids`` (full blocks only).

    Block ``i``'s hash is ``blake2b(prev_hash || tokens[i])``; the previous
    hash seeds the chain so equal block content at different prefix depths
    produces different keys, exactly like vLLM's `hash_block_tokens`.
    """
    num_blocks = len(token_ids) // tokens_per_block
    hashes: list[bytes] = []
    prev: bytes = b""
    for i in range(num_blocks):
        start = i * tokens_per_block
        chunk = token_ids[start : start + tokens_per_block]
        h = hashlib.blake2b(_HASH_PERSON + salt, digest_size=16)
        h.update(prev)
        for t in chunk:
            h.update(int(t).to_bytes(8, "little"))
        prev = h.digest()
        hashes.append(prev)
    return hashes


@dataclass(frozen=True)
class BlockKey:
    """Store key of one saved page: fingerprint identity + block hash."""

    fingerprint_id: str
    block_hash: bytes

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"BlockKey({self.fingerprint_id[:8]}, {self.block_hash.hex()[:16]})"

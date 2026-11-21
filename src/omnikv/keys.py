"""Block keys: chained per-block hashes over prompt tokens (research.md §7.1).

M1 keys are chained (each block's hash includes the previous block's hash),
mirroring vLLM's prefix-cache hashing: block ``i``'s key therefore identifies
the entire token prefix ending at block ``i``. The store namespaces keys by
the producing `FormatFingerprint.identity`, so pages from different layouts,
dtypes or quant modes never collide.

The engine's own `Request.block_hashes` are preferred when available (they
include vLLM's extra-key handling); this module provides the fallback path
used when the engine has not hashed the request.

M5: the token stream is packed once per call (one C-level ``struct.pack``
instead of a Python ``int.to_bytes``/``update`` loop per token) and chains
can be *extended* from a cached digest — the scheduler re-queries requests
whose lookup is deferred while saves are in flight, and ``all_token_ids``
is append-only, so a cached chain is always a valid prefix of the next one.
The packed byte stream is identical to the M1-M4 per-token encoding, so
block hashes (and therefore store keys) are unchanged.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass

_HASH_PERSON = b"omnikv-block-hash-v1"


def _pack(tokens: Sequence[int]) -> bytes:
    # '<q' = little-endian signed 64-bit: byte-identical to
    # int(t).to_bytes(8, "little") for every non-negative token id.
    return struct.pack(f"<{len(tokens)}q", *tokens)


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
    if num_blocks == 0:
        return []
    packed = memoryview(_pack(token_ids))
    width = tokens_per_block * 8
    person = _HASH_PERSON + salt
    hashes: list[bytes] = []
    prev = b""
    for i in range(num_blocks):
        h = hashlib.blake2b(person, digest_size=16)
        h.update(prev)
        h.update(packed[i * width : (i + 1) * width])
        prev = h.digest()
        hashes.append(prev)
    return hashes


def extend_block_hashes(
    prev_digest: bytes,
    blocks_done: int,
    token_ids: Sequence[int],
    tokens_per_block: int,
    salt: bytes = b"",
) -> list[bytes]:
    """Continue a chain started by `chain_block_hashes` for blocks
    ``[blocks_done, len(token_ids) // tokens_per_block)``.

    Raises ValueError if ``blocks_done`` exceeds the number of complete
    blocks in ``token_ids`` (a chain may only be extended, never rewound).
    """
    total = len(token_ids) // tokens_per_block
    if blocks_done > total:
        raise ValueError(
            f"cannot extend a chain past its tokens: {blocks_done} blocks "
            f"hashed but only {total} complete blocks exist"
        )
    if blocks_done == total:
        return []
    tail = token_ids[blocks_done * tokens_per_block :]
    packed = memoryview(_pack(tail))
    width = tokens_per_block * 8
    person = _HASH_PERSON + salt
    hashes: list[bytes] = []
    prev = prev_digest
    for i in range(total - blocks_done):
        h = hashlib.blake2b(person, digest_size=16)
        h.update(prev)
        h.update(packed[i * width : (i + 1) * width])
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

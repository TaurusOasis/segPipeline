"""Hashing helpers (Step 02)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the hex sha256 of a file, streaming in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the hex sha256 of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()
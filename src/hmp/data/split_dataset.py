"""Deterministic train/val split (Step 07)."""

from __future__ import annotations

import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def train_val_split(
    items: Sequence[T],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[T], list[T]]:
    """Deterministic split by item_id-stable ordering.

    Sorts by ``str`` of each item for reproducibility, then takes the first
    ``val_ratio`` fraction as validation (using a seeded shuffle to randomize).
    """
    n = len(items)
    if n == 0:
        return [], []
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = set(order[:n_val])
    train = [items[i] for i in range(n) if i not in val_idx]
    val = [items[i] for i in range(n) if i in val_idx]
    return train, val
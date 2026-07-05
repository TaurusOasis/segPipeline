"""Binary mask I/O and geometry helpers (Step 04).

Masks are stored as single-channel uint8 PNG with 0/255 values. No heavy
dependencies: only numpy + Pillow/cv2 (already in base.txt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def read_binary_mask(path: str | Path) -> np.ndarray:
    """Read a mask file as a bool ndarray (True = foreground)."""
    import cv2  # local lazy import

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def write_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    """Write a bool/0-1 ndarray as a uint8 0/255 PNG."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), arr)


def write_uint8_image(path: str | Path, arr: np.ndarray) -> None:
    """Write a single-channel uint8 image preserving exact values (e.g. trimaps 0/128/255)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr).astype(np.uint8)
    if a.ndim == 3:
        a = a[..., 0]
    cv2.imwrite(str(path), a)


def mask_to_bbox_xyxy(mask: np.ndarray) -> Optional[list[int]]:
    """Return [x1, y1, x2, y2] bounding box of foreground, or None if empty."""
    mask = np.asarray(mask) > 0
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_area_ratio(mask: np.ndarray) -> float:
    """Fraction of the image occupied by foreground."""
    mask = np.asarray(mask) > 0
    return float(mask.sum()) / float(mask.size)


def remove_small_components(mask: np.ndarray, min_area: int = 64) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` pixels."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    # label 0 is background
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 1
    return out.astype(bool)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m.astype(bool)
    # pick largest non-background label
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask via contour filling."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    # flood-fill from corners to mark background, invert
    h, w = m.shape
    inv = (m == 0).astype(np.uint8)
    # use floodFill to mark true background reachable from border
    flooded = inv.copy()
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, mask_flood, (0, 0), 0)
    # holes = inv - flooded (interior background not connected to border)
    holes = (inv > 0) & (flooded > 0)
    out = (m > 0) | holes
    return out


def combine_instance_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    """Union of multiple boolean masks."""
    masks = list(masks)
    if not masks:
        raise ValueError("combine_instance_masks requires at least one mask")
    out = np.zeros_like(np.asarray(masks[0]), dtype=bool)
    for m in masks:
        out |= np.asarray(m) > 0
    return out
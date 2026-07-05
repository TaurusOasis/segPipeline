"""Boundary extraction for binary masks (Step 05).

Boundary-aware metrics are central to this project: human boundary quality and
temporal stability matter more than COCO-style mask AP. These helpers are
pure-numpy/cv2 — no torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _to_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def mask_to_boundary(
    mask: np.ndarray,
    *,
    pixel_width: Optional[int] = None,
    dilation_ratio: float = 0.02,
) -> np.ndarray:
    """Extract the boundary band of a binary mask.

    If ``pixel_width`` is given, the band is that many pixels wide (centered on
    the contour). Otherwise the width is derived from ``dilation_ratio`` of the
    mask's diagonal length (the Boundary IoU paper formulation).
    """
    import cv2

    m = _to_bool(mask).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)

    h, w = m.shape
    if pixel_width is not None:
        k = max(1, int(pixel_width))
    else:
        diag = float(np.sqrt(h * h + w * w))
        k = max(1, int(round(dilation_ratio * diag)))

    # Boundary-IoU (DIS) formulation: 3x3 kernel with `k` iterations, padded so
    # the image border is treated as boundary. A 1x1 kernel would be identity
    # and produce no band, hence the 3x3 kernel.
    kernel = np.ones((3, 3), np.uint8)
    pad = k
    padded = np.pad(m, pad, mode="constant")
    dilated = cv2.dilate(padded, kernel, iterations=k)
    eroded = cv2.erode(padded, kernel, iterations=k)
    boundary = (dilated - eroded) > 0
    # crop back to original size
    return boundary[pad : pad + h, pad : pad + w]


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    """A band of given pixel ``width`` around the foreground contour."""
    return mask_to_boundary(mask, pixel_width=width)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def boundary_iou(pred: np.ndarray, gt: np.ndarray, **kw) -> float:
    """IoU computed on the boundary bands of pred and gt.

    Returns 0.0 when both masks are empty (defined behaviour); 0.0 when only
    one is empty.
    """
    pb = mask_to_boundary(pred, **kw)
    gb = mask_to_boundary(gt, **kw)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    if union == 0:
        # both empty -> perfect by convention
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def boundary_precision_recall_fscore(
    pred: np.ndarray,
    gt: np.ndarray,
    **kw,
) -> tuple[float, float, float]:
    """Precision / recall / F-score on the boundary bands.

    pred defines the predicted boundary; gt is the reference boundary.
    """
    pb = mask_to_boundary(pred, **kw)
    gb = mask_to_boundary(gt, **kw)
    tp = np.logical_and(pb, gb).sum()
    precision = _safe_div(float(tp), float(pb.sum()))
    recall = _safe_div(float(tp), float(gb.sum()))
    if precision + recall == 0:
        f = 0.0
    else:
        f = 2 * precision * recall / (precision + recall)
    return precision, recall, float(f)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard mask IoU with safe empty handling (both empty -> 1.0)."""
    p = _to_bool(pred)
    g = _to_bool(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)
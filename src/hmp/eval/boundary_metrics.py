"""Boundary-aware evaluation metrics (Step 05).

Public re-exports of the boundary helpers from :mod:`hmp.refine.boundary` so
the ``eval`` namespace owns the metric API while the ``refine`` namespace owns
the boundary extraction used during refinement.
"""

from __future__ import annotations

import numpy as np

from ..refine.boundary import (
    boundary_band,
    boundary_iou,
    boundary_precision_recall_fscore,
    mask_iou,
    mask_to_boundary,
)

__all__ = [
    "boundary_band",
    "boundary_iou",
    "boundary_precision_recall_fscore",
    "mask_iou",
    "mask_to_boundary",
    "boundary_f_score",
]


def boundary_f_score(pred: np.ndarray, gt: np.ndarray, **kw) -> float:
    """Convenience: return only the F-score from boundary precision/recall."""
    _, _, f = boundary_precision_recall_fscore(pred, gt, **kw)
    return f


def aggregate_boundary_metrics(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    **kw,
) -> dict[str, float]:
    """Mean boundary IoU / precision / recall / F over a list of pairs."""
    ious, precs, recs, fs = [], [], [], []
    for p, g in zip(preds, gts):
        ious.append(boundary_iou(p, g, **kw))
        pr, rc, f = boundary_precision_recall_fscore(p, g, **kw)
        precs.append(pr)
        recs.append(rc)
        fs.append(f)
    n = max(len(ious), 1)
    return {
        "boundary_iou": float(np.mean(ious)) if ious else 0.0,
        "boundary_precision": float(np.mean(precs)) if precs else 0.0,
        "boundary_recall": float(np.mean(recs)) if recs else 0.0,
        "boundary_f_score": float(np.mean(fs)) if fs else 0.0,
        "count": float(len(ious)),
    }
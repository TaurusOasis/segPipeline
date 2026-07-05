"""Temporal consistency metrics for video matting / masklets (pipeline step 8).

These complement :mod:`hmp.eval.mqe`, which only computes a pairwise raw
frame-to-frame alpha delta inside ``rule_based_qa``. This module generalizes
that to a full sequence and adds motion-compensated (optical-flow-warped)
error plus masklet identity consistency, so temporal supervision and the
stage-8 temporal QA can be measured, not just gated.

CPU smoke path: when no optical-flow function is supplied, the warped error
degrades to the raw frame-diff (zero flow), matching the roadmap's
"RAFT/GMFlow with frame-diff fallback for CPU smoke tests". Supply a
``flow_fn(prev_alpha, cur_alpha) -> (H, W, 2)`` (dx, dy in pixels) to enable
motion compensation.

All alpha arrays are float in ``[0, 1]``; masks are boolean / binary.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "temporal_flicker",
    "masklet_temporal_iou",
    "warp_with_flow",
    "temporal_warped_error",
    "frame_diff_flow",
    "aggregate_temporal_metrics",
]

FlowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _as_float(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)


def _band_select(arr: np.ndarray, band: Optional[np.ndarray]) -> np.ndarray:
    return arr[band] if band is not None else arr


def temporal_flicker(
    alpha_seq: Sequence[np.ndarray],
    *,
    band: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Mean absolute frame-to-frame alpha delta over a sequence.

    Returns ``{"mean_flicker", "per_transition", "n_frames"}``. With fewer than
    two frames the mean is 0.0 and ``per_transition`` is empty.
    """
    n = len(alpha_seq)
    if n < 2:
        return {"mean_flicker": 0.0, "per_transition": [], "n_frames": n}
    deltas: list[float] = []
    for prev, cur in zip(alpha_seq[:-1], alpha_seq[1:]):
        d = np.abs(_as_float(prev) - _as_float(cur))
        sel = _band_select(d, band)
        deltas.append(float(sel.mean()) if sel.size else 0.0)
    return {
        "mean_flicker": float(np.mean(deltas)),
        "per_transition": deltas,
        "n_frames": n,
    }


def masklet_temporal_iou(mask_seq: Sequence[np.ndarray]) -> dict[str, object]:
    """Mean IoU between consecutive masks in a masklet (identity/track consistency).

    Empty-frame pairs (both masks empty) score 1.0 (stable empty track).
    """
    n = len(mask_seq)
    if n < 2:
        return {"mean_iou": 1.0, "per_transition": [], "n_frames": n}
    ious: list[float] = []
    for prev, cur in zip(mask_seq[:-1], mask_seq[1:]):
        a = np.asarray(prev).astype(bool)
        b = np.asarray(cur).astype(bool)
        if a.shape != b.shape:
            raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        ious.append(float(inter / union) if union else 1.0)
    return {
        "mean_iou": float(np.mean(ious)),
        "per_transition": ious,
        "n_frames": n,
    }


def warp_with_flow(prev: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp ``prev`` to align with the next frame using ``flow``.

    ``flow`` is the **backward displacement** (dx, dy in pixels): for each pixel
    ``p`` in the *current* frame, ``flow[p]`` says where that content came from
    in ``prev``. We sample ``prev`` at ``p + flow[p]`` via ``cv2.remap`` (border
    replication), producing an image aligned with the current frame. So to
    compensate a feature that moved right by ``d`` between prev and cur, supply
    ``flow_x = -d`` (look back to where the content was).
    """
    import cv2

    prev = np.asarray(prev, dtype=np.float32)
    h, w = prev.shape[:2]
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape != (h, w, 2):
        raise ValueError(f"flow shape {flow.shape} != ({h}, {w}, 2)")
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + flow[..., 0]
    map_y = ys + flow[..., 1]
    return cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def temporal_warped_error(
    alpha_seq: Sequence[np.ndarray],
    *,
    flow_fn: Optional[FlowFn] = None,
    band: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Motion-compensated temporal error between consecutive alphas.

    For each transition, ``prev`` is warped to ``cur``'s frame with ``flow_fn``
    (when supplied) and the L1 error vs ``cur`` is computed. With
    ``flow_fn=None`` the warp is skipped (zero-flow CPU fallback), so the
    warped error equals the raw flicker.
    """
    n = len(alpha_seq)
    if n < 2:
        return {"mean_warped_error": 0.0, "per_transition": [], "n_frames": n}
    errs: list[float] = []
    for prev, cur in zip(alpha_seq[:-1], alpha_seq[1:]):
        prev_f = _as_float(prev)
        cur_f = _as_float(cur)
        if prev_f.shape != cur_f.shape:
            raise ValueError(f"alpha shape mismatch: {prev_f.shape} vs {cur_f.shape}")
        if flow_fn is None:
            warped = prev_f
        else:
            flow = flow_fn(prev_f, cur_f)
            warped = warp_with_flow(prev_f, flow)
        d = np.abs(warped - cur_f)
        sel = _band_select(d, band)
        errs.append(float(sel.mean()) if sel.size else 0.0)
    return {
        "mean_warped_error": float(np.mean(errs)),
        "per_transition": errs,
        "n_frames": n,
    }


def frame_diff_flow(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """CPU fallback flow: zeros (no motion compensation).

    Using this as ``flow_fn`` makes :func:`temporal_warped_error` identical to
    :func:`temporal_flicker`, which is the documented CPU smoke behavior.
    """
    h, w = np.asarray(prev).shape[:2]
    return np.zeros((h, w, 2), dtype=np.float32)


def aggregate_temporal_metrics(
    alpha_seqs: Sequence[Sequence[np.ndarray]],
    *,
    flow_fn: Optional[FlowFn] = None,
    bands: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> dict[str, float]:
    """Mean flicker / warped-error / masklet-IoU over many alpha sequences.

    ``bands`` optionally restricts each sequence's flicker/warped error to an
    unknown band. ``masklet_iou`` is computed on the binarized alphas (>0.5).
    """
    if bands is not None and len(bands) != len(alpha_seqs):
        raise ValueError("bands length does not match alpha_seqs")
    flickers: list[float] = []
    warps: list[float] = []
    ious: list[float] = []
    for i, seq in enumerate(alpha_seqs):
        band = bands[i] if bands is not None else None
        flickers.append(temporal_flicker(seq, band=band)["mean_flicker"])
        warps.append(temporal_warped_error(seq, flow_fn=flow_fn, band=band)["mean_warped_error"])
        ious.append(masklet_temporal_iou([np.asarray(a) > 0.5 for a in seq])["mean_iou"])
    if not alpha_seqs:
        return {"n_sequences": 0.0, "mean_flicker": 0.0, "mean_warped_error": 0.0, "mean_masklet_iou": 1.0}
    return {
        "n_sequences": float(len(alpha_seqs)),
        "mean_flicker": float(np.mean(flickers)),
        "mean_warped_error": float(np.mean(warps)),
        "mean_masklet_iou": float(np.mean(ious)),
    }
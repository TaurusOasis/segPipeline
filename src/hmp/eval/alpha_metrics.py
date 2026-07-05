"""Alpha-matte quality metrics (pipeline step 8 / matting evaluation).

Implements the standard alphamatting.com benchmark metrics so matting quality
can be measured against alpha-GT datasets (COCO-Matting, HIM, HHM — see
``configs/datasets.yaml``). COCONut itself only has binary masks, so these
metrics apply to the matting datasets, not the COCONut mask benchmark.

All metrics take alpha / gt as float arrays in ``[0, 1]``. An optional trimap
(or boolean unknown mask) restricts the computation to the unknown band; when
omitted the metric is computed over the whole image.

Metrics:
* :func:`sad`      — sum of absolute differences (×1e-3, alphamatting convention)
* :func:`mad`      — mean absolute difference
* :func:`mse`      — mean squared error
* :func:`gradient_error` — L1 of gradient magnitudes (×1e-3)
* :func:`connectivity_error` — Xu et al. 2017 connectivity term (×1e-3)
* :func:`aggregate_alpha_metrics` — mean over a list of (pred, gt) pairs
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "sad",
    "mad",
    "mse",
    "gradient_error",
    "connectivity_error",
    "aggregate_alpha_metrics",
]


def _unknown_mask(trimap: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Normalize a trimap into a boolean unknown-region mask.

    Accepts either a boolean array (True = unknown) or a uint8 trimap where
    128 (or any value strictly between 0 and 255) marks the unknown band.
    Returns ``None`` when no trimap is given (compute over the whole image).
    """
    if trimap is None:
        return None
    trimap = np.asarray(trimap)
    if trimap.shape != shape:
        raise ValueError(f"trimap shape {trimap.shape} != alpha shape {shape}")
    if trimap.dtype == bool:
        return trimap
    f = trimap.astype(np.float32)
    return (f > 0.0) & (f < 255.0)


def _select(arr: np.ndarray, unknown: Optional[np.ndarray]) -> np.ndarray:
    return arr[unknown] if unknown is not None else arr.ravel()


def sad(pred: np.ndarray, gt: np.ndarray, trimap: Optional[np.ndarray] = None) -> float:
    """Sum of absolute differences in thousands (alphamatting.com convention)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    unknown = _unknown_mask(trimap, pred.shape)
    return float(np.abs(_select(pred, unknown) - _select(gt, unknown)).sum() / 1000.0)


def mad(pred: np.ndarray, gt: np.ndarray, trimap: Optional[np.ndarray] = None) -> float:
    """Mean absolute difference in [0, 1]."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    unknown = _unknown_mask(trimap, pred.shape)
    diff = _select(pred, unknown) - _select(gt, unknown)
    return float(np.mean(np.abs(diff))) if diff.size else 0.0


def mse(pred: np.ndarray, gt: np.ndarray, trimap: Optional[np.ndarray] = None) -> float:
    """Mean squared error in [0, 1]."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    unknown = _unknown_mask(trimap, pred.shape)
    diff = _select(pred, unknown) - _select(gt, unknown)
    return float(np.mean(diff * diff)) if diff.size else 0.0


def gradient_error(pred: np.ndarray, gt: np.ndarray, trimap: Optional[np.ndarray] = None) -> float:
    """L1 of gradient magnitudes in thousands.

    Uses central differences (np.gradient). The per-pixel gradient magnitude
    difference is summed over the unknown band and divided by 1000.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    unknown = _unknown_mask(trimap, pred.shape)

    def _grad_mag(a: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(a)
        return np.sqrt(gx * gx + gy * gy)

    diff = np.abs(_grad_mag(pred) - _grad_mag(gt))
    return float(_select(diff, unknown).sum() / 1000.0)


def _distance_to_foreground(fg: np.ndarray) -> np.ndarray:
    """Distance from each pixel to the nearest foreground pixel.

    SciPy gives the reference EDT when available. OpenCV is a good lightweight
    fallback in the project envs. The final NumPy path is intentionally simple
    and mainly keeps CPU tests independent of binary scipy wheels.
    """
    try:
        from scipy import ndimage

        return ndimage.distance_transform_edt(~fg)
    except Exception:
        pass

    try:
        import cv2

        return cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    except Exception:
        pass

    coords = np.argwhere(fg)
    if coords.size == 0:
        return np.full(fg.shape, np.inf, dtype=np.float64)
    yy, xx = np.indices(fg.shape)
    dist2 = np.full(fg.shape, np.inf, dtype=np.float64)
    for y, x in coords:
        dist2 = np.minimum(dist2, (yy - y) * (yy - y) + (xx - x) * (xx - x))
    return np.sqrt(dist2)


def _connectivity_map(alpha: np.ndarray, unknown: np.ndarray, rad: int, dt: float) -> np.ndarray:
    """Per-pixel connectivity term for one alpha (Xu et al. 2017).

    For each threshold level the foreground set is ``(alpha >= level)``. A pixel
    is "connected" at that level if it lies within ``rad`` of the foreground set
    (via an EDT of the complement). The connectivity term is
    ``1 - (1/N) * sum_levels connected``, accumulated only inside ``unknown``.
    """
    h, w = alpha.shape
    levels = np.arange(0.0, 1.0, dt)
    n_levels = len(levels)
    acc = np.zeros((h, w), dtype=np.float64)
    a255 = alpha * 255.0
    for level in levels:
        fg = a255 >= (level * 255.0)
        # EDT of the complement: distance from every pixel to the nearest fg pixel.
        dist = _distance_to_foreground(fg)
        connected = (dist <= rad).astype(np.float64)
        acc += connected
    acc /= n_levels
    conn = 1.0 - acc
    conn[~unknown] = 0.0
    return conn


def connectivity_error(
    pred: np.ndarray,
    gt: np.ndarray,
    trimap: Optional[np.ndarray] = None,
    *,
    rad: Optional[int] = None,
    dt: float = 0.05,
) -> float:
    """Connectivity error (Xu et al. 2017) in thousands.

    Requires an unknown band (trimap). When no trimap is given the metric is
    undefined and returns 0.0. ``rad`` defaults to ``ceil(sqrt(1/(dt*pi)))``
    as in the reference benchmark. ``dt`` is the level-set step (default 0.05
    for speed; the canonical value is 0.01).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    unknown = _unknown_mask(trimap, pred.shape)
    if unknown is None or not unknown.any():
        return 0.0
    if rad is None:
        rad = int(np.ceil(np.sqrt(1.0 / (dt * np.pi))))
    c_pred = _connectivity_map(pred, unknown, rad, dt)
    c_gt = _connectivity_map(gt, unknown, rad, dt)
    diff = np.abs(c_pred - c_gt)
    diff[~unknown] = 0.0
    return float(diff.sum() / 1000.0)


def aggregate_alpha_metrics(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    trimaps: Optional[list[Optional[np.ndarray]]] = None,
    *,
    with_connectivity: bool = False,
) -> dict[str, float]:
    """Mean SAD / MAD / MSE / gradient (and optionally connectivity) over pairs."""
    if len(preds) != len(gts):
        raise ValueError(f"preds and gts length mismatch: {len(preds)} vs {len(gts)}")
    if trimaps is not None and len(trimaps) != len(preds):
        raise ValueError("trimaps length does not match preds")
    if not preds:
        return {"count": 0.0, "sad": 0.0, "mad": 0.0, "mse": 0.0, "gradient_error": 0.0}

    sads: list[float] = []
    mads: list[float] = []
    mses: list[float] = []
    grads: list[float] = []
    conns: list[float] = []
    for i, (p, g) in enumerate(zip(preds, gts)):
        t = trimaps[i] if trimaps is not None else None
        sads.append(sad(p, g, t))
        mads.append(mad(p, g, t))
        mses.append(mse(p, g, t))
        grads.append(gradient_error(p, g, t))
        if with_connectivity:
            conns.append(connectivity_error(p, g, t))

    out: dict[str, float] = {
        "count": float(len(preds)),
        "sad": float(np.mean(sads)),
        "mad": float(np.mean(mads)),
        "mse": float(np.mean(mses)),
        "gradient_error": float(np.mean(grads)),
    }
    if with_connectivity:
        out["connectivity_error"] = float(np.mean(conns))
    return out

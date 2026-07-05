"""Tests for boundary metrics (Step 05)."""

from __future__ import annotations

import numpy as np

from hmp.eval.boundary_metrics import (
    aggregate_boundary_metrics,
    boundary_f_score,
    boundary_iou,
    boundary_precision_recall_fscore,
    mask_iou,
    mask_to_boundary,
)


def _square(h=50, w=50, r=10, s=20):
    m = np.zeros((h, w), bool)
    m[s : s + r, s : s + r] = True
    return m


def test_mask_to_boundary_nonempty():
    b = mask_to_boundary(_square())
    assert b.any()
    # interior is not part of the boundary band
    assert not b[22:28, 22:28].any()


def test_mask_to_boundary_empty_returns_zeros():
    b = mask_to_boundary(np.zeros((20, 20), bool))
    assert not b.any()


def test_perfect_mask_boundary_iou_is_one():
    m = _square()
    assert abs(boundary_iou(m, m) - 1.0) < 1e-9


def test_shifted_mask_lower_boundary_iou():
    m = _square()
    shifted = np.zeros_like(m)
    shifted[22:32, 22:32] = True  # shifted by 2px
    biou = boundary_iou(m, shifted)
    assert 0.0 < biou < 1.0


def test_partial_overlap_has_mid_iou():
    m = _square()
    other = np.zeros_like(m)
    other[20:35, 20:35] = True  # large overlap, partial
    assert 0.0 < boundary_iou(m, other) < 1.0


def test_both_empty_boundary_iou_one():
    assert boundary_iou(np.zeros((20, 20), bool), np.zeros((20, 20), bool)) == 1.0


def test_one_empty_boundary_iou_zero():
    assert boundary_iou(_square(), np.zeros((50, 50), bool)) == 0.0


def test_boundary_precision_recall_fscore_perfect():
    m = _square()
    p, r, f = boundary_precision_recall_fscore(m, m)
    assert abs(p - 1.0) < 1e-9
    assert abs(r - 1.0) < 1e-9
    assert abs(f - 1.0) < 1e-9


def test_boundary_f_score_helper():
    m = _square()
    assert abs(boundary_f_score(m, m) - 1.0) < 1e-9


def test_mask_iou_basic():
    a = _square()
    b = _square()
    assert abs(mask_iou(a, b) - 1.0) < 1e-9
    c = np.zeros_like(a)
    c[25:35, 25:35] = True
    iou = mask_iou(a, c)
    assert 0.0 < iou < 1.0


def test_mask_iou_both_empty():
    assert mask_iou(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) == 1.0


def test_aggregate_boundary_metrics():
    m = _square()
    res = aggregate_boundary_metrics([m, m], [m, m])
    assert abs(res["boundary_iou"] - 1.0) < 1e-9
    assert res["count"] == 2.0
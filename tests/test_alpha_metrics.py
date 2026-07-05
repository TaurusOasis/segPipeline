"""Tests for alpha-matte quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hmp.eval.alpha_metrics import (
    aggregate_alpha_metrics,
    connectivity_error,
    gradient_error,
    mad,
    mse,
    sad,
)


def test_perfect_match_zero_error():
    a = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    assert sad(a, a) == 0.0
    assert mad(a, a) == 0.0
    assert mse(a, a) == 0.0
    assert gradient_error(a, a) == 0.0


def test_mad_and_mse_known_values():
    gt = np.zeros((10, 10), dtype=np.float32)
    pred = np.full((10, 10), 0.2, dtype=np.float32)
    # |0.2 - 0| = 0.2 everywhere -> MAD = 0.2, MSE = 0.04
    assert mad(pred, gt) == pytest.approx(0.2, abs=1e-6)
    assert mse(pred, gt) == pytest.approx(0.04, abs=1e-6)
    # SAD = sum|0.2| = 0.2 * 100 = 20 -> /1000 = 0.02
    assert sad(pred, gt) == pytest.approx(0.02, abs=1e-6)


def test_sad_halves_with_half_band_error():
    gt = np.zeros((10, 10), dtype=np.float32)
    pred = np.zeros((10, 10), dtype=np.float32)
    pred[:5] = 0.5  # 50 px off by 0.5 -> SAD = 25/1000 = 0.025
    assert sad(pred, gt) == pytest.approx(0.025, abs=1e-6)


def test_trimap_restricts_to_unknown_band():
    gt = np.zeros((10, 10), dtype=np.float32)
    pred = np.full((10, 10), 0.5, dtype=np.float32)
    trimap = np.zeros((10, 10), dtype=np.uint8)
    trimap[:, :] = 0  # background
    trimap[5:, :] = 255  # foreground
    trimap[3:5, :] = 128  # unknown band (2 rows = 20 px)
    # Only the 20 unknown px count: MAD = 0.5, SAD = 0.5*20/1000 = 0.01
    assert mad(pred, gt, trimap) == pytest.approx(0.5, abs=1e-6)
    assert sad(pred, gt, trimap) == pytest.approx(0.01, abs=1e-6)
    # The full-image MAD would be 0.5; trimap must reduce the counted set, not the value here.
    assert mad(pred, gt) == pytest.approx(0.5, abs=1e-6)


def test_boolean_unknown_mask_supported():
    gt = np.zeros((8, 8), dtype=np.float32)
    pred = np.full((8, 8), 0.25, dtype=np.float32)
    unknown = np.zeros((8, 8), dtype=bool)
    unknown[0:2, :] = True  # 16 px
    assert mad(pred, gt, unknown) == pytest.approx(0.25, abs=1e-6)
    assert sad(pred, gt, unknown) == pytest.approx(0.25 * 16 / 1000, abs=1e-6)


def test_trimap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sad(np.zeros((4, 4)), np.zeros((4, 4)), trimap=np.zeros((5, 5), dtype=np.uint8))


def test_gradient_error_nonzero_for_shifted_edge():
    gt = np.zeros((20, 20), dtype=np.float32)
    gt[:, :10] = 1.0  # vertical edge at x=10
    pred = np.zeros((20, 20), dtype=np.float32)
    pred[:, :11] = 1.0  # edge shifted by 1 px
    assert gradient_error(pred, gt) > 0.0
    # Identical -> zero.
    assert gradient_error(gt, gt) == 0.0


def test_connectivity_error_zero_for_identical():
    a = np.zeros((30, 30), dtype=np.float32)
    a[5:25, 5:25] = 1.0
    trimap = np.zeros((30, 30), dtype=np.uint8)
    trimap[3:27, 3:27] = 128
    assert connectivity_error(a, a, trimap) == 0.0


def test_connectivity_error_nonzero_for_different_structure():
    gt = np.zeros((40, 40), dtype=np.float32)
    gt[10:30, 10:30] = 1.0  # solid square
    pred = gt.copy()
    pred[15:25, 15:25] = 0.0  # hole -> very different connectivity
    trimap = np.zeros((40, 40), dtype=np.uint8)
    trimap[8:32, 8:32] = 128
    assert connectivity_error(pred, gt, trimap) > 0.0


def test_connectivity_error_without_trimap_returns_zero():
    a = np.zeros((10, 10), dtype=np.float32)
    b = np.full((10, 10), 0.5, dtype=np.float32)
    assert connectivity_error(a, b, trimap=None) == 0.0


def test_aggregate_alpha_metrics_mean_over_pairs():
    gt = [np.zeros((10, 10), dtype=np.float32), np.zeros((10, 10), dtype=np.float32)]
    pred = [np.full((10, 10), 0.2, dtype=np.float32), np.full((10, 10), 0.4, dtype=np.float32)]
    agg = aggregate_alpha_metrics(pred, gt)
    assert agg["count"] == 2.0
    assert agg["mad"] == pytest.approx(0.3, abs=1e-6)
    assert agg["mse"] == pytest.approx((0.04 + 0.16) / 2, abs=1e-6)


def test_aggregate_alpha_metrics_empty():
    agg = aggregate_alpha_metrics([], [])
    assert agg["count"] == 0.0
    assert agg["sad"] == 0.0


def test_aggregate_alpha_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        aggregate_alpha_metrics([np.zeros((4, 4))], [np.zeros((4, 4)), np.zeros((4, 4))])
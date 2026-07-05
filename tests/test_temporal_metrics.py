"""Tests for temporal consistency metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hmp.eval.temporal_metrics import (
    aggregate_temporal_metrics,
    frame_diff_flow,
    masklet_temporal_iou,
    temporal_flicker,
    temporal_warped_error,
    warp_with_flow,
)


def test_temporal_flicker_constant_sequence_zero():
    a = np.zeros((10, 10), dtype=np.float32)
    a[2:8, 2:8] = 0.5
    seq = [a, a, a, a]
    out = temporal_flicker(seq)
    assert out["mean_flicker"] == 0.0
    assert out["n_frames"] == 4
    assert len(out["per_transition"]) == 3


def test_temporal_flicker_known_step():
    s0 = np.zeros((4, 4), dtype=np.float32)
    s1 = np.full((4, 4), 0.2, dtype=np.float32)  # delta 0.2
    s2 = np.full((4, 4), 0.5, dtype=np.float32)  # delta 0.3
    out = temporal_flicker([s0, s1, s2])
    assert out["per_transition"] == [pytest.approx(0.2), pytest.approx(0.3)]
    assert out["mean_flicker"] == pytest.approx(0.25)


def test_temporal_flicker_band_restricts():
    s0 = np.zeros((4, 4), dtype=np.float32)
    s1 = np.full((4, 4), 0.5, dtype=np.float32)
    band = np.zeros((4, 4), dtype=bool)
    band[0, 0] = True  # only one pixel counted
    out = temporal_flicker([s0, s1], band=band)
    assert out["mean_flicker"] == pytest.approx(0.5)


def test_temporal_flicker_short_sequence():
    assert temporal_flicker([np.zeros((4, 4))])["mean_flicker"] == 0.0
    assert temporal_flicker([])["mean_flicker"] == 0.0


def test_masklet_temporal_iou_identical_is_one():
    m = np.zeros((10, 10), dtype=bool)
    m[3:7, 3:7] = True
    out = masklet_temporal_iou([m, m, m])
    assert out["mean_iou"] == 1.0
    assert out["per_transition"] == [1.0, 1.0]


def test_masklet_temporal_iou_disjoint_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert masklet_temporal_iou([a, b])["mean_iou"] == 0.0


def test_masklet_temporal_iou_both_empty_is_one():
    empty = np.zeros((10, 10), dtype=bool)
    assert masklet_temporal_iou([empty, empty])["mean_iou"] == 1.0


def test_masklet_temporal_iou_shape_mismatch_raises():
    with pytest.raises(ValueError):
        masklet_temporal_iou([np.zeros((4, 4), dtype=bool), np.zeros((5, 5), dtype=bool)])


def test_warp_with_flow_identity_returns_same():
    a = np.zeros((8, 8), dtype=np.float32)
    a[2:6, 2:6] = 1.0
    flow = np.zeros((8, 8, 2), dtype=np.float32)
    warped = warp_with_flow(a, flow)
    assert np.allclose(warped, a)


def test_warp_with_flow_shifts_block():
    a = np.zeros((10, 10), dtype=np.float32)
    a[4:6, 4:6] = 1.0
    flow = np.zeros((10, 10, 2), dtype=np.float32)
    flow[..., 0] = -2.0  # backward displacement: content moved right by 2
    warped = warp_with_flow(a, flow)
    # The bright block should move right by ~2px (now at cols 6:8).
    assert warped[4:6, 6:8].max() > 0.5
    assert warped[4:6, 4:6].max() < 0.5


def test_warp_with_flow_shape_mismatch_raises():
    with pytest.raises(ValueError):
        warp_with_flow(np.zeros((4, 4), dtype=np.float32), np.zeros((5, 5, 2), dtype=np.float32))


def test_temporal_warped_error_no_flow_equals_flicker():
    s0 = np.zeros((4, 4), dtype=np.float32)
    s1 = np.full((4, 4), 0.3, dtype=np.float32)
    s2 = np.full((4, 4), 0.6, dtype=np.float32)
    seq = [s0, s1, s2]
    assert temporal_warped_error(seq)["mean_warped_error"] == pytest.approx(
        temporal_flicker(seq)["mean_flicker"]
    )


def test_temporal_warped_error_frame_diff_flow_matches_no_flow():
    s0 = np.zeros((4, 4), dtype=np.float32)
    s1 = np.full((4, 4), 0.4, dtype=np.float32)
    seq = [s0, s1]
    e_none = temporal_warped_error(seq)["mean_warped_error"]
    e_fdf = temporal_warped_error(seq, flow_fn=frame_diff_flow)["mean_warped_error"]
    assert e_none == pytest.approx(e_fdf)


def test_temporal_warped_error_motion_compensated_reduces_error():
    # A block translating by 1px/frame: raw flicker is large, flow-compensated ~0.
    a = np.zeros((10, 10), dtype=np.float32)
    a[4:7, 4:7] = 1.0
    b = np.zeros((10, 10), dtype=np.float32)
    b[4:7, 5:8] = 1.0  # shifted +1 in x

    def shift_flow(prev, cur):
        # backward displacement: content moved right by 1 -> look back -1 in x
        f = np.zeros(prev.shape[:2] + (2,), dtype=np.float32)
        f[..., 0] = -1.0
        return f

    raw = temporal_flicker([a, b])["mean_flicker"]
    comp = temporal_warped_error([a, b], flow_fn=shift_flow)["mean_warped_error"]
    assert comp < raw


def test_temporal_warped_error_shape_mismatch_raises():
    with pytest.raises(ValueError):
        temporal_warped_error([np.zeros((4, 4)), np.zeros((5, 5))])


def test_aggregate_temporal_metrics_means_over_sequences():
    seq_a = [np.zeros((4, 4), dtype=np.float32), np.full((4, 4), 0.6, dtype=np.float32)]
    seq_b = [np.zeros((4, 4), dtype=np.float32), np.full((4, 4), 0.8, dtype=np.float32)]
    agg = aggregate_temporal_metrics([seq_a, seq_b])
    assert agg["n_sequences"] == 2.0
    assert agg["mean_flicker"] == pytest.approx(0.7)
    assert agg["mean_warped_error"] == pytest.approx(0.7)
    assert agg["mean_masklet_iou"] == 0.0  # binarized: empty vs full -> IoU 0


def test_aggregate_temporal_metrics_empty():
    agg = aggregate_temporal_metrics([])
    assert agg["n_sequences"] == 0.0
    assert agg["mean_masklet_iou"] == 1.0


def test_aggregate_temporal_metrics_bands_length_mismatch_raises():
    with pytest.raises(ValueError):
        aggregate_temporal_metrics([[np.zeros((4, 4))]], bands=[None, None])
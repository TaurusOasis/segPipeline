"""Tests for adaptive trimap generation."""

from __future__ import annotations

import numpy as np

from hmp.matting.adaptive_trimap import make_adaptive_trimap


def test_adaptive_trimap_has_three_regions():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    tri, roi = make_adaptive_trimap(mask, base_radius=4, max_radius=8)
    assert tri.shape == mask.shape
    assert set(np.unique(tri)).issubset({0, 128, 255})
    assert roi["foreground_core"].any()
    assert roi["background_core"].any()
    assert roi["unknown_roi"].any()
    assert not np.any(roi["foreground_core"] & roi["background_core"])


def test_adaptive_trimap_widens_with_motion_flag():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    tri_plain, _ = make_adaptive_trimap(mask, base_radius=4, motion_blur=False)
    tri_motion, _ = make_adaptive_trimap(mask, base_radius=4, motion_blur=True)
    assert np.sum(tri_motion == 128) >= np.sum(tri_plain == 128)

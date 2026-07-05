"""Tests for hmp.data.mask_io and hmp.refine.mask_postprocess (Step 04)."""

from __future__ import annotations

import numpy as np

from hmp.data.mask_io import (
    combine_instance_masks,
    fill_holes,
    keep_largest_component,
    mask_area_ratio,
    mask_to_bbox_xyxy,
    read_binary_mask,
    remove_small_components,
    write_binary_mask,
)
from hmp.refine.mask_postprocess import postprocess_from_config, postprocess_mask


def test_write_and_read_roundtrip(tmp_path):
    m = np.zeros((10, 10), bool)
    m[2:5, 3:7] = True
    p = tmp_path / "m.png"
    write_binary_mask(p, m)
    back = read_binary_mask(p)
    assert back.shape == (10, 10)
    assert np.array_equal(back, m)


def test_mask_to_bbox():
    m = np.zeros((20, 30), bool)
    m[5:10, 7:15] = True
    assert mask_to_bbox_xyxy(m) == [7, 5, 15, 10]


def test_mask_to_bbox_empty():
    assert mask_to_bbox_xyxy(np.zeros((10, 10), bool)) is None


def test_mask_area_ratio():
    m = np.zeros((10, 10), bool)
    m[0:5, 0:5] = True  # 25 / 100
    assert abs(mask_area_ratio(m) - 0.25) < 1e-9


def test_remove_small_components():
    m = np.zeros((50, 50), bool)
    m[5:25, 5:25] = True   # big component (400 px)
    m[40:42, 40:42] = True  # tiny (4 px)
    out = remove_small_components(m, min_area=64)
    assert out[5:25, 5:25].any()
    assert not out[40:42, 40:42].any()


def test_fill_holes():
    m = np.zeros((20, 20), bool)
    m[3:17, 3:17] = True
    m[9:11, 9:11] = False  # a hole
    out = fill_holes(m)
    assert out[9:11, 9:11].all()  # hole filled
    assert out[3:17, 3:17].all()


def test_fill_holes_empty():
    out = fill_holes(np.zeros((10, 10), bool))
    assert not out.any()


def test_keep_largest_component():
    m = np.zeros((50, 50), bool)
    m[2:6, 2:6] = True    # 16 px
    m[20:40, 20:40] = True  # 400 px (largest)
    out = keep_largest_component(m)
    assert out[20:40, 20:40].any()
    assert not out[2:6, 2:6].any()


def test_combine_instance_masks():
    a = np.zeros((10, 10), bool); a[0:3, 0:3] = True
    b = np.zeros((10, 10), bool); b[5:8, 5:8] = True
    out = combine_instance_masks([a, b])
    assert out[0:3, 0:3].any() and out[5:8, 5:8].any()
    assert out.sum() == a.sum() + b.sum()


def test_postprocess_mask_pipeline():
    m = np.zeros((50, 50), bool)
    m[5:25, 5:25] = True
    m[10:14, 10:14] = False  # a true interior hole
    m[40:41, 40:41] = True  # tiny speck
    out = postprocess_mask(m, min_component_area=64)
    assert not out[40, 40]  # speck removed
    assert out[10:14, 10:14].all()  # hole filled
    assert out[5:25, 5:25].all()


def test_postprocess_from_config(tmp_path):
    from hmp.config import Config

    m = np.zeros((60, 60), bool)
    m[5:25, 5:25] = True
    m[45:46, 45:46] = True  # tiny
    cfg = Config(
        {
            "local_postprocess": {
                "remove_small_components": True,
                "min_component_area": 64,
                "fill_holes": True,
                "keep_largest_component": False,
            }
        }
    )
    out = postprocess_from_config(m, cfg)
    assert not out[45, 45]
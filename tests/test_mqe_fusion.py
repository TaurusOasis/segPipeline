"""Tests for MQE rule-based QA and alpha fusion."""

from __future__ import annotations

import numpy as np

from hmp.eval.mqe import rule_based_qa
from hmp.matting.alpha_fusion import fuse_alpha_branches


def test_rule_based_qa_perfect_alpha():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    alpha = mask.astype(np.float32)
    scores, reliable, failed = rule_based_qa(alpha=alpha, mask=mask)
    assert scores["core_score"] > 0.9
    assert failed == []


def test_rule_based_qa_detects_core_hole():
    mask = np.ones((16, 16), dtype=bool)
    alpha = np.zeros((16, 16), dtype=np.float32)
    _, _, failed = rule_based_qa(alpha=alpha, mask=mask, min_core_fill=0.85)
    assert "core_hole" in failed


def test_fuse_alpha_prefers_video_core_and_image_boundary():
    h, w = 16, 16
    fg = np.zeros((h, w), dtype=bool)
    fg[4:12, 4:12] = True
    unknown = np.zeros((h, w), dtype=bool)
    unknown[3:13, 3:13] = True
    unknown[fg] = False
    bg = ~(fg | unknown)
    branches = {
        "Bv": np.full((h, w), 0.9, dtype=np.float32),
        "Bi": np.full((h, w), 0.4, dtype=np.float32),
        "Bd": np.full((h, w), 0.35, dtype=np.float32),
        "Bs": np.full((h, w), 0.8, dtype=np.float32),
    }
    reliable = np.full((h, w), 0.9, dtype=np.float32)
    fused, eval_map, source, _ = fuse_alpha_branches(
        branches=branches,
        reliable_map=reliable,
        fg_core=fg,
        unknown_roi=unknown,
        bg_core=bg,
    )
    assert np.allclose(fused[fg], 0.9)
    assert np.allclose(fused[unknown], 0.4)
    assert source["Bv"]
    assert source["Bi"]

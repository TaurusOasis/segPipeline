"""Tests for the heuristic_v2 RL Prompt Agent."""

from __future__ import annotations

import numpy as np

from hmp.agents.prompt_agent import plan_prompts, select_keyframe


def _box_prompts(decision) -> list[dict]:
    return [p for p in decision.prompts if p["type"] == "box"]


def _neg_points(decision) -> list[tuple[int, int]]:
    return [
        tuple(p["xy"])
        for p in decision.prompts
        if p["type"] == "negative_point"
    ]


def _pos_points(decision) -> list[tuple[int, int]]:
    return [tuple(p["xy"]) for p in decision.prompts if p["type"] == "positive_point"]


def test_plan_prompts_v2_policy_and_baseline_prompts():
    d = plan_prompts(bbox_xyxy=[10, 10, 40, 50], width=64, height=64)
    assert d.policy == "heuristic_v2"
    # 1 box + 1 positive point + 1 background negative point.
    assert len(_box_prompts(d)) == 1
    assert len(_pos_points(d)) == 1
    assert len(_neg_points(d)) == 1
    assert d.needs_scribble is False
    assert d.confidence == 0.75


def test_neighbor_bboxes_add_one_negative_point_per_neighbor():
    d = plan_prompts(
        bbox_xyxy=[10, 10, 40, 50],
        width=128,
        height=128,
        neighbor_bboxes=[[60, 10, 90, 50], [10, 70, 40, 100]],
    )
    # 1 background negative + 2 neighbor negatives.
    negs = _neg_points(d)
    assert len(negs) == 3
    # Neighbor centers should appear: (75, 30) and (25, 85).
    assert (75, 30) in negs
    assert (25, 85) in negs
    assert "multi_person" in d.error_tags
    # Confidence drops with neighbors but is floored at 0.4.
    assert 0.4 <= d.confidence < 0.75


def test_neighbor_confidence_floor_with_many_neighbors():
    d = plan_prompts(
        bbox_xyxy=[10, 10, 40, 50],
        width=256,
        height=256,
        neighbor_bboxes=[[c, c, c + 10, c + 10] for c in range(0, 100, 10)],
    )
    assert d.confidence == 0.4


def test_neighbor_points_clamped_to_image_bounds():
    d = plan_prompts(
        bbox_xyxy=[0, 0, 10, 10],
        width=32,
        height=32,
        neighbor_bboxes=[[-100, -100, -90, -90]],  # center would be negative
    )
    negs = _neg_points(d)
    # Neighbor center clamped to (0, 0).
    assert (0, 0) in negs


def test_boundary_f1_below_threshold_forces_scribble():
    d = plan_prompts(bbox_xyxy=[10, 10, 40, 50], width=64, height=64, boundary_f1=0.55)
    assert d.needs_scribble is True
    assert "bad_boundary" in d.error_tags
    assert d.confidence <= 0.4


def test_boundary_f1_at_threshold_does_not_force_scribble():
    d = plan_prompts(bbox_xyxy=[10, 10, 40, 50], width=64, height=64, boundary_f1=0.7)
    assert d.needs_scribble is False
    assert "bad_boundary" not in d.error_tags


def test_small_person_area_ratio_triggers_scribble_and_tag():
    gt = np.zeros((64, 64), dtype=bool)
    gt[30:33, 30:33] = True  # 9 px / 4096 ~ 0.0022
    d = plan_prompts(bbox_xyxy=[10, 10, 40, 50], width=64, height=64, gt_mask=gt)
    assert d.needs_scribble is True
    assert "small_person" in d.error_tags


def test_large_person_area_ratio_triggers_scribble_and_tag():
    gt = np.zeros((64, 64), dtype=bool)
    gt[0:60, 0:60] = True  # 3600 / 4096 ~ 0.88
    d = plan_prompts(bbox_xyxy=[0, 0, 64, 64], width=64, height=64, gt_mask=gt)
    assert d.needs_scribble is True
    assert "large_person" in d.error_tags


def test_select_keyframe_picks_lowest_score():
    idx = select_keyframe(frame_indices=[0, 5, 10, 15], scores=[3.0, 1.0, 5.0, 2.0])
    assert idx == 5


def test_select_keyframe_ties_break_to_earliest():
    idx = select_keyframe(frame_indices=[0, 5, 10], scores=[2.0, 2.0, 2.0])
    assert idx == 0


def test_select_keyframe_empty_returns_zero():
    assert select_keyframe(frame_indices=[], scores=[]) == 0


def test_select_keyframe_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        select_keyframe(frame_indices=[0, 1], scores=[1.0])
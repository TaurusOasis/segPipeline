"""Tests for label-spec helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from hmp.data.label_spec import (
    active_yolo_class_names,
    compute_final_quality,
    decision_for_quality,
    load_class_map,
    load_qa_schema,
    quality_tier,
    train_weight_for_quality,
)

ROOT = Path(__file__).resolve().parents[1]


def test_active_yolo_class_names_excludes_alpha_layer():
    class_map = load_class_map(ROOT / "configs/class_map.yaml")
    assert active_yolo_class_names(class_map) == ["person"]
    assert class_map["semantic_layers"]["person_alpha"]["yolo_training"] is False


def test_compute_final_quality_uses_inverse_risk_fields():
    qa = load_qa_schema(ROOT / "configs/qa_schema.yaml")
    strong = {
        "class_score": 1.0,
        "box_score": 1.0,
        "mask_iou_agreement": 1.0,
        "boundary_score": 1.0,
        "edge_alignment": 1.0,
        "area_ratio_score": 1.0,
        "overlap_conflict": 0.0,
        "small_object_risk": 0.0,
        "teacher_disagreement": 0.0,
    }
    weak = dict(strong, overlap_conflict=1.0, small_object_risk=1.0, teacher_disagreement=1.0)
    assert compute_final_quality(strong, qa) == pytest.approx(1.0)
    assert compute_final_quality(weak, qa) < compute_final_quality(strong, qa)


def test_quality_tier_weight_and_decision_mapping():
    qa = load_qa_schema(ROOT / "configs/qa_schema.yaml")
    assert quality_tier(0.95, qa) == "gold"
    assert quality_tier(0.80, qa) == "silver"
    assert quality_tier(0.60, qa) == "bronze"
    assert quality_tier(0.20, qa) == "reject"
    assert train_weight_for_quality(0.80, qa) == pytest.approx(0.75)
    assert decision_for_quality(0.60, qa) == "review"
    assert decision_for_quality(0.20, qa) == "reject"

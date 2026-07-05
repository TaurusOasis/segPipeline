"""Tests for Phase-0 label spec, class map, and QA schema configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_phase0_label_spec_files_exist():
    for rel in ["LABEL_SPEC_zh.md", "configs/class_map.yaml", "configs/qa_schema.yaml", "configs/code_targets.yaml"]:
        assert (ROOT / rel).exists(), rel


def test_class_map_matches_code_targets_human_layers():
    class_map = _load_yaml("configs/class_map.yaml")
    code_targets = _load_yaml("configs/code_targets.yaml")

    semantic_layers = class_map["semantic_layers"]
    for layer, target in code_targets["human_labels"].items():
        assert layer in semantic_layers
        assert bool(semantic_layers[layer]["yolo_training"]) is bool(target["yolo_training"])

    active_set = class_map["class_sets"][class_map["export_defaults"]["active_class_set"]]
    class_ids = [row["class_id"] for row in active_set["classes"]]
    assert len(class_ids) == len(set(class_ids))
    for row in active_set["classes"]:
        layer = semantic_layers[row["semantic_layer"]]
        assert layer["target_type"] != "soft_alpha"
        assert layer["yolo_training"] is True


def test_qa_schema_quality_formula_and_tiers_are_consistent():
    qa = _load_yaml("configs/qa_schema.yaml")

    assert "final_quality" in qa["score_fields"]
    weights = qa["final_quality_formula"]["weights"]
    assert sum(float(v) for v in weights.values()) == pytest.approx(1.0, abs=1e-6)

    tiers = qa["quality_tiers"]
    assert tiers["gold"]["min_final_quality"] > tiers["silver"]["min_final_quality"] > tiers["bronze"]["min_final_quality"]
    assert tiers["gold"]["train_weight"] > tiers["silver"]["train_weight"] > tiers["bronze"]["train_weight"] > tiers["reject"]["train_weight"]
    assert tiers["reject"]["max_final_quality"] == tiers["bronze"]["min_final_quality"]

    for name, field in qa["score_fields"].items():
        lo, hi = field["range"]
        assert 0.0 <= float(lo) <= float(hi) <= 1.0, name


def test_qa_artifact_contract_keeps_yolo_and_alpha_separate():
    class_map = _load_yaml("configs/class_map.yaml")
    qa = _load_yaml("configs/qa_schema.yaml")

    assert class_map["semantic_layers"]["person_alpha"]["alpha_training"] is True
    assert class_map["semantic_layers"]["person_alpha"]["yolo_training"] is False
    assert qa["artifact_contract"]["train_weight_format"]["image_segmentation"] == "per_instance_float"
    assert qa["artifact_contract"]["train_weight_format"]["video_matting"] == "per_pixel_png_or_exr"

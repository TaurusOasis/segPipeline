"""Tests for edge vs GPU teacher model tiers."""

from __future__ import annotations

from hmp.config import Config
from hmp.labeling.auto_label_core import labeling_runtime_from_config
from hmp.labeling.labeler_factory import make_labeler
from hmp.models.tiers import load_model_tiers, resolve_teacher


def test_load_model_tiers_defaults():
    registry = load_model_tiers(Config({}))
    assert registry.edge.name == "yolo26s-seg"
    assert "sam2" in registry.teachers
    assert "samhq" in registry.teachers
    assert registry.teachers["yolo26x-seg"].kind == "distill"


def test_resolve_teacher_samhq():
    registry = load_model_tiers(Config({}))
    spec = resolve_teacher(registry, teacher_key="samhq")
    assert spec.backend == "samhq"


def test_labeling_runtime_uses_edge_and_teacher():
    cfg = Config(
        {
            "labeling": {
                "segment_teacher": "samhq",
                "teacher_weights": "sam_hq_vit_b.pt",
            }
        }
    )
    runtime = labeling_runtime_from_config(cfg)
    assert "yolo26s-seg" in runtime.yolo_weights
    assert runtime.teacher_key == "samhq"
    assert runtime.teacher is not None
    assert runtime.teacher.weights == "sam_hq_vit_b.pt"


def test_make_labeler_yolo_samhq(tmp_path):
    cfg = Config({"paths": {"masks_raw_dir": str(tmp_path / "masks")}})
    labeler = make_labeler(cfg, project_root=tmp_path, provider="yolo_samhq")
    assert labeler.teacher_key == "samhq"

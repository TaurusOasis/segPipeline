"""Tests for YOLO detector, SAM2 adapter, and benchmark compare."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import numpy as np
import pytest
import yaml

from hmp.agents.prompt_agent import plan_prompts
from hmp.eval.benchmark_compare import run_coconut_compare
from hmp.labeling.sam2_adapter import segment_with_sam2
from hmp.labeling.yolo_person_detector import (
    PersonDetection,
    bbox_iou,
    detect_persons,
    match_detection_for_gt,
)


def test_bbox_iou():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert bbox_iou([0, 0, 10, 10], [10, 10, 20, 20]) == pytest.approx(0.0)
    assert bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_match_detection_for_gt():
    dets = [
        PersonDetection(bbox_xyxy=[0, 0, 20, 20], score=0.9),
        PersonDetection(bbox_xyxy=[50, 50, 80, 80], score=0.8),
    ]
    used: set[int] = set()
    matched, iou = match_detection_for_gt(dets, [0, 0, 18, 18], used_indices=used, iou_threshold=0.3)
    assert matched is not None
    assert iou > 0.5
    assert 0 in used
    second, _ = match_detection_for_gt(dets, [52, 52, 78, 78], used_indices=used, iou_threshold=0.3)
    assert second is not None
    assert 1 in used


def test_sam2_adapter_falls_back_to_grabcut():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    gt = np.zeros((32, 32), dtype=bool)
    gt[8:24, 8:24] = True
    decision = plan_prompts(bbox_xyxy=[8, 8, 24, 24], width=32, height=32, gt_mask=gt)
    with patch.dict("sys.modules", {"ultralytics": None}):
        pred = segment_with_sam2(image, decision, fallback_grabcut=True)
    assert pred.shape == (32, 32)


def test_detect_persons_filters_person_class(monkeypatch):
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    class FakeBoxes:
        def __init__(self):
            self.xyxy = MagicMock()
            self.xyxy.cpu.return_value.numpy.return_value = np.array([[1, 2, 10, 12]], dtype=np.float32)
            self.conf = MagicMock()
            self.conf.cpu.return_value.numpy.return_value = np.array([0.91], dtype=np.float32)
            self.cls = MagicMock()
            self.cls.cpu.return_value.numpy.return_value = np.array([0], dtype=np.float32)

        def __len__(self):
            return 1

    class FakeResult:
        boxes = FakeBoxes()

    class FakeModel:
        def predict(self, **kwargs):
            assert kwargs.get("classes") == [0]
            return [FakeResult()]

    fake_yolo = MagicMock()
    fake_yolo.YOLO.return_value = FakeModel()
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_yolo)

    dets = detect_persons(image, weights="fake.pt")
    assert len(dets) == 1
    assert dets[0].bbox_xyxy == [1, 2, 10, 12]
    assert dets[0].score == pytest.approx(0.91)


def test_coconut_compare_dry_run(tmp_path):
    cfg = {
        "project": {"seed": 42},
        "coconut_benchmark": {
            "limit": 2,
            "output_dir": str(tmp_path / "bench"),
        },
        "coconut_compare": {
            "output_dir": str(tmp_path / "compare"),
            "modes": [["gt_bbox", "oracle"]],
        },
    }
    p = tmp_path / "compare.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    from hmp.config import load_config

    out = run_coconut_compare(load_config(p), project_root=tmp_path, dry_run=True)
    assert out.name == "compare_summary.md"


def test_coconut_compare_writes_iteration_plan(tmp_path):
    cfg = {
        "project": {"seed": 42},
        "local_postprocess": {"remove_small_components": True, "min_component_area": 8, "fill_holes": True},
        "coconut_benchmark": {
            "coconut_root": "/home/genesis/Train/Dataset/coconut",
            "image_root": "/home/genesis/Train/Dataset/coco2017",
            "json_path": "/home/genesis/Train/Dataset/coconut/relabeled_coco_val.json",
            "mask_dir": "/home/genesis/Train/Dataset/coconut/relabeled_coco_val",
            "image_subdir": "val2017",
            "limit": 2,
            "seed": 42,
            "output_dir": str(tmp_path / "bench"),
        },
        "coconut_compare": {
            "output_dir": str(tmp_path / "compare"),
            "modes": [["gt_bbox", "oracle"]],
        },
    }
    p = tmp_path / "compare.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    from hmp.config import load_config

    out = run_coconut_compare(load_config(p), project_root=tmp_path)
    assert out.exists()
    plan_path = tmp_path / "compare" / "iteration_plan.json"
    patch_path = tmp_path / "compare" / "next_config_patch.yaml"
    assert plan_path.exists()
    assert patch_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["selected_mode"]["detector_mode"] == "gt_bbox"
    assert plan["selected_mode"]["sam_mode"] == "oracle"
    assert plan["next_config_patch"]["coconut_benchmark"]["sam_mode"] == "oracle"


def test_coconut_compare_excludes_oracle_from_selection_when_real_mode_exists(tmp_path):
    from hmp.common.jsonl import write_jsonl
    from hmp.config import Config
    from hmp.schemas import BenchmarkRecord

    out_root = tmp_path / "compare"
    for det, sam, iou, bf1 in [
        ("gt_bbox", "oracle", 1.0, 1.0),
        ("gt_bbox", "grabcut", 0.7, 0.8),
    ]:
        mode_dir = out_root / f"{det}__{sam}"
        mode_dir.mkdir(parents=True)
        write_jsonl(
            mode_dir / "benchmark_records.jsonl",
            [
                BenchmarkRecord(
                    item_id="demo",
                    instance_id="person_0",
                    image_path="/tmp/demo.jpg",
                    gt_mask_path="/tmp/gt.png",
                    detector_mode=det,
                    sam_mode=sam,
                    mask_iou=iou,
                    boundary_f_score=bf1,
                    decision="accept",
                    elapsed_ms=1.0,
                )
            ],
            overwrite=True,
        )
        (mode_dir / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "instances": 1,
                    "detector_mode": det,
                    "sam_mode": sam,
                    "mean_mask_iou": iou,
                    "mean_boundary_f1": bf1,
                    "mean_bbox_iou": 1.0,
                    "accept_rate": 1.0,
                    "review_rate": 0.0,
                    "reject_rate": 0.0,
                    "decision_counts": {"accept": 1},
                    "error_buckets": {},
                    "quality_gates": {},
                    "mean_elapsed_ms": 1.0,
                    "instances_per_second": 1.0,
                    "total_seconds": 0.001,
                }
            ),
            encoding="utf-8",
        )

    cfg = Config(
        {
            "coconut_compare": {
                "output_dir": str(out_root),
                "modes": [["gt_bbox", "oracle"], ["gt_bbox", "grabcut"]],
                "allow_oracle_selection": False,
            }
        }
    )
    out = run_coconut_compare(cfg, project_root=tmp_path)
    assert out.exists()
    plan = json.loads((out_root / "iteration_plan.json").read_text(encoding="utf-8"))
    assert plan["ranked_modes"][0]["sam_mode"] == "oracle"
    assert plan["selected_mode"]["sam_mode"] == "grabcut"
    assert plan["selection_policy"]["oracle_modes_excluded"] is True

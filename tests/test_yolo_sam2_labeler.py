"""Tests for YOLO+SAM2 labeler and pipeline provider wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from typer.testing import CliRunner

from hmp.cli import app
from hmp.labeling.labeler_factory import make_labeler
from hmp.labeling.yolo_person_detector import PersonDetection
from hmp.labeling.yolo_sam2_labeler import YoloSam2Labeler
from hmp.schemas import MediaItem

runner = CliRunner()


def _item() -> MediaItem:
    return MediaItem(
        item_id="demo_00",
        path="/tmp/demo.jpg",
        width=64,
        height=96,
        sha256="abc",
    )


def test_make_labeler_providers(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"dummy": {}, "labeling": {}}), encoding="utf-8")
    from hmp.config import load_config

    cfg = load_config(cfg_path)
    assert make_labeler(cfg, project_root=tmp_path, provider="mock").name == "dummy"
    assert make_labeler(cfg, project_root=tmp_path, provider="yolo_sam2").segment_mode == "sam2"
    assert make_labeler(cfg, project_root=tmp_path, provider="yolo_grabcut").segment_mode == "grabcut"


def test_yolo_sam2_labeler_mocked(tmp_path, monkeypatch):
    cfg = {
        "paths": {
            "masks_raw_dir": str(tmp_path / "masks"),
            "annotation_path": str(tmp_path / "ann.jsonl"),
        },
        "labeling": {"segment_mode": "grabcut"},
        "local_postprocess": {"remove_small_components": False, "fill_holes": False},
    }
    from hmp.config import Config

    labeler = YoloSam2Labeler(Config(cfg), project_root=tmp_path, segment_mode="grabcut")

    image = np.zeros((96, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(labeler, "_read_image_bgr", lambda item: image)
    monkeypatch.setattr(
        "hmp.labeling.yolo_sam2_labeler.detect_persons",
        lambda *a, **k: [PersonDetection(bbox_xyxy=[16, 20, 48, 80], score=0.92)],
    )
    monkeypatch.setattr(
        "hmp.labeling.yolo_sam2_labeler.segment_with_prompts",
        lambda img, decision, **kw: np.ones((96, 64), dtype=bool),
    )

    instances = labeler.label_one(_item())
    assert len(instances) == 1
    assert instances[0].source == "yolo+grabcut"
    assert instances[0].prompt_history
    assert Path(instances[0].mask_path).exists()


def test_label_yolo_sam2_cli_dry_run(tmp_path):
    cfg = {
        "paths": {
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "masks_raw_dir": str(tmp_path / "masks"),
            "annotation_path": str(tmp_path / "ann.jsonl"),
        },
        "labeling": {},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    result = runner.invoke(app, ["label", "yolo-sam2", "--config", str(p), "--dry-run"])
    assert result.exit_code == 0, result.output

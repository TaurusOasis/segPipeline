"""Tests for COCONut accept-mask -> YOLO distillation bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from hmp.common.jsonl import write_jsonl
from hmp.config import Config
from hmp.data.mask_io import write_binary_mask
from hmp.schemas import AnnotationRecord, InstanceAnnotation, MediaItem
from hmp.yolo.coconut_distill_bridge import build_coconut_distill_plan, overlay_accept_labels_on_coconut


def _write_benchmark(tmp_path: Path, item_id: str = "img001") -> Path:
    benchmark_dir = tmp_path / "bench"
    pred_dir = benchmark_dir / "pred_masks"
    pred_dir.mkdir(parents=True)
    image_path = tmp_path / f"{item_id}.jpg"
    Image.new("RGB", (40, 40), (128, 64, 32)).save(image_path)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:25] = True
    mask_path = pred_dir / f"{item_id}_person_0_pred.png"
    write_binary_mask(mask_path, mask)
    write_jsonl(
        benchmark_dir / "manifest.jsonl",
        [
            MediaItem(
                item_id=item_id,
                media_type="image",
                path=str(image_path),
                width=40,
                height=40,
                sha256="abc",
            )
        ],
    )
    write_jsonl(
        benchmark_dir / "annotations_pred.jsonl",
        [
            AnnotationRecord(
                item_id=item_id,
                instances=[
                    InstanceAnnotation(
                        instance_id="person_0",
                        bbox_xyxy=[10, 10, 25, 30],
                        mask_path=str(mask_path),
                        prompt_history=[{"decision": "accept"}],
                    )
                ],
            )
        ],
    )
    return benchmark_dir


def _write_base_yolo(tmp_path: Path, item_id: str = "img001") -> Path:
    base = tmp_path / "COCONut_b_yolo_seg_v2"
    labels_val = base / "labels" / "val"
    images_val = base / "images" / "val"
    labels_train = base / "labels" / "train"
    images_train = base / "images" / "train"
    for d in (labels_val, images_val, labels_train, images_train):
        d.mkdir(parents=True)
    (labels_val / f"{item_id}.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")
    Image.new("RGB", (40, 40)).save(images_val / f"{item_id}.jpg")
    return base


def test_overlay_accept_labels_patches_val(tmp_path):
    item_id = "img001"
    benchmark_dir = _write_benchmark(tmp_path, item_id)
    base_root = _write_base_yolo(tmp_path, item_id)
    out_root = tmp_path / "overlay"
    cfg = Config(
        {
            "coconut_distill_bridge": {
                "benchmark_dir": str(benchmark_dir),
                "base_yolo_root": str(base_root),
                "output_yolo_root": str(out_root),
                "import_decisions": ["accept"],
            }
        }
    )
    data_yaml = overlay_accept_labels_on_coconut(cfg, project_root=tmp_path)
    assert data_yaml.exists()
    label = (out_root / "labels" / "val" / f"{item_id}.txt").read_text(encoding="utf-8")
    assert label.startswith("0 ")
    stats = yaml.safe_load((out_root / "accept_overlay_stats.json").read_text(encoding="utf-8"))
    assert stats["patched_count"] == 1


def test_build_coconut_distill_plan_includes_ultralytics_command(tmp_path):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "coconut_distill_bridge": {
                    "output_yolo_root": str(tmp_path / "overlay"),
                    "ultralytics_root": str(tmp_path / "ultralytics"),
                    "student_weights": "/tmp/student.pt",
                    "teacher_weights": "/tmp/teacher.pt",
                }
            }
        ),
        encoding="utf-8",
    )
    from hmp.config import load_config

    cfg = load_config(cfg_path)
    plan = build_coconut_distill_plan(cfg, project_root=tmp_path, data_yaml=tmp_path / "overlay" / "data.yaml")
    assert "train_yolo26s_seg_coconut_distill.py" in plan["command"]
    assert "/tmp/teacher.pt" in plan["command"]
    assert "/tmp/student.pt" in plan["command"]

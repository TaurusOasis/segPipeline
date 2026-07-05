"""Tests for benchmark -> pipeline bridge helpers."""

from __future__ import annotations

import yaml

from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config, load_config
from hmp.eval.benchmark_bridge import apply_iteration_patch, import_benchmark_annotations
from hmp.schemas import AnnotationRecord, InstanceAnnotation


def test_import_benchmark_annotations(tmp_path):
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    write_jsonl(
        benchmark_dir / "annotations_pred.jsonl",
        [
            AnnotationRecord(
                item_id="img1",
                instances=[
                    InstanceAnnotation(
                        instance_id="person_0",
                        bbox_xyxy=[0, 0, 10, 10],
                        mask_path=str(tmp_path / "pred.png"),
                    )
                ],
            )
        ],
    )
    out = import_benchmark_annotations(
        benchmark_dir,
        annotation_path=tmp_path / "annotations_raw.jsonl",
    )
    rows = read_jsonl_list(out, model=AnnotationRecord)
    assert len(rows) == 1
    assert rows[0].item_id == "img1"


def test_apply_iteration_patch_merges_nested_keys(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "labeling": {"yolo_conf": 0.25, "quality_gates": {"min_accept_iou": 0.85}},
                "coconut_benchmark": {"limit": 128},
            }
        ),
        encoding="utf-8",
    )
    patch = tmp_path / "next_config_patch.yaml"
    patch.write_text(
        yaml.safe_dump({"labeling": {"quality_gates": {"min_accept_iou": 0.80}}}),
        encoding="utf-8",
    )
    cfg = load_config(base)
    merged_path = apply_iteration_patch(cfg, patch, out_path=tmp_path / "merged.yaml")
    merged = yaml.safe_load(merged_path.read_text(encoding="utf-8"))
    assert merged["labeling"]["yolo_conf"] == 0.25
    assert merged["labeling"]["quality_gates"]["min_accept_iou"] == 0.80
    assert merged["coconut_benchmark"]["limit"] == 128

"""Tests for benchmark -> pipeline bridge helpers."""

from __future__ import annotations

import yaml

from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config, load_config
from hmp.eval.benchmark_bridge import apply_iteration_patch, import_benchmark_annotations
from hmp.schemas import AnnotationRecord, InstanceAnnotation


def test_filter_annotation_records_by_decision():
    from hmp.eval.benchmark_bridge import filter_annotation_records

    records = [
        AnnotationRecord(
            item_id="img",
            instances=[
                InstanceAnnotation(
                    instance_id="p0",
                    bbox_xyxy=[0, 0, 10, 10],
                    prompt_history=[{"decision": "accept"}],
                ),
                InstanceAnnotation(
                    instance_id="p1",
                    bbox_xyxy=[1, 1, 11, 11],
                    prompt_history=[{"decision": "reject"}],
                ),
            ],
        )
    ]
    kept = filter_annotation_records(records, decisions=("accept",))
    assert len(kept) == 1
    assert len(kept[0].instances) == 1
    assert kept[0].instances[0].instance_id == "p0"


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


def test_bootstrap_from_benchmark(tmp_path):
    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    write_jsonl(
        benchmark_dir / "manifest.jsonl",
        [
            {
                "item_id": "img1",
                "media_type": "image",
                "path": "/tmp/img1.jpg",
                "width": 64,
                "height": 64,
                "sha256": "abc",
            }
        ],
    )
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
                        prompt_history=[{"decision": "accept"}],
                    ),
                    InstanceAnnotation(
                        instance_id="person_1",
                        bbox_xyxy=[1, 1, 11, 11],
                        prompt_history=[{"decision": "reject"}],
                    ),
                ],
            )
        ],
    )
    write_jsonl(
        benchmark_dir / "review_queue.jsonl",
        [
            {
                "item_id": "img1",
                "instance_id": "person_1",
                "image_path": "/tmp/img1.jpg",
                "decision": "reject",
            }
        ],
    )
    cfg = Config(
        {
            "paths": {
                "manifest_path": str(tmp_path / "manifest.jsonl"),
                "annotation_path": str(tmp_path / "annotations.jsonl"),
            },
            "relabel": {"hitl_queue_path": str(tmp_path / "hitl.jsonl")},
            "coconut_bridge": {"benchmark_dir": str(benchmark_dir), "import_decisions": ["accept"]},
        }
    )
    from hmp.eval.benchmark_bridge import bootstrap_from_benchmark

    out = bootstrap_from_benchmark(cfg, project_root=tmp_path)
    anns = read_jsonl_list(out["annotation_path"], model=AnnotationRecord)
    assert len(anns) == 1
    assert len(anns[0].instances) == 1
    hitl = read_jsonl_list(out["hitl_path"])
    assert len(hitl) == 1
    assert hitl[0]["decision"] == "reject"


def test_export_bad_boundary_queue(tmp_path):
    from hmp.eval.benchmark_bridge import export_bad_boundary_queue
    from hmp.schemas import BenchmarkRecord

    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    write_jsonl(
        benchmark_dir / "benchmark_records.jsonl",
        [
            BenchmarkRecord(
                item_id="img1",
                instance_id="person_0",
                image_path="/tmp/img1.jpg",
                gt_mask_path="/tmp/gt.png",
                decision="review",
                error_tags=["bad_boundary", "needs_scribble"],
                mask_iou=0.6,
                boundary_f_score=0.5,
                elapsed_ms=10.0,
            ),
            BenchmarkRecord(
                item_id="img2",
                instance_id="person_0",
                image_path="/tmp/img2.jpg",
                gt_mask_path="/tmp/gt2.png",
                decision="accept",
                error_tags=["background_leak"],
                mask_iou=0.9,
                boundary_f_score=0.88,
                elapsed_ms=10.0,
            ),
        ],
    )
    out = export_bad_boundary_queue(benchmark_dir)
    rows = read_jsonl_list(out)
    assert len(rows) == 1
    assert rows[0]["teacher"] == "samhq"
    assert "bad_boundary" in rows[0]["error_tags"]


def test_relabel_bad_boundary_dry_run(tmp_path):
    from hmp.eval.benchmark_bridge import relabel_bad_boundary_instances
    from hmp.schemas import BenchmarkRecord

    benchmark_dir = tmp_path / "bench"
    benchmark_dir.mkdir()
    write_jsonl(
        benchmark_dir / "benchmark_records.jsonl",
        [
            BenchmarkRecord(
                item_id="img1",
                instance_id="person_0",
                image_path="/tmp/img1.jpg",
                gt_mask_path="/tmp/gt.png",
                decision="review",
                error_tags=["bad_boundary"],
                mask_iou=0.6,
                boundary_f_score=0.5,
                elapsed_ms=10.0,
            ),
        ],
    )
    write_jsonl(
        benchmark_dir / "annotations_pred.jsonl",
        [
            AnnotationRecord(
                item_id="img1",
                instances=[
                    InstanceAnnotation(
                        instance_id="person_0",
                        bbox_xyxy=[0, 0, 10, 10],
                    )
                ],
            )
        ],
    )
    cfg = Config({"labeling": {"segment_teacher": "samhq"}})
    stats = relabel_bad_boundary_instances(cfg, benchmark_dir, dry_run=True)
    assert stats["targets"] == 1
    assert stats["dry_run"] is True

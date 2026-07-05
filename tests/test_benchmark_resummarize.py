"""Tests for benchmark backfill/resummarize/review export."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from hmp.cli import app
from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.data.mask_io import write_binary_mask, write_uint8_image
from hmp.eval.coconut_benchmark import (
    backfill_benchmark_records,
    export_benchmark_review_queue,
    resummarize_benchmark_dir,
    write_benchmark_contact_sheet,
)
from hmp.schemas import BenchmarkRecord

runner = CliRunner()


def test_backfill_legacy_record_gets_decision():
    record = BenchmarkRecord(
        item_id="demo",
        instance_id="person_0",
        image_path="/tmp/a.jpg",
        gt_mask_path="/tmp/gt.png",
        mask_iou=0.92,
        boundary_f_score=0.91,
        elapsed_ms=10.0,
    )
    updated = backfill_benchmark_records([record])[0]
    assert updated.decision == "accept"
    assert updated.improvement_hint
    assert updated.review_priority is not None


def test_resummarize_benchmark_dir(tmp_path):
    records = [
        BenchmarkRecord(
            item_id="demo",
            instance_id="person_0",
            image_path="/tmp/a.jpg",
            gt_mask_path="/tmp/gt.png",
            detector_mode="gt_bbox",
            sam_mode="oracle",
            mask_iou=0.99,
            boundary_f_score=0.99,
            elapsed_ms=5.0,
        )
    ]
    out_dir = tmp_path / "bench"
    out_dir.mkdir()
    write_jsonl(out_dir / "benchmark_records.jsonl", records, overwrite=True)
    summary = resummarize_benchmark_dir(out_dir)
    assert summary["instances"] == 1
    assert summary["decision_counts"]["accept"] == 1
    assert "area_buckets" in summary
    assert "tag_metrics" in summary
    assert "primary_error_buckets" in summary
    assert (out_dir / "benchmark_summary.md").exists()


def test_export_benchmark_review_queue(tmp_path):
    records = [
        BenchmarkRecord(
            item_id="good",
            instance_id="person_0",
            image_path="/tmp/good.jpg",
            gt_mask_path="/tmp/gt.png",
            mask_iou=0.99,
            boundary_f_score=0.99,
            decision="accept",
            elapsed_ms=1.0,
        ),
        BenchmarkRecord(
            item_id="bad",
            instance_id="person_0",
            image_path="/tmp/bad.jpg",
            gt_mask_path="/tmp/gt.png",
            mask_iou=0.20,
            boundary_f_score=0.30,
            decision="reject",
            error_tags=["low_iou"],
            elapsed_ms=1.0,
        ),
        BenchmarkRecord(
            item_id="worse",
            instance_id="person_0",
            image_path="/tmp/worse.jpg",
            gt_mask_path="/tmp/gt.png",
            mask_iou=0.10,
            boundary_f_score=0.10,
            decision="reject",
            error_tags=["detector_miss", "low_iou"],
            elapsed_ms=1.0,
        ),
    ]
    out_dir = tmp_path / "bench"
    out_dir.mkdir()
    write_jsonl(out_dir / "benchmark_records.jsonl", records, overwrite=True)
    review_path = export_benchmark_review_queue(out_dir)
    rows = read_jsonl_list(review_path)
    assert len(rows) == 2
    assert rows[0]["item_id"] == "worse"
    assert rows[0]["review_priority"] >= rows[1]["review_priority"]
    assert rows[0]["primary_error"] == "detector_miss"
    assert "gt_area_bucket" in rows[0]


def test_coconut_resummarize_cli_accepts_config_quality_gates(tmp_path):
    records = [
        BenchmarkRecord(
            item_id="demo",
            instance_id="person_0",
            image_path="/tmp/a.jpg",
            gt_mask_path="/tmp/gt.png",
            detector_mode="gt_bbox",
            sam_mode="oracle",
            mask_iou=0.99,
            boundary_f_score=0.99,
            elapsed_ms=5.0,
        )
    ]
    out_dir = tmp_path / "bench"
    out_dir.mkdir()
    write_jsonl(out_dir / "benchmark_records.jsonl", records, overwrite=True)
    cfg = {"coconut_benchmark": {"quality_gates": {"min_accept_iou": 0.9}, "worst_k": 3}}
    cfg_path = tmp_path / "coconut.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = runner.invoke(
        app,
        ["eval", "coconut-resummarize", "--benchmark-dir", str(out_dir), "--config", str(cfg_path)],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["quality_gates"]["min_accept_iou"] == 0.9


def test_write_benchmark_contact_sheet(tmp_path):
    import cv2
    import numpy as np

    out_dir = tmp_path / "bench"
    out_dir.mkdir()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    image_path = out_dir / "image.jpg"
    cv2.imwrite(str(image_path), image)
    gt = np.zeros((64, 64), dtype=bool)
    gt[12:42, 14:44] = True
    pred = np.zeros((64, 64), dtype=bool)
    pred[16:46, 18:48] = True
    diff = np.zeros((64, 64), dtype=np.uint8)
    diff[gt & pred] = 255
    diff[gt & ~pred] = 85
    diff[~gt & pred] = 170
    gt_path = out_dir / "gt.png"
    pred_path = out_dir / "pred.png"
    diff_path = out_dir / "diff.png"
    write_binary_mask(gt_path, gt)
    write_binary_mask(pred_path, pred)
    write_uint8_image(diff_path, diff)
    record = BenchmarkRecord(
        item_id="img",
        instance_id="person_0",
        image_path=str(image_path),
        gt_mask_path=str(gt_path),
        pred_mask_path=str(pred_path),
        diff_mask_path=str(diff_path),
        mask_iou=0.7,
        boundary_f_score=0.8,
        elapsed_ms=1.0,
    )
    out = write_benchmark_contact_sheet(out_dir, records=[record], max_items=1, tile_width=64)
    assert out.exists()
    sheet = cv2.imread(str(out))
    assert sheet is not None
    assert sheet.shape[1] == 64 * 4

"""Tests for benchmark compare production mode selection."""

from __future__ import annotations

from hmp.eval.benchmark_compare import _apply_production_preference, _mode_score, _select_best_mode


def _row(detector: str, sam: str, iou: float, accept: float) -> dict[str, object]:
    row = {
        "detector_mode": detector,
        "sam_mode": sam,
        "instances": 100,
        "mean_mask_iou": iou,
        "mean_boundary_f1": iou + 0.1,
        "accept_rate": accept,
        "reject_rate": 0.05,
        "mean_elapsed_ms": 400.0,
        "mode_score": 0.0,
    }
    row["mode_score"] = _mode_score(row)
    return row


def test_apply_production_preference_picks_yolo_within_margin():
    gt = _row("gt_bbox", "sam2", 0.79, 0.37)
    yolo = _row("yolo_person", "sam2", 0.77, 0.35)
    rows = sorted([gt, yolo], key=lambda r: float(r["mode_score"]), reverse=True)
    cfg = {"prefer_production_detector": "yolo_person", "production_score_margin": 0.03}
    best = _apply_production_preference(rows, cfg)
    assert best is not None
    assert best["detector_mode"] == "yolo_person"
    assert best["sam_mode"] == "sam2"


def test_select_best_mode_excludes_oracle_by_default():
    oracle = _row("gt_bbox", "oracle", 0.99, 0.99)
    oracle["sam_mode"] = "oracle"
    prod = _row("yolo_person", "sam2", 0.77, 0.35)
    rows = sorted([oracle, prod], key=lambda r: float(r["mode_score"]), reverse=True)
    best, policy = _select_best_mode(rows, {"allow_oracle_selection": False})
    assert best is not None
    assert best["detector_mode"] == "yolo_person"
    assert policy["oracle_modes_excluded"] is True

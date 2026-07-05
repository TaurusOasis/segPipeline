"""Tests for image-segmentation quality_json + train_weight export (V0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmp.data.quality_export import (
    DEFAULT_QUALITY_TIERS,
    QualityExportRow,
    build_export_rows,
    derive_quality_score,
    export_quality_jsonl,
    export_train_weights,
    load_quality_tiers,
    quality_to_tier,
    summarize_tiers,
    tier_train_weight,
)


# ---------------------------------------------------------------------- #
# load_quality_tiers
# ---------------------------------------------------------------------- #
def test_load_quality_tiers_defaults():
    tiers = load_quality_tiers(None)
    assert tiers["gold"]["min_quality"] == 0.90
    assert tiers["silver"]["min_quality"] == 0.75
    assert tiers["reject"]["base_weight"] == 0.0


def test_load_quality_tiers_overrides_from_yaml():
    raw = {
        "gold": {"min_quality": 0.88, "training_use": "high_weight"},
        "silver": {"min_quality": 0.70, "training_use": "normal_weight"},
        "bronze": {"min_quality": 0.50, "training_use": "low_weight_or_distillation_only"},
        "reject": {"max_quality": 0.50, "training_use": "reject_or_human_review"},
    }
    tiers = load_quality_tiers(raw)
    assert tiers["gold"]["min_quality"] == 0.88
    assert tiers["silver"]["base_weight"] == 0.60
    assert tiers["reject"]["max_quality"] == 0.50


# ---------------------------------------------------------------------- #
# quality_to_tier
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "score,expected",
    [
        (0.95, "gold"),
        (0.90, "gold"),
        (0.89, "silver"),
        (0.75, "silver"),
        (0.74, "bronze"),
        (0.55, "bronze"),
        (0.54, "reject"),
        (0.10, "reject"),
    ],
)
def test_quality_to_tier_boundaries(score, expected):
    assert quality_to_tier(score) == expected


def test_quality_to_tier_respects_overridden_reject_max():
    tiers = load_quality_tiers({"reject": {"max_quality": 0.40}})
    assert quality_to_tier(0.45, tiers) == "bronze"
    assert quality_to_tier(0.35, tiers) == "reject"


# ---------------------------------------------------------------------- #
# tier_train_weight
# ---------------------------------------------------------------------- #
def test_tier_train_weight_reject_is_zero():
    assert tier_train_weight("reject", 0.2) == 0.0


def test_tier_train_weight_modulates_by_score():
    w_low = tier_train_weight("silver", 0.75)
    w_high = tier_train_weight("silver", 0.89)
    assert 0.0 < w_low < w_high <= 1.0


def test_tier_train_weight_gold_high_score_near_one():
    w = tier_train_weight("gold", 1.0)
    assert w == pytest.approx(1.0, abs=1e-4)


def test_tier_train_weight_bronze_below_gold():
    assert tier_train_weight("bronze", 0.6) < tier_train_weight("gold", 0.6)


# ---------------------------------------------------------------------- #
# derive_quality_score
# ---------------------------------------------------------------------- #
def test_derive_quality_score_explicit():
    assert derive_quality_score({"quality_score": 0.82}) == 0.82


def test_derive_quality_score_mean_of_dict():
    assert derive_quality_score({"quality_scores": {"iou": 0.8, "boundary": 0.6}}) == pytest.approx(0.7)


def test_derive_quality_score_from_decision():
    assert derive_quality_score({"decision": "accept"}) == 0.9
    assert derive_quality_score({"decision": "review"}) == 0.6
    assert derive_quality_score({"decision": "reject"}) == 0.3


def test_derive_quality_score_none_when_nothing():
    assert derive_quality_score({"item_id": "x"}) is None


# ---------------------------------------------------------------------- #
# build_export_rows
# ---------------------------------------------------------------------- #
def test_build_export_rows_from_dicts():
    records = [
        {"item_id": "i1", "instance_id": "p1", "quality_score": 0.95, "decision": "accept",
         "error_tags": [], "improvement_hint": "ok"},
        {"item_id": "i2", "instance_id": "p2", "quality_score": 0.60, "decision": "review",
         "error_tags": ["bad_boundary"], "improvement_hint": "fix boundary"},
        {"item_id": "i3", "instance_id": "p3", "quality_score": 0.20, "decision": "reject",
         "error_tags": ["empty_prediction"]},
    ]
    rows = build_export_rows(records)
    assert [r.tier for r in rows] == ["gold", "bronze", "reject"]
    assert [r.decision for r in rows] == ["accept", "review", "reject"]
    assert rows[0].train_weight > 0.0
    assert rows[2].train_weight == 0.0
    assert rows[1].error_tags == ["bad_boundary"]
    assert rows[0].improvement_hint == "ok"


def test_build_export_rows_reject_decision_forces_reject_tier_even_if_score_high():
    # A record marked reject but with a leftover high score still lands in reject.
    rows = build_export_rows([{"item_id": "i", "instance_id": "p", "quality_score": 0.9, "decision": "reject"}])
    assert rows[0].tier == "reject"
    assert rows[0].train_weight == 0.0


def test_build_export_rows_skips_records_without_score():
    records = [
        {"item_id": "i1", "instance_id": "p1", "quality_score": 0.9},
        {"item_id": "i2", "instance_id": "p2"},  # no score derivable
    ]
    rows = build_export_rows(records)
    assert len(rows) == 1
    assert rows[0].item_id == "i1"


def test_build_export_rows_skips_missing_ids():
    rows = build_export_rows([{"quality_score": 0.9}])
    assert rows == []


def test_build_export_rows_accepts_pydantic_records():
    from hmp.schemas import BenchmarkRecord

    rec = BenchmarkRecord(
        item_id="i1",
        instance_id="p1",
        image_path="x.png",
        gt_mask_path="g.png",
        mask_iou=0.92,
        boundary_f_score=0.90,
        elapsed_ms=10.0,
        decision="accept",
        error_tags=[],
    )
    rows = build_export_rows([rec])
    assert len(rows) == 1
    # BenchmarkRecord has quality_scores dict (empty) and no quality_score ->
    # falls back to decision -> accept -> 0.9 -> gold.
    assert rows[0].tier == "gold"


# ---------------------------------------------------------------------- #
# export_quality_jsonl + export_train_weights
# ---------------------------------------------------------------------- #
def test_export_quality_jsonl_writes_rows(tmp_path: Path):
    records = [
        {"item_id": "i1", "instance_id": "p1", "quality_score": 0.95, "decision": "accept",
         "error_tags": [], "improvement_hint": "ok"},
        {"item_id": "i2", "instance_id": "p2", "quality_score": 0.30, "decision": "reject",
         "error_tags": ["empty_prediction"]},
    ]
    out = tmp_path / "quality.jsonl"
    n = export_quality_jsonl(records, out)
    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["item_id"] == "i1"
    assert row0["tier"] == "gold"
    assert row0["decision"] == "accept"
    assert row0["train_weight"] > 0.0
    row1 = json.loads(lines[1])
    assert row1["tier"] == "reject"
    assert row1["train_weight"] == 0.0
    assert row1["error_tags"] == ["empty_prediction"]


def test_export_train_weights_slim_rows(tmp_path: Path):
    records = [
        {"item_id": "i1", "instance_id": "p1", "quality_score": 0.80, "decision": "accept"},
        {"item_id": "i2", "instance_id": "p2", "quality_score": 0.40, "decision": "reject"},
    ]
    out = tmp_path / "train_weights.jsonl"
    n = export_train_weights(records, out)
    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    row0 = json.loads(lines[0])
    assert set(row0.keys()) == {"item_id", "instance_id", "tier", "quality_score", "train_weight"}
    assert row0["tier"] == "silver"


def test_export_consistency_between_quality_and_weights(tmp_path: Path):
    records = [
        {"item_id": f"i{k}", "instance_id": f"p{k}", "quality_score": s, "decision": "accept"}
        for k, s in enumerate([0.95, 0.80, 0.60, 0.30])
    ]
    q = tmp_path / "quality.jsonl"
    w = tmp_path / "train_weights.jsonl"
    nq = export_quality_jsonl(records, q)
    nw = export_train_weights(records, w)
    assert nq == nw == 4
    qrows = [json.loads(l) for l in q.read_text().strip().splitlines()]
    wrows = [json.loads(l) for l in w.read_text().strip().splitlines()]
    for qr, wr in zip(qrows, wrows):
        assert qr["tier"] == wr["tier"]
        assert qr["train_weight"] == wr["train_weight"]


# ---------------------------------------------------------------------- #
# summarize_tiers
# ---------------------------------------------------------------------- #
def test_summarize_tiers_counts_all_buckets():
    records = [
        {"item_id": "i1", "instance_id": "p1", "quality_score": 0.95},  # gold
        {"item_id": "i2", "instance_id": "p2", "quality_score": 0.80},  # silver
        {"item_id": "i3", "instance_id": "p3", "quality_score": 0.60},  # bronze
        {"item_id": "i4", "instance_id": "p4", "quality_score": 0.30},  # reject
        {"item_id": "i5", "instance_id": "p5", "quality_score": 0.10},  # reject
    ]
    counts = summarize_tiers(records)
    assert counts["gold"] == 1
    assert counts["silver"] == 1
    assert counts["bronze"] == 1
    assert counts["reject"] == 2
    assert counts["total"] == 5


def test_summarize_tiers_empty():
    counts = summarize_tiers([])
    assert counts["total"] == 0
    assert counts["gold"] == 0


# ---------------------------------------------------------------------- #
# QualityExportRow.to_dict
# ---------------------------------------------------------------------- #
def test_quality_export_row_to_dict():
    row = QualityExportRow(
        item_id="i", instance_id="p", decision="accept", error_tags=["x"],
        quality_score=0.9, tier="gold", train_weight=0.95, improvement_hint="h",
    )
    d = row.to_dict()
    assert d["tier"] == "gold"
    assert d["error_tags"] == ["x"]
    assert d["quality_score"] == 0.9
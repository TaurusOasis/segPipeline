"""Image segmentation export: ``quality_json`` + ``train_weight`` artifacts.

Pipeline step 7 (V0 image-segmentation data engine, ``next_code_priorities``
#3 in ``configs/code_targets.yaml``). Once masks are exported as YOLO seg
labels (``yolo_seg_io.py``) and scored by the shared quality gate
(:func:`hmp.eval.label_quality.decision_and_tags`), this module turns the
per-instance decisions/scores into two artifacts the downstream YOLO student
training consumes:

* ``quality.jsonl`` — one row per instance with ``item_id``, ``instance_id``,
  ``decision``, ``error_tags``, ``quality_score``, ``tier``,
  ``train_weight``, ``improvement_hint``.
* ``train_weights.jsonl`` — slim per-instance
  ``{item_id, instance_id, tier, quality_score, train_weight}`` for the
  loss-weighting / sampling pass.

Tiers and weights follow ``configs/code_targets.yaml`` ``quality_tiers``:

    gold   min_quality 0.90  high_weight
    silver min_quality 0.75  normal_weight
    bronze min_quality 0.55  low_weight_or_distillation_only
    reject max_quality 0.55  reject_or_human_review

CPU-only (stdlib + pydantic + pyyaml), no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..common.jsonl import write_jsonl

__all__ = [
    "DEFAULT_QUALITY_TIERS",
    "TIER_ORDER",
    "QualityExportRow",
    "load_quality_tiers",
    "quality_to_tier",
    "derive_quality_score",
    "tier_train_weight",
    "export_quality_jsonl",
    "export_train_weights",
    "summarize_tiers",
]

TIER_ORDER = ["gold", "silver", "bronze", "reject"]

# Default tier table; mirrors configs/code_targets.yaml quality_tiers.
# ``base_weight`` is the tier's nominal training weight; the per-instance
# weight modulates it by the quality score so a 0.76 silver is downweighted
# relative to a 0.89 silver (see :func:`tier_train_weight`).
DEFAULT_QUALITY_TIERS: dict[str, dict[str, float]] = {
    "gold": {"min_quality": 0.90, "base_weight": 1.00},
    "silver": {"min_quality": 0.75, "base_weight": 0.60},
    "bronze": {"min_quality": 0.55, "base_weight": 0.30},
    "reject": {"max_quality": 0.55, "base_weight": 0.00},
}


def load_quality_tiers(raw: Optional[Mapping[str, Any]] = None) -> dict[str, dict[str, float]]:
    """Build a tier table, optionally overriding defaults from yaml.

    ``raw`` is the ``quality_tiers:`` mapping from ``configs/code_targets.yaml``
    where each tier entry may carry ``min_quality`` (or ``max_quality`` for
    reject) and ``training_use``. ``training_use`` is mapped to a base weight
    when no explicit ``base_weight`` is given.
    """
    tiers = {k: dict(v) for k, v in DEFAULT_QUALITY_TIERS.items()}
    if raw is None:
        return tiers
    use_to_weight = {
        "high_weight": 1.00,
        "normal_weight": 0.60,
        "low_weight_or_distillation_only": 0.30,
        "reject_or_human_review": 0.00,
    }
    for name, entry in raw.items():
        if name not in tiers:
            continue
        e = dict(entry)
        if "min_quality" in e:
            tiers[name]["min_quality"] = float(e["min_quality"])
        if "max_quality" in e:
            tiers[name]["max_quality"] = float(e["max_quality"])
        if "base_weight" in e:
            tiers[name]["base_weight"] = float(e["base_weight"])
        elif "training_use" in e:
            tiers[name]["base_weight"] = use_to_weight.get(
                str(e["training_use"]), tiers[name]["base_weight"]
            )
    return tiers


def quality_to_tier(score: float, tiers: Optional[Mapping[str, Mapping[str, float]]] = None) -> str:
    """Map a 0-1 quality score to a tier name.

    ``reject`` is checked first via its ``max_quality`` (default 0.55), then
    gold / silver / bronze by descending ``min_quality``.
    """
    tiers = tiers or DEFAULT_QUALITY_TIERS
    s = float(score)
    reject_max = float(tiers["reject"].get("max_quality", 0.55))
    if s < reject_max:
        return "reject"
    for tier in ("gold", "silver", "bronze"):
        if s >= float(tiers[tier].get("min_quality", 0.0)):
            return tier
    # Score is above the reject cutoff but below bronze_min (a config gap);
    # treat it as the lowest non-reject tier rather than rejecting.
    return "bronze"


def derive_quality_score(record: Mapping[str, Any]) -> Optional[float]:
    """Best-effort scalar quality score from a heterogenous record.

    Order: explicit ``quality_score`` → mean of ``quality_scores`` dict →
    derived from ``decision`` (accept=0.9, review=0.6, reject=0.3) → None.
    """
    if "quality_score" in record and record["quality_score"] is not None:
        return float(record["quality_score"])
    qs = record.get("quality_scores")
    if isinstance(qs, Mapping) and qs:
        vals = [float(v) for v in qs.values() if isinstance(v, (int, float)) and v >= 0]
        if vals:
            return sum(vals) / len(vals)
    decision = record.get("decision")
    if decision == "accept":
        return 0.9
    if decision == "review":
        return 0.6
    if decision == "reject":
        return 0.3
    return None


def tier_train_weight(
    tier: str,
    score: float,
    tiers: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    """Per-instance training weight.

    ``reject`` → 0.0. Otherwise ``base_weight`` modulated by the score within
    the tier's usable range, so a borderline silver (0.75) is downweighted
    relative to a strong silver (0.89). The result is clamped to ``[0, 1]``
    and rounded to 4 decimals.
    """
    tiers = tiers or DEFAULT_QUALITY_TIERS
    if tier == "reject":
        return 0.0
    base = float(tiers.get(tier, {}).get("base_weight", 0.0))
    if base <= 0.0:
        return 0.0
    s = max(0.0, min(1.0, float(score)))
    # Modulate within [0.5, 1.0] of base so the floor is half the tier weight.
    weight = base * (0.5 + 0.5 * s)
    return round(max(0.0, min(1.0, weight)), 4)


@dataclass
class QualityExportRow:
    """One per-instance row written to ``quality.jsonl``."""

    item_id: str
    instance_id: str
    decision: str
    error_tags: list[str]
    quality_score: float
    tier: str
    train_weight: float
    improvement_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "instance_id": self.instance_id,
            "decision": self.decision,
            "error_tags": list(self.error_tags),
            "quality_score": round(float(self.quality_score), 4),
            "tier": self.tier,
            "train_weight": self.train_weight,
            "improvement_hint": self.improvement_hint,
        }


def _coerce_record(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")  # type: ignore[union-attr]
    if hasattr(record, "__dict__"):
        return record.__dict__
    raise TypeError(f"unsupported record type: {type(record).__name__}")


def build_export_rows(
    records: Iterable[Any],
    *,
    tiers: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> list[QualityExportRow]:
    """Build :class:`QualityExportRow` list from input records.

    Each record must expose ``item_id`` and ``instance_id`` and at least one
    of ``quality_score`` / ``quality_scores`` / ``decision``. Records with no
    derivable score are skipped (logged via the returned rows — caller can
    detect gaps by comparing input length).
    """
    tier_table = tiers or DEFAULT_QUALITY_TIERS
    rows: list[QualityExportRow] = []
    for rec in records:
        r = _coerce_record(rec)
        item_id = r.get("item_id")
        instance_id = r.get("instance_id")
        if item_id is None or instance_id is None:
            continue
        score = derive_quality_score(r)
        if score is None:
            continue
        decision = str(r.get("decision") or ("reject" if score < 0.55 else "accept"))
        # An explicit reject decision always forces the reject tier.
        tier = "reject" if decision == "reject" else quality_to_tier(score, tier_table)
        weight = tier_train_weight(tier, score, tier_table)
        tags = list(r.get("error_tags") or [])
        hint = str(r.get("improvement_hint") or "")
        rows.append(
            QualityExportRow(
                item_id=str(item_id),
                instance_id=str(instance_id),
                decision=decision,
                error_tags=tags,
                quality_score=float(score),
                tier=tier,
                train_weight=weight,
                improvement_hint=hint,
            )
        )
    return rows


def export_quality_jsonl(
    records: Iterable[Any],
    out_path: str | Path,
    *,
    tiers: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> int:
    """Write ``quality.jsonl`` and return the number of rows written."""
    rows = build_export_rows(records, tiers=tiers)
    write_jsonl(out_path, [row.to_dict() for row in rows])
    return len(rows)


def export_train_weights(
    records: Iterable[Any],
    out_path: str | Path,
    *,
    tiers: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> int:
    """Write ``train_weights.jsonl`` (slim per-instance weight rows)."""
    rows = build_export_rows(records, tiers=tiers)
    slim = [
        {
            "item_id": row.item_id,
            "instance_id": row.instance_id,
            "tier": row.tier,
            "quality_score": round(row.quality_score, 4),
            "train_weight": row.train_weight,
        }
        for row in rows
    ]
    write_jsonl(out_path, slim)
    return len(slim)


def summarize_tiers(
    records: Iterable[Any],
    *,
    tiers: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> dict[str, int]:
    """Count rows per tier (including ``reject``). Returns all four keys."""
    rows = build_export_rows(records, tiers=tiers)
    counts = {t: 0 for t in TIER_ORDER}
    for row in rows:
        counts[row.tier] = counts.get(row.tier, 0) + 1
    counts["total"] = len(rows)  # type: ignore[assignment]
    return counts
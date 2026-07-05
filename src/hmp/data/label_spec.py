"""Helpers for label-spec, class-map, and QA-tier configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a plain dictionary."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def load_class_map(path: str | Path = "configs/class_map.yaml") -> dict[str, Any]:
    return load_yaml_config(path)


def load_qa_schema(path: str | Path = "configs/qa_schema.yaml") -> dict[str, Any]:
    return load_yaml_config(path)


def active_yolo_classes(class_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active YOLO classes sorted by class id, excluding non-YOLO layers."""
    active_name = class_map.get("export_defaults", {}).get("active_class_set")
    class_sets = class_map.get("class_sets", {})
    if active_name not in class_sets:
        raise ValueError(f"active_class_set {active_name!r} not found in class_sets")
    semantic_layers = class_map.get("semantic_layers", {})
    rows = list(class_sets[active_name].get("classes", []))
    out: list[dict[str, Any]] = []
    for row in rows:
        layer_name = row.get("semantic_layer")
        layer = semantic_layers.get(layer_name)
        if layer is None:
            raise ValueError(f"class {row.get('name')!r} references missing semantic_layer {layer_name!r}")
        if layer.get("target_type") == "soft_alpha" or not bool(layer.get("yolo_training", False)):
            continue
        out.append(row)
    return sorted(out, key=lambda r: int(r["class_id"]))


def active_yolo_class_names(class_map: dict[str, Any]) -> list[str]:
    """Return Ultralytics class names for the active YOLO class set."""
    return [str(row.get("export_name") or row["name"]) for row in active_yolo_classes(class_map)]


def compute_final_quality(scores: dict[str, float], qa_schema: dict[str, Any]) -> float:
    """Compute weighted final quality from a score dictionary.

    Weights ending in ``_inverse`` use ``1 - scores[base_name]`` so risk fields
    such as ``overlap_conflict`` can contribute positively when low.
    """
    weights = qa_schema.get("final_quality_formula", {}).get("weights", {})
    total = 0.0
    for name, weight in weights.items():
        key = str(name)
        inverse = key.endswith("_inverse")
        score_key = key.removesuffix("_inverse")
        raw = float(scores.get(score_key, 0.0))
        value = 1.0 - raw if inverse else raw
        total += float(weight) * value
    lo, hi = qa_schema.get("final_quality_formula", {}).get("clamp", [0.0, 1.0])
    return round(max(float(lo), min(float(hi), total)), 6)


def quality_tier(final_quality: float, qa_schema: dict[str, Any]) -> str:
    """Map a final quality score to gold/silver/bronze/reject."""
    tiers = qa_schema.get("quality_tiers", {})
    score = float(final_quality)
    ordered = sorted(
        (
            (name, float(spec.get("min_final_quality", -1.0)))
            for name, spec in tiers.items()
            if "min_final_quality" in spec
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, threshold in ordered:
        if score >= threshold:
            return str(name)
    return "reject"


def train_weight_for_quality(final_quality: float, qa_schema: dict[str, Any]) -> float:
    tier = quality_tier(final_quality, qa_schema)
    return float(qa_schema.get("quality_tiers", {}).get(tier, {}).get("train_weight", 0.0))


def decision_for_quality(final_quality: float, qa_schema: dict[str, Any]) -> str:
    tier = quality_tier(final_quality, qa_schema)
    return str(qa_schema.get("quality_tiers", {}).get(tier, {}).get("decision", "reject"))

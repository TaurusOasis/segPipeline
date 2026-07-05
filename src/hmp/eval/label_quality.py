"""Shared label quality gates for benchmark (with GT) and production (proxy)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..data.mask_io import mask_area_ratio

Decision = Literal["accept", "review", "reject"]

DEFAULT_QUALITY_GATES: dict[str, float] = {
    "min_accept_iou": 0.85,
    "min_accept_boundary_f1": 0.85,
    "min_review_iou": 0.50,
    "min_review_boundary_f1": 0.65,
    "max_accept_false_positive_ratio": 0.25,
    "max_accept_false_negative_ratio": 0.25,
}


def quality_gates_from_config(raw: object | None = None) -> dict[str, float]:
    gates = dict(DEFAULT_QUALITY_GATES)
    if raw is None:
        return gates
    if hasattr(raw, "get"):
        for key in DEFAULT_QUALITY_GATES:
            if key in raw:
                gates[key] = float(raw.get(key))
    elif isinstance(raw, dict):
        for key in DEFAULT_QUALITY_GATES:
            if key in raw:
                gates[key] = float(raw[key])
    return gates


def mask_error_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_b = np.asarray(pred) > 0
    gt_b = np.asarray(gt) > 0
    fp = pred_b & ~gt_b
    fn = gt_b & ~pred_b
    gt_area = int(gt_b.sum())
    pred_area = int(pred_b.sum())
    return {
        "gt_area_ratio": mask_area_ratio(gt_b),
        "pred_area_ratio": mask_area_ratio(pred_b),
        "false_positive_ratio": float(fp.sum()) / float(max(pred_area, 1)),
        "false_negative_ratio": float(fn.sum()) / float(max(gt_area, 1)),
    }


def decision_and_tags(
    *,
    iou: float | None,
    boundary: float | None,
    stats: dict[str, float],
    gates: dict[str, float],
    prompt_needs_scribble: bool,
    detector_meta: dict[str, float] | None = None,
    multi_person: bool = False,
    pred_empty: bool = False,
    prompt_confidence: float | None = None,
) -> tuple[Decision, list[str], str]:
    """Return accept/review/reject, error tags, and improvement hint."""
    detector_meta = detector_meta or {}
    tags: list[str] = []
    if pred_empty:
        tags.append("empty_prediction")
    if iou is not None and iou < gates["min_review_iou"]:
        tags.append("low_iou")
    if boundary is not None and boundary < gates["min_review_boundary_f1"]:
        tags.append("bad_boundary")
    if stats.get("false_negative_ratio", 0.0) > gates["max_accept_false_negative_ratio"]:
        tags.append("missed_foreground")
    if stats.get("false_positive_ratio", 0.0) > gates["max_accept_false_positive_ratio"]:
        tags.append("background_leak")
    if stats.get("gt_area_ratio", 0.0) and stats["gt_area_ratio"] < 0.01:
        tags.append("small_person")
    if stats.get("gt_area_ratio", 0.0) and stats["gt_area_ratio"] > 0.40:
        tags.append("large_person")
    if prompt_needs_scribble:
        tags.append("needs_scribble")
    if multi_person:
        tags.append("multi_person")
    if detector_meta.get("det_matched") == 0.0:
        tags.append("detector_miss")

    has_gt_metrics = iou is not None and boundary is not None
    if has_gt_metrics:
        accept = (
            iou >= gates["min_accept_iou"]
            and boundary >= gates["min_accept_boundary_f1"]
            and stats.get("false_positive_ratio", 0.0) <= gates["max_accept_false_positive_ratio"]
            and stats.get("false_negative_ratio", 0.0) <= gates["max_accept_false_negative_ratio"]
        )
        reject = pred_empty or iou < gates["min_review_iou"] or boundary < gates["min_review_boundary_f1"]
    else:
        conf = prompt_confidence if prompt_confidence is not None else 0.5
        accept = conf >= 0.75 and not prompt_needs_scribble and "detector_miss" not in tags
        reject = pred_empty or conf < 0.35

    decision: Decision = "accept" if accept else ("reject" if reject else "review")
    hint = _improvement_hint(decision, tags)
    return decision, tags, hint


def _improvement_hint(decision: Decision, tags: list[str]) -> str:
    if "detector_miss" in tags:
        return "improve detector / lower yolo_conf / add GroundingDINO fallback"
    if "background_leak" in tags and "multi_person" in tags:
        return "add negative prompts for nearby people and enable identity-aware masklet QA"
    if "background_leak" in tags:
        return "add negative prompts, tighten bbox/ROI, and try keep-largest postprocess"
    if "missed_foreground" in tags or "needs_scribble" in tags:
        return "add positive point or scribble on missed foreground; rerun SAM2 correction"
    if "bad_boundary" in tags:
        return "route boundary ROI to HQ-SAM / Bd diffusion refine"
    if decision == "accept":
        return "accept as segmentation-core sample"
    return "review sample and collect correction history for prompt agent"


def parse_decision_from_prompt_history(prompt_history: list[dict[str, object]]) -> Decision | None:
    for entry in reversed(prompt_history):
        raw = entry.get("decision")
        if raw in {"accept", "review", "reject"}:
            return raw  # type: ignore[return-value]
    return None

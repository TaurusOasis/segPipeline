"""Heuristic / mock RL Prompt Agent (pipeline step 3).

Policy ``heuristic_v2`` adds over v1:
* **Multi-person negative points** — when other person bboxes are present in
  the same frame, drop a negative point on each neighbor's center so SAM2 does
  not leak across identities (the dominant COCONut failure bucket is
  ``multi_person``).
* **Boundary-aware scribble gate** — request a scribble not only on extreme
  area ratios but also when a prior boundary F1 is below threshold, closing the
  3 -> 4 -> 8 -> 3 repair loop.
* **Keyframe selection** — :func:`select_keyframe` picks the clearest frame
  (lowest blur / motion score) to anchor a masklet prompt on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

PromptType = Literal["box", "positive_point", "negative_point", "mask", "scribble"]

DEFAULT_BOUNDARY_SCRIBBBLE_THRESHOLD = 0.7


@dataclass(frozen=True)
class PromptDecision:
    keyframe_index: int
    prompts: tuple[dict[str, object], ...]
    needs_scribble: bool
    confidence: float
    policy: str = "heuristic_v2"
    error_tags: tuple[str, ...] = field(default_factory=tuple)


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _bbox_center(bbox_xyxy: Sequence[int], width: int, height: int) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = _clip((x1 + x2) // 2, 0, width - 1)
    cy = _clip((y1 + y2) // 2, 0, height - 1)
    return cx, cy


def plan_prompts(
    *,
    bbox_xyxy: list[int],
    width: int,
    height: int,
    frame_index: int = 0,
    gt_mask: Optional[np.ndarray] = None,
    neighbor_bboxes: Optional[Sequence[Sequence[int]]] = None,
    boundary_f1: Optional[float] = None,
    boundary_scribble_threshold: float = DEFAULT_BOUNDARY_SCRIBBBLE_THRESHOLD,
) -> PromptDecision:
    """Choose box + point prompts from a person bbox (CPU heuristic agent v2).

    Parameters
    ----------
    neighbor_bboxes:
        Other person bboxes (xyxy) in the same frame. A negative point is
        placed on each neighbor's center to suppress identity leakage in
        multi-person scenes.
    boundary_f1:
        Optional prior boundary F1 for this instance. Below
        ``boundary_scribble_threshold`` it forces ``needs_scribble`` so the
        repair loop can re-prompt with a scribble.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = _bbox_center(bbox_xyxy, width, height)
    prompts: list[dict[str, object]] = [
        {"type": "box", "bbox_xyxy": bbox_xyxy, "frame_index": frame_index},
        {"type": "positive_point", "xy": [cx, cy], "frame_index": frame_index},
    ]

    # Background negative point just outside the box (v1 behavior).
    neg_x = _clip(x1 - max(4, (x2 - x1) // 8), 0, width - 1)
    neg_y = _clip(y1 - max(4, (y2 - y1) // 8), 0, height - 1)
    prompts.append({"type": "negative_point", "xy": [neg_x, neg_y], "frame_index": frame_index})

    error_tags: list[str] = []
    needs_scribble = False
    confidence = 0.75

    if gt_mask is not None and gt_mask.any():
        area_ratio = float(gt_mask.sum()) / float(max(1, width * height))
        if area_ratio < 0.01 or area_ratio > 0.65:
            needs_scribble = True
            confidence = 0.45
            if area_ratio < 0.01:
                error_tags.append("small_person")
            else:
                error_tags.append("large_person")

    # Multi-person: one negative point per neighbor center.
    if neighbor_bboxes:
        for nb in neighbor_bboxes:
            nx, ny = _bbox_center(nb, width, height)
            prompts.append(
                {"type": "negative_point", "xy": [nx, ny], "frame_index": frame_index, "reason": "neighbor"}
            )
        # More neighbors -> lower confidence that a box+point prompt suffices.
        confidence = min(confidence, max(0.4, 0.75 - 0.05 * len(neighbor_bboxes)))
        error_tags.append("multi_person")

    # Boundary-feedback scribble gate (closes the 3 -> 4 -> 8 -> 3 loop).
    if boundary_f1 is not None and boundary_f1 < boundary_scribble_threshold:
        needs_scribble = True
        confidence = min(confidence, 0.4)
        error_tags.append("bad_boundary")

    return PromptDecision(
        keyframe_index=frame_index,
        prompts=tuple(prompts),
        needs_scribble=needs_scribble,
        confidence=confidence,
        policy="heuristic_v2",
        error_tags=tuple(error_tags),
    )


def select_keyframe(
    *,
    frame_indices: Sequence[int],
    scores: Sequence[float],
) -> int:
    """Pick the keyframe with the lowest blur/motion score (clearest frame).

    ``scores`` are per-frame quality penalties (higher = worse, e.g. blur or
    motion magnitude). Returns the frame index with the minimum penalty. Ties
    break toward the earliest frame. Returns 0 when empty.
    """
    if not frame_indices:
        return 0
    if len(frame_indices) != len(scores):
        raise ValueError(
            f"frame_indices and scores length mismatch: {len(frame_indices)} vs {len(scores)}"
        )
    best_idx = int(np.argmin(np.asarray(scores, dtype=np.float64)))
    return int(frame_indices[best_idx])
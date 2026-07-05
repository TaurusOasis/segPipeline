"""Heuristic / mock RL Prompt Agent (pipeline step 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

PromptType = Literal["box", "positive_point", "negative_point", "mask", "scribble"]


@dataclass(frozen=True)
class PromptDecision:
    keyframe_index: int
    prompts: tuple[dict[str, object], ...]
    needs_scribble: bool
    confidence: float
    policy: str = "heuristic_v1"


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def plan_prompts(
    *,
    bbox_xyxy: list[int],
    width: int,
    height: int,
    frame_index: int = 0,
    gt_mask: Optional[np.ndarray] = None,
) -> PromptDecision:
    """Choose box + point prompts from a person bbox (CPU heuristic agent)."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = _clip((x1 + x2) // 2, 0, width - 1)
    cy = _clip((y1 + y2) // 2, 0, height - 1)
    prompts: list[dict[str, object]] = [
        {"type": "box", "bbox_xyxy": bbox_xyxy, "frame_index": frame_index},
        {"type": "positive_point", "xy": [cx, cy], "frame_index": frame_index},
    ]

    neg_x = _clip(x1 - max(4, (x2 - x1) // 8), 0, width - 1)
    neg_y = _clip(y1 - max(4, (y2 - y1) // 8), 0, height - 1)
    prompts.append({"type": "negative_point", "xy": [neg_x, neg_y], "frame_index": frame_index})

    needs_scribble = False
    confidence = 0.75
    if gt_mask is not None and gt_mask.any():
        area_ratio = float(gt_mask.sum()) / float(max(1, width * height))
        if area_ratio < 0.01 or area_ratio > 0.65:
            needs_scribble = True
            confidence = 0.45
    return PromptDecision(
        keyframe_index=frame_index,
        prompts=tuple(prompts),
        needs_scribble=needs_scribble,
        confidence=confidence,
    )

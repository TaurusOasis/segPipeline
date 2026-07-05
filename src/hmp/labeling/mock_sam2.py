"""Mock SAM2 / VOS segmentation from prompts (pipeline step 4, CPU path)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..agents.prompt_agent import PromptDecision


def _bbox_xywh(bbox_xyxy: list[int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def segment_with_prompts(
    image_bgr: np.ndarray,
    decision: PromptDecision,
    *,
    gt_mask: Optional[np.ndarray] = None,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Segment a person instance using GrabCut + point prompts.

    When ``gt_mask`` is supplied (benchmark/debug), optional ``noise_level`` injects
    boundary errors to simulate imperfect SAM2 output.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    if gt_mask is not None and noise_level <= 0:
        return gt_mask.astype(bool)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    box_prompt = next((p for p in decision.prompts if p["type"] == "box"), None)
    if box_prompt is None:
        return np.zeros((h, w), dtype=bool)
    rect = _bbox_xywh(list(box_prompt["bbox_xyxy"]))  # type: ignore[arg-type]
    cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    point_mask = mask.copy()
    for prompt in decision.prompts:
        if prompt["type"] == "positive_point":
            x, y = prompt["xy"]  # type: ignore[index]
            cv2.circle(point_mask, (int(x), int(y)), 4, cv2.GC_FGD, -1)
        elif prompt["type"] == "negative_point":
            x, y = prompt["xy"]  # type: ignore[index]
            cv2.circle(point_mask, (int(x), int(y)), 4, cv2.GC_BGD, -1)
    cv2.grabCut(image_bgr, point_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)

    seg = np.isin(point_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))

    if gt_mask is not None and noise_level > 0:
        import cv2

        k = max(1, int(round(noise_level * 8)))
        kernel = np.ones((3, 3), np.uint8)
        noisy = gt_mask.astype(np.uint8)
        if np.random.rand() < 0.5:
            noisy = cv2.dilate(noisy, kernel, iterations=k)
        else:
            noisy = cv2.erode(noisy, kernel, iterations=k)
        seg = noisy.astype(bool)
    return seg

"""Ultralytics SAM2 adapter for prompt-driven segmentation (pipeline step 4)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..agents.prompt_agent import PromptDecision


def _prompts_to_sam_kwargs(decision: PromptDecision) -> dict[str, object]:
    bboxes: list[list[int]] = []
    points: list[list[int]] = []
    labels: list[int] = []
    for prompt in decision.prompts:
        ptype = prompt["type"]
        if ptype == "box":
            bboxes.append(list(prompt["bbox_xyxy"]))  # type: ignore[arg-type]
        elif ptype == "positive_point":
            x, y = prompt["xy"]  # type: ignore[index]
            points.append([int(x), int(y)])
            labels.append(1)
        elif ptype == "negative_point":
            x, y = prompt["xy"]  # type: ignore[index]
            points.append([int(x), int(y)])
            labels.append(0)
    # Ultralytics stateless SAM.predict accepts either bbox *or* point prompts
    # per call, not both. When a box is present we prefer it (more reliable for
    # person localization), which means neighbor negative points are *not*
    # consumed here yet. Feeding box + negative points together to suppress
    # multi-person identity leakage requires the stateful SAM2 predictor
    # (set_image + add_new_points_or_box) — see A2 masklet work. The multi-person
    # signal still flows through prompt.error_tags / needs_scribble / confidence
    # into decision_and_tags for review routing.
    if bboxes:
        return {"bboxes": bboxes[0] if len(bboxes) == 1 else bboxes}
    if points:
        return {"points": points, "labels": labels}
    return {}


def _mask_from_sam_result(result, height: int, width: int) -> np.ndarray:
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((height, width), dtype=bool)
    mask = result.masks.data[0].cpu().numpy()
    if mask.shape != (height, width):
        import cv2

        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return mask > 0.5


def segment_with_sam2(
    image_bgr: np.ndarray,
    decision: PromptDecision,
    *,
    weights: str = "sam2_b.pt",
    device: str | int = 0,
    fallback_grabcut: bool = True,
    gt_mask: Optional[np.ndarray] = None,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Segment with Ultralytics SAM2; optionally fall back to GrabCut mock."""
    if gt_mask is not None and noise_level <= 0:
        return gt_mask.astype(bool)

    try:
        from ultralytics import SAM
    except ImportError:
        if not fallback_grabcut:
            raise
        from .mock_sam2 import segment_with_prompts

        return segment_with_prompts(
            image_bgr,
            decision,
            gt_mask=gt_mask,
            noise_level=noise_level,
        )

    h, w = image_bgr.shape[:2]
    sam_kwargs = _prompts_to_sam_kwargs(decision)
    if not sam_kwargs:
        return np.zeros((h, w), dtype=bool)

    model = SAM(weights)
    results = model.predict(
        source=image_bgr,
        device=device,
        verbose=False,
        **sam_kwargs,
    )
    if not results:
        seg = np.zeros((h, w), dtype=bool)
    else:
        seg = _mask_from_sam_result(results[0], h, w)

    if gt_mask is not None and noise_level > 0:
        from .mock_sam2 import segment_with_prompts

        return segment_with_prompts(
            image_bgr,
            decision,
            gt_mask=gt_mask,
            noise_level=noise_level,
        )
    return seg

"""GPU segment teachers (SAM2 / SamHQ) for auto-label — not edge deploy models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..agents.prompt_agent import PromptDecision
from ..models.tiers import TeacherModelSpec
from .mock_sam2 import segment_with_prompts
from .sam2_adapter import segment_with_sam2

TeacherBackend = Literal["grabcut", "sam2", "samhq"]


def segment_with_teacher(
    image_bgr: np.ndarray,
    decision: PromptDecision,
    *,
    teacher: TeacherModelSpec | None = None,
    backend: TeacherBackend = "sam2",
    weights: str = "sam2_b.pt",
    device: str | int = 0,
    fallback_grabcut: bool = True,
    gt_mask: Optional[np.ndarray] = None,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Run a GPU segment teacher (SAM2/SamHQ) or CPU GrabCut ablation."""
    resolved_backend: TeacherBackend = backend
    resolved_weights = weights
    if teacher is not None:
        resolved_backend = teacher.backend  # type: ignore[assignment]
        if teacher.weights:
            resolved_weights = teacher.weights
        device = teacher.device

    if resolved_backend == "grabcut":
        return segment_with_prompts(image_bgr, decision, gt_mask=gt_mask, noise_level=noise_level)

    if resolved_backend in {"sam2", "samhq"}:
        if teacher is not None and teacher.command_template:
            return _segment_external_command(image_bgr, decision, teacher=teacher)
        return segment_with_sam2(
            image_bgr,
            decision,
            weights=resolved_weights,
            device=device,
            fallback_grabcut=fallback_grabcut,
            gt_mask=gt_mask,
            noise_level=noise_level,
        )

    if fallback_grabcut:
        return segment_with_prompts(image_bgr, decision, gt_mask=gt_mask, noise_level=noise_level)
    raise ValueError(f"unsupported teacher backend: {resolved_backend!r}")


def teacher_segment_source(backend: TeacherBackend) -> str:
    if backend == "grabcut":
        return "mock_grabcut"
    if backend == "samhq":
        return "samhq"
    return "sam2"


def _segment_external_command(
    image_bgr: np.ndarray,
    decision: PromptDecision,
    *,
    teacher: TeacherModelSpec,
) -> np.ndarray:
    """Placeholder for external SamHQ / custom teacher subprocess adapters."""
    from ..common.subprocess_utils import render_command, run_command

    h, w = image_bgr.shape[:2]
    bbox = decision.bbox_xyxy
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.png"
        mask_path = tmp_path / "mask.png"
        import cv2

        cv2.imwrite(str(image_path), image_bgr)
        cmd = render_command(
            teacher.command_template,
            {
                "image_path": str(image_path),
                "bbox_xyxy": ",".join(str(v) for v in bbox),
                "mask_path": str(mask_path),
                "weights": teacher.weights,
            },
        )
        run_command(cmd, dry_run=False)
        if not mask_path.exists():
            return np.zeros((h, w), dtype=bool)
        from ..data.mask_io import read_binary_mask

        return read_binary_mask(mask_path)

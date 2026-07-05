"""Select labeler implementation for pipeline step 2.

Edge detector is always yolo26s-seg (or configured edge weights).
Segment teacher is a separate GPU model: SAM2, SamHQ, or GrabCut ablation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import Config
from .base import Labeler
from .dummy_labeler import DummyLabeler
from .yolo_sam2_labeler import YoloSam2Labeler

ProviderName = Literal["mock", "yolo_grabcut", "yolo_sam2", "yolo_samhq"]

_PROVIDER_TEACHER: dict[str, str] = {
    "yolo_grabcut": "grabcut",
    "yolo_sam2": "sam2",
    "yolo_samhq": "samhq",
}


def make_labeler(cfg: Config, *, project_root: Path, provider: str = "mock") -> Labeler:
    if provider == "mock":
        return DummyLabeler(cfg, project_root=project_root)
    teacher_key = _PROVIDER_TEACHER.get(provider)
    if teacher_key:
        return YoloSam2Labeler(
            cfg,
            project_root=project_root,
            segment_mode=teacher_key,  # type: ignore[arg-type]
            teacher_key=teacher_key,
        )
    raise ValueError(
        f"unknown labeling provider: {provider!r}; use mock | yolo_grabcut | yolo_sam2 | yolo_samhq"
    )

"""Model tier registry: edge deploy vs GPU teachers."""

from .tiers import (
    EDGE_YOLO26S_SEG,
    ModelTierRegistry,
    TeacherKind,
    load_model_tiers,
    resolve_teacher,
)

__all__ = [
    "EDGE_YOLO26S_SEG",
    "ModelTierRegistry",
    "TeacherKind",
    "load_model_tiers",
    "resolve_teacher",
]

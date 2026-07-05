"""Edge deploy model vs GPU teacher model registry.

Architecture:
- **Edge (deploy)**: yolo26s-seg — RK3576 / mobile inference target.
- **Teachers (GPU)**: SAM2, SamHQ, matting/distill models — auto-label, clean, distill only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

from ..config import Config, resolve_path

TeacherKind = Literal["segment", "boundary", "matting", "distill"]
TeacherBackend = Literal["grabcut", "sam2", "samhq", "external"]

EDGE_YOLO26S_SEG = "/home/genesis/Train/Code/ultralytics/yolo26s-seg.pt"
DEFAULT_DISTILL_TEACHER = "/home/genesis/Train/Code/ultralytics/yolo26x-seg.pt"


@dataclass(frozen=True)
class EdgeModelSpec:
    name: str
    role: str
    weights: str
    deploy_target: str = "rk3576"


@dataclass(frozen=True)
class TeacherModelSpec:
    name: str
    kind: TeacherKind
    backend: TeacherBackend
    weights: str = ""
    device: str | int = 0
    command_template: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ModelTierRegistry:
    edge: EdgeModelSpec
    teachers: dict[str, TeacherModelSpec] = field(default_factory=dict)

    def segment_teachers(self) -> dict[str, TeacherModelSpec]:
        return {k: v for k, v in self.teachers.items() if v.kind == "segment"}

    def default_segment_teacher(self) -> TeacherModelSpec:
        for key in ("sam2", "samhq"):
            if key in self.teachers:
                return self.teachers[key]
        if self.teachers:
            return next(iter(self.teachers.values()))
        return TeacherModelSpec(name="sam2", kind="segment", backend="sam2", weights="sam2_b.pt")


def _default_registry() -> ModelTierRegistry:
    return ModelTierRegistry(
        edge=EdgeModelSpec(
            name="yolo26s-seg",
            role="edge_segment_student",
            weights=EDGE_YOLO26S_SEG,
            deploy_target="rk3576",
        ),
        teachers={
            "sam2": TeacherModelSpec(
                name="sam2_b",
                kind="segment",
                backend="sam2",
                weights="sam2_b.pt",
                notes="Ultralytics SAM2; default auto-label teacher",
            ),
            "samhq": TeacherModelSpec(
                name="sam_hq_vit_b",
                kind="segment",
                backend="samhq",
                weights="sam_hq_vit_b.pt",
                notes="SamHQ weights via Ultralytics SAM API or external adapter",
            ),
            "grabcut": TeacherModelSpec(
                name="grabcut_cpu",
                kind="segment",
                backend="grabcut",
                notes="CPU ablation only; not for production labels",
            ),
            "yolo26x-seg": TeacherModelSpec(
                name="yolo26x-seg",
                kind="distill",
                backend="external",
                weights=DEFAULT_DISTILL_TEACHER,
                notes="Ultralytics distillation teacher; trained in ultralytics/ repo",
            ),
        },
    )


def _cfg_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return {}


def load_model_tiers(cfg: Config, *, project_root: Optional[Path] = None) -> ModelTierRegistry:
    """Load tier registry from config ``models`` block and/or ``configs/models.yaml``."""
    root = Path(project_root) if project_root else Path.cwd()
    base = _default_registry()
    raw = _cfg_mapping(cfg.get("models"))
    if not raw:
        registry_path = resolve_path(root, "configs/models.yaml")
        if registry_path.exists():
            raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not raw:
        return base

    edge_raw = _cfg_mapping(raw.get("edge"))
    edge = EdgeModelSpec(
        name=str(edge_raw.get("name", base.edge.name)),
        role=str(edge_raw.get("role", base.edge.role)),
        weights=str(edge_raw.get("weights", base.edge.weights)),
        deploy_target=str(edge_raw.get("deploy_target", base.edge.deploy_target)),
    )
    teachers: dict[str, TeacherModelSpec] = {}
    for key, spec in _cfg_mapping(raw.get("teachers")).items():
        if not isinstance(spec, dict):
            continue
        teachers[str(key)] = TeacherModelSpec(
            name=str(spec.get("name", key)),
            kind=str(spec.get("kind", "segment")),  # type: ignore[arg-type]
            backend=str(spec.get("backend", "sam2")),  # type: ignore[arg-type]
            weights=str(spec.get("weights", "")),
            device=spec.get("device", 0),
            command_template=str(spec.get("command_template", "")),
            notes=str(spec.get("notes", "")),
        )
    if not teachers:
        teachers = dict(base.teachers)
    return ModelTierRegistry(edge=edge, teachers=teachers)


def resolve_teacher(
    registry: ModelTierRegistry,
    *,
    teacher_key: str | None = None,
    segment_mode: str | None = None,
) -> TeacherModelSpec:
    """Resolve segment teacher from explicit key or legacy segment_mode string."""
    if teacher_key and teacher_key in registry.teachers:
        return registry.teachers[teacher_key]
    mode = segment_mode or "sam2"
    if mode in registry.teachers:
        return registry.teachers[mode]
    if mode == "grabcut" and "grabcut" in registry.teachers:
        return registry.teachers["grabcut"]
    if mode in {"sam2", "samhq"}:
        return registry.teachers.get(mode, registry.default_segment_teacher())
    return registry.default_segment_teacher()

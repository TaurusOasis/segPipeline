"""Shared auto-labeling core for benchmark and production (steps 2-4).

Edge detector: yolo26s-seg (deploy student).
GPU teachers: SAM2 / SamHQ for mask generation and label cleaning only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import numpy as np

from ..agents.prompt_agent import PromptDecision, plan_prompts
from ..config import Config
from ..eval.label_quality import Decision, decision_and_tags, quality_gates_from_config
from ..models.tiers import ModelTierRegistry, TeacherModelSpec, load_model_tiers, resolve_teacher
from ..refine.mask_postprocess import postprocess_from_config
from .mock_sam2 import segment_with_prompts
from .sam_teacher import segment_with_teacher, teacher_segment_source
from .yolo_person_detector import PersonDetection, detect_persons

SegmentMode = Literal["grabcut", "sam2", "samhq", "oracle", "noisy_oracle"]


@dataclass(frozen=True)
class LabelingRuntime:
    segment_mode: SegmentMode = "sam2"
    teacher_key: str = "sam2"
    teacher: TeacherModelSpec | None = None
    yolo_weights: str = "/home/genesis/Train/Code/ultralytics/yolo26s-seg.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.7
    sam2_weights: str = "sam2_b.pt"
    device: str | int = 0
    max_instances: int = 16
    noise_level: float = 0.15
    quality_gates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceLabelResult:
    mask: np.ndarray
    bbox_xyxy: list[int]
    prompt: PromptDecision
    segment_source: str
    det_score: float | None = None
    decision: Decision = "review"
    error_tags: list[str] = field(default_factory=list)
    improvement_hint: str = ""
    quality_scores: dict[str, object] = field(default_factory=dict)
    detector_meta: dict[str, float] = field(default_factory=dict)


def _registry_from_cfg(cfg: Config) -> ModelTierRegistry:
    return load_model_tiers(cfg)


def labeling_runtime_from_config(
    cfg: Config,
    *,
    segment_mode: SegmentMode | None = None,
    teacher_key: str | None = None,
) -> LabelingRuntime:
    lcfg = cfg.get("labeling", {})
    bcfg = cfg.get("coconut_benchmark", {})
    registry = _registry_from_cfg(cfg)
    edge = registry.edge

    mode = segment_mode or lcfg.get("segment_mode", lcfg.get("sam_mode", bcfg.get("sam_mode", "sam2")))
    key = teacher_key or lcfg.get("segment_teacher", lcfg.get("teacher", mode))
    teacher = resolve_teacher(registry, teacher_key=str(key), segment_mode=str(mode))
    gates_raw = lcfg.get("quality_gates") or bcfg.get("quality_gates")
    yolo_weights = str(
        lcfg.get("yolo_weights", lcfg.get("edge_weights", edge.weights or bcfg.get("yolo_weights", edge.weights)))
    )
    teacher_weights = str(lcfg.get("teacher_weights", lcfg.get("sam2_weights", teacher.weights or "sam2_b.pt")))
    if teacher_weights:
        teacher = replace(teacher, weights=teacher_weights)

    return LabelingRuntime(
        segment_mode=str(mode),  # type: ignore[arg-type]
        teacher_key=str(key),
        teacher=teacher,
        yolo_weights=yolo_weights,
        yolo_conf=float(lcfg.get("yolo_conf", bcfg.get("yolo_conf", 0.25))),
        yolo_iou=float(lcfg.get("yolo_iou", bcfg.get("yolo_iou", 0.7))),
        sam2_weights=teacher.weights or teacher_weights,
        device=lcfg.get("device", bcfg.get("device", teacher.device if teacher else 0)),
        max_instances=int(lcfg.get("max_instances", 16)),
        noise_level=float(bcfg.get("noise_level", 0.15)),
        quality_gates=quality_gates_from_config(gates_raw),
    )


def segment_from_prompt(
    image_bgr: np.ndarray,
    prompt: PromptDecision,
    runtime: LabelingRuntime,
    *,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    mode = runtime.segment_mode
    if mode in {"oracle", "noisy_oracle"}:
        return segment_with_prompts(
            image_bgr,
            prompt,
            gt_mask=gt_mask if mode == "oracle" or mode == "noisy_oracle" else None,
            noise_level=runtime.noise_level if mode == "noisy_oracle" else 0.0,
        )

    backend = runtime.teacher.backend if runtime.teacher else ("grabcut" if mode == "grabcut" else mode)  # type: ignore[assignment]
    return segment_with_teacher(
        image_bgr,
        prompt,
        teacher=runtime.teacher,
        backend=backend,  # type: ignore[arg-type]
        weights=runtime.sam2_weights,
        device=runtime.device,
        fallback_grabcut=True,
        gt_mask=gt_mask,
        noise_level=runtime.noise_level,
    )


def label_instance_from_bbox(
    image_bgr: np.ndarray,
    *,
    bbox_xyxy: list[int],
    width: int,
    height: int,
    runtime: LabelingRuntime,
    cfg: Config,
    gt_mask: np.ndarray | None = None,
    multi_person: bool = False,
    detector_meta: dict[str, float] | None = None,
    det_score: float | None = None,
    neighbor_bboxes: list[list[int]] | None = None,
    boundary_f1: float | None = None,
) -> InstanceLabelResult:
    prompt = plan_prompts(
        bbox_xyxy=bbox_xyxy,
        width=width,
        height=height,
        gt_mask=gt_mask,
        neighbor_bboxes=neighbor_bboxes,
        boundary_f1=boundary_f1,
    )
    mask = segment_from_prompt(image_bgr, prompt, runtime, gt_mask=gt_mask)
    mask = postprocess_from_config(mask, cfg)
    backend = runtime.teacher.backend if runtime.teacher else runtime.segment_mode  # type: ignore[assignment]
    segment_source = teacher_segment_source(backend)  # type: ignore[arg-type]

    iou = boundary = None
    stats: dict[str, float] = {}
    if gt_mask is not None:
        from ..eval.boundary_metrics import boundary_f_score, mask_iou

        iou = float(mask_iou(mask, gt_mask))
        boundary = float(boundary_f_score(mask, gt_mask))
        from ..eval.label_quality import mask_error_stats

        stats = mask_error_stats(mask, gt_mask)
    decision, tags, hint = decision_and_tags(
        iou=iou,
        boundary=boundary,
        stats=stats,
        gates=runtime.quality_gates,
        prompt_needs_scribble=bool(prompt.needs_scribble),
        detector_meta=detector_meta or {},
        multi_person=multi_person,
        pred_empty=not bool(np.asarray(mask).any()),
        prompt_confidence=float(prompt.confidence),
    )
    quality_scores = {
        "semantic_score": float(iou if iou is not None else prompt.confidence),
        "boundary_score": float(boundary if boundary is not None else prompt.confidence),
        "identity_score": 1.0 if (iou is not None and iou > 0.5) else float(prompt.confidence),
        **stats,
        **(detector_meta or {}),
    }
    return InstanceLabelResult(
        mask=mask,
        bbox_xyxy=bbox_xyxy,
        prompt=prompt,
        segment_source=segment_source,
        det_score=det_score,
        decision=decision,
        error_tags=tags,
        improvement_hint=hint,
        quality_scores=quality_scores,
        detector_meta=detector_meta or {},
    )


def detect_persons_for_image(image_bgr: np.ndarray, runtime: LabelingRuntime) -> list[PersonDetection]:
    return detect_persons(
        image_bgr,
        weights=runtime.yolo_weights,
        conf=runtime.yolo_conf,
        iou=runtime.yolo_iou,
        device=runtime.device,
    )[: runtime.max_instances]

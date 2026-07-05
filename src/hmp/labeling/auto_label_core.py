"""Shared auto-labeling core for benchmark and production (steps 2-4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from ..agents.prompt_agent import PromptDecision, plan_prompts
from ..config import Config
from ..eval.label_quality import Decision, decision_and_tags, quality_gates_from_config
from ..refine.mask_postprocess import postprocess_from_config
from .mock_sam2 import segment_with_prompts
from .sam2_adapter import segment_with_sam2
from .yolo_person_detector import PersonDetection, detect_persons

SegmentMode = Literal["grabcut", "sam2", "oracle", "noisy_oracle"]


@dataclass(frozen=True)
class LabelingRuntime:
    segment_mode: SegmentMode = "sam2"
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
    quality_scores: dict[str, float] = field(default_factory=dict)
    detector_meta: dict[str, float] = field(default_factory=dict)


def labeling_runtime_from_config(cfg: Config, *, segment_mode: SegmentMode | None = None) -> LabelingRuntime:
    lcfg = cfg.get("labeling", {})
    bcfg = cfg.get("coconut_benchmark", {})
    mode = segment_mode or lcfg.get("segment_mode", lcfg.get("sam_mode", bcfg.get("sam_mode", "sam2")))
    gates_raw = lcfg.get("quality_gates") or bcfg.get("quality_gates")
    return LabelingRuntime(
        segment_mode=str(mode),  # type: ignore[arg-type]
        yolo_weights=str(lcfg.get("yolo_weights", bcfg.get("yolo_weights", "/home/genesis/Train/Code/ultralytics/yolo26s-seg.pt"))),
        yolo_conf=float(lcfg.get("yolo_conf", bcfg.get("yolo_conf", 0.25))),
        yolo_iou=float(lcfg.get("yolo_iou", bcfg.get("yolo_iou", 0.7))),
        sam2_weights=str(lcfg.get("sam2_weights", bcfg.get("sam2_weights", "sam2_b.pt"))),
        device=lcfg.get("device", bcfg.get("device", 0)),
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
    if mode == "grabcut":
        return segment_with_prompts(image_bgr, prompt, gt_mask=None)
    if mode == "sam2":
        return segment_with_sam2(
            image_bgr,
            prompt,
            weights=runtime.sam2_weights,
            device=runtime.device,
            fallback_grabcut=True,
        )
    return segment_with_prompts(
        image_bgr,
        prompt,
        gt_mask=gt_mask if mode in {"oracle", "noisy_oracle"} else None,
        noise_level=runtime.noise_level if mode == "noisy_oracle" else 0.0,
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
) -> InstanceLabelResult:
    prompt = plan_prompts(
        bbox_xyxy=bbox_xyxy,
        width=width,
        height=height,
        gt_mask=gt_mask,
    )
    mask = segment_from_prompt(image_bgr, prompt, runtime, gt_mask=gt_mask)
    mask = postprocess_from_config(mask, cfg)
    segment_source = "sam2" if runtime.segment_mode == "sam2" else "mock_sam2"

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

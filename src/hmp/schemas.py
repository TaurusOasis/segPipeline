"""Core data contracts for hmp (Step 01).

Pydantic models for the three JSONL formats used across the pipeline:

* manifest JSONL       -> :class:`MediaItem`
* annotation JSONL     -> :class:`AnnotationRecord` (with :class:`InstanceAnnotation`)
* quality score JSONL  -> :class:`QualityRecord`

These are the *stable* on-disk contracts; downstream stages read/write them.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MediaItem(BaseModel):
    """One row per image or video frame in the manifest."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    media_type: Literal["image", "video", "frame"] = "image"
    path: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    sha256: str
    source_video: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)
    timestamp_ms: Optional[int] = Field(default=None, ge=0)
    split: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    source_dataset: Optional[str] = None
    stratification: Optional[StratificationTags] = None
    license_meta: dict[str, str] = Field(default_factory=dict)


class StratificationTags(BaseModel):
    """Sampling bucket tags for hard-case mining and balanced training."""

    model_config = ConfigDict(extra="allow")

    person_distance: Optional[Literal["near", "mid", "far"]] = None
    hair_complexity: Optional[Literal["low", "mid", "high"]] = None
    occlusion: Optional[Literal["none", "partial", "heavy"]] = None
    multi_person: Optional[bool] = None
    motion_blur: Optional[Literal["none", "light", "heavy"]] = None
    lighting: Optional[Literal["normal", "backlit", "low_light", "high_contrast"]] = None
    background_complexity: Optional[Literal["plain", "moderate", "complex"]] = None


class InstanceAnnotation(BaseModel):
    """One detected person instance within an image/frame."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    category: str = "person"
    bbox_xyxy: list[int] = Field(..., min_length=4, max_length=4)
    mask_path: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: Optional[str] = None
    track_id: Optional[str] = None
    target_id: Optional[str] = None
    keypoints_path: Optional[str] = None
    prompt_history: list[dict[str, object]] = Field(default_factory=list)

    @field_validator("bbox_xyxy")
    @classmethod
    def _validate_bbox(cls, v: list[int]) -> list[int]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"bbox_xyxy must satisfy x2>x1 and y2>y1, got {v}"
            )
        if any(c < 0 for c in v):
            raise ValueError(f"bbox_xyxy must be non-negative, got {v}")
        return v


class AnnotationRecord(BaseModel):
    """One row per image/frame, with zero or more person instances."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    instances: list[InstanceAnnotation] = Field(default_factory=list)


class QualityRecord(BaseModel):
    """One row per item with merged quality scores and a keep/refine/review/drop decision."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    decision: Literal["keep", "refine", "review", "drop"] = "keep"
    reason: str = ""


class RelabelStep(BaseModel):
    """One stage in the mask-to-matte relabeling pipeline (steps 0-11)."""

    model_config = ConfigDict(extra="allow")

    index: int = Field(..., ge=0, le=11)
    name: str
    status: Literal["pending", "ready", "done", "skipped", "failed", "review"] = "pending"
    tool_options: list[str] = Field(default_factory=list)
    notes: str = ""


class RelabelTask(BaseModel):
    """One person instance queued for mask-to-matte alpha relabeling."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    item_id: str
    instance_id: str
    media_type: Literal["image", "video", "frame"] = "image"
    image_path: Optional[str] = None
    source_video: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)
    timestamp_ms: Optional[int] = Field(default=None, ge=0)
    mask_path: Optional[str] = None
    masklet_path: Optional[str] = None
    trimap_path: Optional[str] = None
    roi_path: Optional[str] = None
    trimap_or_roi_path: Optional[str] = None
    alpha_path: str
    alpha_exr_path: Optional[str] = None
    eval_map_path: Optional[str] = None
    bbox_path: str
    bbox_xyxy: list[int] = Field(..., min_length=4, max_length=4)
    keypoints_path: Optional[str] = None
    video_track_id: Optional[str] = None
    target_id: Optional[str] = None
    source_dataset: Optional[str] = None
    stratification: Optional[StratificationTags] = None
    candidate_tools: dict[str, list[str]] = Field(default_factory=dict)
    branch_outputs: dict[str, Optional[str]] = Field(default_factory=dict)
    expected_outputs: dict[str, Optional[str]] = Field(default_factory=dict)
    steps: list[RelabelStep] = Field(default_factory=list)
    quality_scores: dict[str, float] = Field(default_factory=dict)
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    branch_source: dict[str, Optional[str]] = Field(default_factory=dict)
    prompt_history: list[dict[str, object]] = Field(default_factory=list)
    license_meta: dict[str, object] = Field(default_factory=dict)
    status: Literal["pending", "ready", "running", "review", "accepted", "rejected"] = "pending"
    review_required: bool = True

    @field_validator("bbox_xyxy")
    @classmethod
    def _validate_bbox(cls, v: list[int]) -> list[int]:
        return validate_bbox_xyxy(v)


class AlphaBranchRecord(BaseModel):
    """One alpha teacher branch output before fusion."""

    model_config = ConfigDict(extra="allow")

    branch: Literal["Bv", "Bi", "Bd", "Bs", "Bg"]
    alpha_path: str
    provider: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MqeRecord(BaseModel):
    """Quality evaluation output for one frame/instance."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    instance_id: str
    reliable_map_path: Optional[str] = None
    eval_map_path: Optional[str] = None
    clip_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    review_required: bool = False
    failed_rules: list[str] = Field(default_factory=list)


class AlphaLabelRecord(BaseModel):
    """Final alpha label manifest row after alpha generation and review."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    instance_id: str
    image_path: str
    source_video: Optional[str] = None
    frame_index: Optional[int] = Field(default=None, ge=0)
    timestamp_ms: Optional[int] = Field(default=None, ge=0)
    alpha_path: str
    alpha_exr_path: Optional[str] = None
    mask_path: Optional[str] = None
    masklet_path: Optional[str] = None
    trimap_path: Optional[str] = None
    roi_path: Optional[str] = None
    trimap_or_roi_path: Optional[str] = None
    eval_map_path: Optional[str] = None
    bbox_path: Optional[str] = None
    bbox_xyxy: list[int] = Field(..., min_length=4, max_length=4)
    video_track_id: Optional[str] = None
    target_id: Optional[str] = None
    keypoints_path: Optional[str] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_scores: dict[str, float] = Field(default_factory=dict)
    branch_source: dict[str, Optional[str]] = Field(default_factory=dict)
    prompt_history: list[dict[str, object]] = Field(default_factory=list)
    license_meta: dict[str, object] = Field(default_factory=dict)
    source_task_id: Optional[str] = None
    review_status: Literal["unreviewed", "accepted", "needs_fix", "rejected"] = "unreviewed"

    @field_validator("bbox_xyxy")
    @classmethod
    def _validate_bbox(cls, v: list[int]) -> list[int]:
        return validate_bbox_xyxy(v)


class BenchmarkRecord(BaseModel):
    """Auto-label vs GT comparison for one person instance."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    instance_id: str
    image_path: str
    gt_mask_path: str
    pred_mask_path: str = ""
    diff_mask_path: str = ""
    detector_mode: str = "gt_bbox"
    sam_mode: str = "grabcut"
    mask_iou: float = Field(..., ge=0.0, le=1.0)
    boundary_f_score: float = Field(..., ge=0.0, le=1.0)
    bbox_iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gt_area_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pred_area_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    false_positive_ratio: Optional[float] = Field(default=None, ge=0.0)
    false_negative_ratio: Optional[float] = Field(default=None, ge=0.0)
    prompt_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    needs_scribble: bool = False
    decision: Literal["accept", "review", "reject"] = "review"
    error_tags: list[str] = Field(default_factory=list)
    improvement_hint: str = ""
    elapsed_ms: float = Field(..., ge=0.0)
    quality_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Convenience constructors / validators
# ---------------------------------------------------------------------------
def validate_bbox_xyxy(bbox: list[int]) -> list[int]:
    """Standalone bbox validation used outside pydantic contexts."""
    if len(bbox) != 4:
        raise ValueError(f"bbox_xyxy must have 4 elements, got {len(bbox)}")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox_xyxy must satisfy x2>x1 and y2>y1, got {bbox}")
    if any(c < 0 for c in bbox):
        raise ValueError(f"bbox_xyxy must be non-negative, got {bbox}")
    return bbox

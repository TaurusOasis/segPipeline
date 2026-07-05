"""Twelve-stage mask-to-matte relabeling pipeline (steps 0-11).

Canonical stage registry for RL-augmented auto-labeling + matting. Heavy model
execution lives in adapters; this module defines contracts and step plans only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..schemas import RelabelStep

StepStatus = Literal["pending", "ready", "done", "skipped", "failed", "review"]

PIPELINE_VERSION = 2


@dataclass(frozen=True)
class PipelineStage:
    index: int
    name: str
    title: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    tool_options: tuple[str, ...] = ()
    hmp_commands: tuple[str, ...] = ()
    notes: str = ""


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        index=0,
        name="data_source_sampling",
        title="数据采样",
        tool_options=("sa_v", "real_videos", "openimages_video", "youtube", "coco_rem", "coconut"),
        hmp_commands=("hmp dataset ingest", "hmp dataset coconut-sample"),
        notes="SA-V / real videos / OpenImages-video / YouTube / COCO-ReM / COCONut",
    ),
    PipelineStage(
        index=1,
        name="video_preprocess_bucket",
        title="视频预处理与分桶",
        tool_options=("shot_detection", "blur_score", "motion_score", "multi_person", "hair", "occlusion"),
        hmp_commands=("hmp dataset stratify", "hmp frames extract"),
        notes="shot / blur / motion / multi-person / hair / occlusion buckets",
    ),
    PipelineStage(
        index=2,
        name="human_discovery",
        title="人体发现",
        tool_options=("detector", "GroundingDINO", "pose", "person_classifier"),
        hmp_commands=("hmp label dummy", "hmp label yolo-sam2", "hmp label ultralytics-auto --dry-run"),
    ),
    PipelineStage(
        index=3,
        name="rl_prompt_agent",
        title="RL Prompt Agent",
        tool_options=("keyframe_select", "box_prompt", "point_prompt", "mask_prompt", "scribble_gate"),
        hmp_commands=("hmp agents prompt",),
        notes="选择关键帧与 prompt；决定是否人工 scribble",
    ),
    PipelineStage(
        index=4,
        name="sam2_vos_masklet",
        title="SAM2 / VOS masklet",
        tool_options=("SAM2", "VOS_tracker", "XMem", "Cutie", "mock_grabcut"),
        hmp_commands=("hmp label mock-sam2", "hmp label yolo-sam2"),
        notes="temporally consistent person masklet",
    ),
    PipelineStage(
        index=5,
        name="masklet_refinement",
        title="masklet refinement",
        tool_options=("SAMRefiner", "HQ-SAM", "temporal_consistency", "identity_check"),
        hmp_commands=("hmp refine masks",),
    ),
    PipelineStage(
        index=6,
        name="matting_critical_roi",
        title="matting-critical ROI",
        tool_options=("adaptive_unknown", "hair_roi", "hand_roi", "motion_blur_roi", "occlusion_roi", "semi_transparent_roi"),
        hmp_commands=("hmp matting make-adaptive-trimap",),
    ),
    PipelineStage(
        index=7,
        name="multi_branch_alpha",
        title="多分支 alpha 生成",
        tool_options=("Bv_video", "Bi_image", "Bd_diffusion", "Bs_segmentation_core"),
        hmp_commands=("hmp matting alpha-teacher", "hmp matting process-queue"),
        notes="Bv/Bi/Bd/Bs",
    ),
    PipelineStage(
        index=8,
        name="mqe_quality_evaluator",
        title="MQE / quality evaluator",
        tool_options=("MQE", "semantic_score", "boundary_score", "temporal_score", "identity_score"),
        hmp_commands=("hmp eval mqe",),
    ),
    PipelineStage(
        index=9,
        name="rl_fusion_repair_agent",
        title="RL Fusion & Repair Agent",
        tool_options=("select_video_alpha", "select_image_alpha", "select_diffusion_alpha", "human_repair", "clip_reject"),
        hmp_commands=("hmp agents fusion",),
    ),
    PipelineStage(
        index=10,
        name="human_in_the_loop",
        title="Human-in-the-loop 局部修正",
        tool_options=("local_boundary_paint", "trimap_edit", "SAM2_repropagate"),
        hmp_commands=("hmp relabel hitl-queue",),
        notes="只修低质量区域，不逐帧全修",
    ),
    PipelineStage(
        index=11,
        name="final_label_output",
        title="输出 label",
        outputs=("alpha.png", "mask.png", "eval_map.png", "branch_source.png", "prompt_history.json", "quality_score.json"),
        hmp_commands=("hmp relabel export-labels",),
    ),
)


def stage_by_index(index: int) -> PipelineStage:
    for stage in PIPELINE_STAGES:
        if stage.index == index:
            return stage
    raise KeyError(f"unknown pipeline stage index: {index}")


def stage_by_name(name: str) -> PipelineStage:
    for stage in PIPELINE_STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"unknown pipeline stage name: {name}")


def build_step_plan(
    *,
    completed_through: int = -1,
    ready_at: Optional[int] = None,
    overrides: Optional[dict[str, StepStatus]] = None,
) -> list[RelabelStep]:
    overrides = overrides or {}
    steps: list[RelabelStep] = []
    for stage in PIPELINE_STAGES:
        if stage.name in overrides:
            status = overrides[stage.name]
        elif stage.index <= completed_through:
            status = "done"
        elif ready_at is not None and stage.index == ready_at:
            status = "ready"
        else:
            status = "pending"
        steps.append(
            RelabelStep(
                index=stage.index,
                name=stage.name,
                status=status,
                tool_options=list(stage.tool_options),
                notes=stage.notes,
            )
        )
    return steps

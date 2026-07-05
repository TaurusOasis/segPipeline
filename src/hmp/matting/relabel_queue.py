"""Build mask-to-matte relabeling queues.

The queue is a CPU-only contract between preprocessing/labeling stages and
future heavy alpha teachers. It writes deterministic JSONL tasks describing
inputs, planned outputs, branch paths, review requirements, and the full
12-stage relabeling lifecycle (steps 0-11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..agents.prompt_agent import plan_prompts
from ..eval.label_quality import parse_decision_from_prompt_history
from ..matting.alpha_branches import plan_branch_outputs
from ..pipeline.stages import PIPELINE_STAGES, build_step_plan
from ..schemas import AnnotationRecord, MediaItem, RelabelTask

log = get_logger("hmp.matting.relabel_queue")


DEFAULT_TOOLS: dict[str, list[str]] = {
    stage.name: list(stage.tool_options) for stage in PIPELINE_STAGES
}


def _safe_id(*parts: str) -> str:
    return "_".join(p.replace("/", "_").replace("\\", "_").replace(" ", "_") for p in parts if p)


def _model_extra_value(model: object, key: str) -> object | None:
    extra = getattr(model, "model_extra", None) or {}
    return getattr(model, key, None) or extra.get(key)


def _tools_from_config(cfg: Config) -> dict[str, list[str]]:
    configured = cfg.get("relabel", {}).get("candidate_tools", {})
    tools = {k: list(v) for k, v in DEFAULT_TOOLS.items()}
    if isinstance(configured, Config):
        configured = configured.to_dict()
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, list):
                tools[key] = [str(v) for v in value]
    return tools


def _qa_from_prompt_history(
    prompt_history: list[dict[str, object]],
) -> tuple[str | None, dict[str, float], list[str]]:
    """Extract QA decision, scores, and tags from the latest labeling history."""
    decision = parse_decision_from_prompt_history(prompt_history)
    quality_scores: dict[str, float] = {}
    error_tags: list[str] = []
    for entry in reversed(prompt_history):
        raw_scores = entry.get("quality_scores")
        if isinstance(raw_scores, dict) and not quality_scores:
            quality_scores = {
                str(k): float(v)
                for k, v in raw_scores.items()
                if isinstance(v, (int, float))
            }
        raw_tags = entry.get("error_tags")
        if isinstance(raw_tags, list) and not error_tags:
            error_tags = [str(tag) for tag in raw_tags]
        if decision and quality_scores:
            break
    return decision, quality_scores, error_tags


def _task_status_from_qa(
    *,
    decision: str | None,
    completed: int,
    has_mask: bool,
) -> str:
    if not has_mask:
        return "pending"
    if decision == "reject":
        return "rejected"
    if decision == "review":
        return "review" if completed >= 6 else "pending"
    if completed >= 6:
        return "ready"
    return "pending"


def _infer_completed_through(*, has_manifest: bool, has_annotation: bool, has_mask: bool, has_trimap: bool) -> int:
    if not has_manifest:
        return -1
    if not has_annotation:
        return 1
    if not has_mask:
        return 3
    if not has_trimap:
        return 5
    return 6


def _read_manifest(path: Path) -> dict[str, MediaItem]:
    if not path.exists():
        return {}
    return {item.item_id: item for item in read_jsonl_list(path, model=MediaItem)}


def _annotation_path(cfg: Config, root: Path) -> Path:
    paths = cfg.get("paths", {})
    relabel = cfg.get("relabel", {})
    explicit = relabel.get("annotation_path")
    if explicit:
        return resolve_path(root, explicit)

    refined = resolve_path(root, paths.get("refined_annotation_path", "data/annotations/annotations_refined.jsonl"))
    if refined.exists():
        return refined
    return resolve_path(root, paths.get("annotation_path", "data/annotations/annotations_raw.jsonl"))


def build_relabel_queue(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = True,
) -> Path:
    """Write a JSONL queue of person instances that need alpha relabeling."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    relabel = cfg.get("relabel", {})

    manifest_path = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))
    ann_path = _annotation_path(cfg, root)
    alpha_dir = resolve_path(root, paths.get("alpha_dir", "data/alpha"))
    queue_path = resolve_path(root, relabel.get("queue_path", str(alpha_dir / "relabel_queue.jsonl")))
    matte_dir = resolve_path(root, relabel.get("alpha_output_dir", str(alpha_dir / "mattes")))
    fused_dir = resolve_path(root, relabel.get("fused_alpha_dir", str(alpha_dir / "fused")))
    alpha_exr_dir = resolve_path(root, relabel.get("alpha_exr_output_dir", str(alpha_dir / "mattes_exr")))
    bbox_dir = resolve_path(root, relabel.get("bbox_output_dir", str(alpha_dir / "bboxes")))
    masklet_dir = resolve_path(root, relabel.get("masklet_output_dir", str(alpha_dir / "masklets")))
    trimap_dir = resolve_path(
        root,
        cfg.get("adaptive_trimap", {}).get("output_dir", cfg.get("trimap", {}).get("output_dir", str(alpha_dir / "trimaps"))),
    )
    roi_dir = resolve_path(root, cfg.get("adaptive_trimap", {}).get("roi_dir", str(alpha_dir / "roi")))
    eval_map_dir = resolve_path(root, relabel.get("eval_map_output_dir", relabel.get("eval_map_dir", str(alpha_dir / "eval_maps"))))
    write_bbox_sidecars = bool(relabel.get("write_bbox_sidecars", True))

    if dry_run and not ann_path.exists():
        log.info("[dry-run] annotation file is not present yet: %s", ann_path)
        log.info("[dry-run] would write relabel tasks to %s after annotation/refine stages", queue_path)
        return queue_path

    manifest = _read_manifest(manifest_path)
    records = read_jsonl_list(ann_path, model=AnnotationRecord)
    tools = _tools_from_config(cfg)

    tasks: list[RelabelTask] = []
    for rec in records:
        item = manifest.get(rec.item_id)
        image_path = item.path if item is not None else None
        media_type = item.media_type if item is not None else "image"
        is_video = media_type in {"video", "frame"} or bool(item and item.source_video)
        source_dataset = None
        license_meta: dict[str, object] = {}
        if item is not None:
            source_dataset = _model_extra_value(item, "source_dataset")
            if not source_dataset and item.tags:
                source_dataset = item.tags[0]
            license_meta = {
                k: v
                for k, v in {
                    "source_dataset": source_dataset,
                    "license": _model_extra_value(item, "license"),
                    "license_url": _model_extra_value(item, "license_url"),
                    "source_url": _model_extra_value(item, "source_url"),
                }.items()
                if v is not None
            }

        for inst in rec.instances:
            task_id = _safe_id(rec.item_id, inst.instance_id)
            planned_alpha = matte_dir / f"{task_id}_alpha.png"
            planned_fused = fused_dir / f"{task_id}_alpha.png"
            planned_alpha_exr = alpha_exr_dir / f"{task_id}_alpha.exr"
            planned_bbox = bbox_dir / f"{task_id}.json"
            planned_masklet = masklet_dir / f"{task_id}_masklet.json"
            planned_trimap = trimap_dir / f"{task_id}_trimap.png"
            planned_roi = roi_dir / f"{task_id}_unknown_roi.png"
            planned_eval = eval_map_dir / f"{task_id}_eval.png"
            trimap_path = str(planned_trimap) if planned_trimap.exists() else None
            roi_path = str(planned_roi) if planned_roi.exists() else None
            trimap_or_roi_path = trimap_path or roi_path or str(planned_roi)
            mask_path = inst.mask_path
            branch_outputs = plan_branch_outputs(task_id=task_id, alpha_dir=alpha_dir)
            branch_source = {
                "Bv": None,
                "Bi": None,
                "Bd": None,
                "Bs": mask_path,
                "fusion": None,
            }
            prompt_history = list(inst.prompt_history)
            if not prompt_history:
                decision = plan_prompts(
                    bbox_xyxy=inst.bbox_xyxy,
                    width=item.width if item is not None else max(inst.bbox_xyxy[2], 1),
                    height=item.height if item is not None else max(inst.bbox_xyxy[3], 1),
                    frame_index=(item.frame_index or 0) if item is not None else 0,
                )
                prompt_history.append(
                    {
                        "stage": "rl_prompt_agent",
                        "policy": decision.policy,
                        "prompt_type": "box",
                        "bbox_xyxy": inst.bbox_xyxy,
                        "keyframe_index": decision.keyframe_index,
                        "confidence": decision.confidence,
                        "source": inst.source,
                    }
                )
                for prompt in decision.prompts:
                    if prompt.get("type") == "box":
                        continue
                    prompt_history.append({"stage": "rl_prompt_agent", "policy": decision.policy, **prompt})
            if mask_path and not any(p.get("prompt_type") == "mask" for p in prompt_history):
                prompt_history.append(
                    {
                        "stage": "sam2_vos_masklet",
                        "prompt_type": "mask",
                        "mask_path": mask_path,
                        "source": inst.source,
                    }
                )
            completed = _infer_completed_through(
                has_manifest=item is not None,
                has_annotation=True,
                has_mask=bool(mask_path),
                has_trimap=bool(trimap_path or roi_path),
            )
            qa_decision, quality_scores, error_tags = _qa_from_prompt_history(prompt_history)
            review_required = qa_decision != "accept" if qa_decision else True
            quality_score = quality_scores.get("semantic_score")
            if quality_score is None and inst.score is not None:
                quality_score = float(inst.score)
            ready_at = 7 if completed >= 6 else (completed + 1 if completed < 11 else None)
            expected_outputs = {
                "image_or_video_frame": image_path,
                "alpha": str(planned_alpha),
                "alpha_png": str(planned_alpha),
                "fused_alpha": str(planned_fused),
                "alpha_exr": str(planned_alpha_exr),
                "mask": mask_path,
                "masklet": str(planned_masklet) if is_video else None,
                "trimap": trimap_path or str(planned_trimap),
                "trimap_or_roi": trimap_or_roi_path,
                "roi": roi_path or str(planned_roi),
                "eval_map": str(planned_eval),
                "bbox": str(planned_bbox),
                "instance_id": inst.instance_id,
                "video_track_id": inst.track_id,
                "target_id": inst.target_id or inst.instance_id,
                "quality_score": None,
                "branch_source": None,
                "prompt_history": None,
                "license_meta": None,
            }
            tasks.append(
                RelabelTask(
                    task_id=task_id,
                    item_id=rec.item_id,
                    instance_id=inst.instance_id,
                    media_type=media_type,
                    image_path=image_path,
                    source_video=item.source_video if item is not None else None,
                    frame_index=item.frame_index if item is not None else None,
                    timestamp_ms=item.timestamp_ms if item is not None else None,
                    mask_path=mask_path,
                    masklet_path=str(planned_masklet) if is_video else None,
                    trimap_path=trimap_path,
                    roi_path=roi_path,
                    trimap_or_roi_path=trimap_or_roi_path,
                    alpha_path=str(planned_alpha),
                    alpha_exr_path=str(planned_alpha_exr),
                    eval_map_path=str(planned_eval),
                    bbox_path=str(planned_bbox),
                    bbox_xyxy=inst.bbox_xyxy,
                    keypoints_path=inst.keypoints_path,
                    video_track_id=inst.track_id,
                    target_id=inst.target_id or inst.instance_id,
                    source_dataset=str(source_dataset) if source_dataset is not None else None,
                    candidate_tools=tools,
                    branch_outputs=branch_outputs,
                    expected_outputs=expected_outputs,
                    steps=build_step_plan(
                        completed_through=completed,
                        ready_at=ready_at,
                        overrides={"video_preprocess_bucket": "done" if is_video else "skipped"},
                    ),
                    branch_source=branch_source,
                    prompt_history=prompt_history,
                    license_meta=license_meta,
                    quality_scores=quality_scores,
                    quality_score=quality_score,
                    status=_task_status_from_qa(
                        decision=qa_decision,
                        completed=completed,
                        has_mask=bool(mask_path),
                    ),
                    review_required=review_required,
                )
            )

    if dry_run:
        log.info("[dry-run] would write %d relabel tasks to %s", len(tasks), queue_path)
        return queue_path

    write_jsonl(queue_path, tasks, overwrite=overwrite)
    if write_bbox_sidecars:
        bbox_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            bbox = {
                "task_id": task.task_id,
                "item_id": task.item_id,
                "instance_id": task.instance_id,
                "bbox_xyxy": task.bbox_xyxy,
                "video_track_id": task.video_track_id,
                "target_id": task.target_id,
                "prompt_history": task.prompt_history,
                "license_meta": task.license_meta,
            }
            Path(task.bbox_path).write_text(json.dumps(bbox, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Wrote %d relabel tasks to %s", len(tasks), queue_path)
    return queue_path

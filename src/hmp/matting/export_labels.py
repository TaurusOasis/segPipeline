"""Export final alpha label manifest (pipeline step 11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..pipeline.stages import build_step_plan
from ..schemas import AlphaLabelRecord, RelabelTask

log = get_logger("hmp.matting.export_labels")


def export_alpha_labels(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = True,
    accept_auto: bool = True,
) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    relabel = cfg.get("relabel", {})
    alpha_dir = resolve_path(root, cfg.get("paths", {}).get("alpha_dir", "data/alpha"))
    queue_path = resolve_path(root, relabel.get("queue_path", str(alpha_dir / "relabel_queue.jsonl")))
    out_path = resolve_path(root, relabel.get("labels_path", str(alpha_dir / "alpha_labels.jsonl")))

    tasks = read_jsonl_list(queue_path, model=RelabelTask)
    labels: list[AlphaLabelRecord] = []

    for task in tasks:
        if accept_auto and task.status == "accepted":
            review_status = "accepted"
        elif task.review_required:
            review_status = "needs_fix"
        else:
            review_status = "accepted"

        fused = task.expected_outputs.get("fused_alpha") or task.alpha_path
        labels.append(
            AlphaLabelRecord(
                item_id=task.item_id,
                instance_id=task.instance_id,
                image_path=task.image_path or "",
                source_video=task.source_video,
                frame_index=task.frame_index,
                timestamp_ms=task.timestamp_ms,
                alpha_path=str(fused),
                alpha_exr_path=task.alpha_exr_path,
                mask_path=task.mask_path,
                masklet_path=task.masklet_path,
                trimap_path=task.trimap_path,
                roi_path=task.roi_path,
                trimap_or_roi_path=task.trimap_or_roi_path,
                eval_map_path=task.eval_map_path,
                bbox_path=task.bbox_path,
                bbox_xyxy=task.bbox_xyxy,
                video_track_id=task.video_track_id,
                target_id=task.target_id,
                keypoints_path=task.keypoints_path,
                quality_score=task.quality_score,
                quality_scores=task.quality_scores,
                branch_source=task.branch_source,
                prompt_history=task.prompt_history,
                license_meta=task.license_meta,
                source_task_id=task.task_id,
                review_status=review_status,
            )
        )

    if dry_run:
        log.info("[dry-run] would export %d alpha labels -> %s", len(labels), out_path)
        return out_path

    write_jsonl(out_path, labels, overwrite=overwrite)

    meta_dir = out_path.parent / "label_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for task, label in zip(tasks, labels):
        stem = task.task_id
        (meta_dir / f"{stem}_prompt_history.json").write_text(
            json.dumps(task.prompt_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (meta_dir / f"{stem}_quality_score.json").write_text(
            json.dumps(
                {
                    "quality_score": task.quality_score,
                    "quality_scores": task.quality_scores,
                    "review_required": task.review_required,
                    "status": task.status,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    updated_tasks = [task.model_copy(update={"steps": build_step_plan(completed_through=11)}) for task in tasks]
    write_jsonl(queue_path, updated_tasks, overwrite=True)
    log.info("Exported %d alpha labels -> %s", len(labels), out_path)
    return out_path

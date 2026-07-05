"""Process relabel queue through alpha generation, MQE, and RL fusion (steps 7-9)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import read_binary_mask, write_uint8_image
from ..eval.mqe import evaluate_instance
from ..pipeline.stages import build_step_plan
from ..schemas import RelabelTask
from .adaptive_trimap import make_adaptive_trimap
from .alpha_fusion import fuse_alpha_from_paths
from .alpha_teacher import generate_all_branch_alphas

log = get_logger("hmp.matting.process_queue")


def _roi_paths(task: RelabelTask, alpha_dir: Path) -> dict[str, str]:
    stem = task.task_id
    roi_dir = alpha_dir / "roi"
    return {
        "foreground_core": str(roi_dir / f"{stem}_fg_core.png"),
        "background_core": str(roi_dir / f"{stem}_bg_core.png"),
        "unknown_roi": str(roi_dir / f"{stem}_unknown_roi.png"),
    }


def _ensure_roi(task: RelabelTask, alpha_dir: Path) -> dict[str, str]:
    paths = _roi_paths(task, alpha_dir)
    if all(Path(p).exists() for p in paths.values()):
        return paths
    if not task.mask_path:
        return paths
    mask = read_binary_mask(task.mask_path)
    tri, roi = make_adaptive_trimap(mask, base_radius=6, max_radius=16)
    roi_dir = alpha_dir / "roi"
    tri_dir = alpha_dir / "adaptive_trimaps"
    roi_dir.mkdir(parents=True, exist_ok=True)
    tri_dir.mkdir(parents=True, exist_ok=True)
    stem = task.task_id
    write_uint8_image(tri_dir / f"{stem}_trimap.png", tri)
    write_uint8_image(paths["foreground_core"], roi["foreground_core"].astype(np.uint8) * 255)
    write_uint8_image(paths["background_core"], roi["background_core"].astype(np.uint8) * 255)
    write_uint8_image(paths["unknown_roi"], roi["unknown_roi"].astype(np.uint8) * 255)
    return paths


def _mark_steps_done(task: RelabelTask, through: int) -> list:
    return build_step_plan(completed_through=through)


def process_relabel_queue(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    provider: str = "mock",
    overwrite: bool = True,
) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    relabel = cfg.get("relabel", {})
    alpha_dir = resolve_path(root, paths.get("alpha_dir", "data/alpha"))
    queue_path = resolve_path(root, relabel.get("queue_path", str(alpha_dir / "relabel_queue.jsonl")))
    fused_dir = resolve_path(root, relabel.get("fused_alpha_dir", str(alpha_dir / "fused")))
    reliable_dir = alpha_dir / "reliable"
    eval_map_dir = resolve_path(root, relabel.get("eval_map_dir", str(alpha_dir / "eval_maps")))

    if not queue_path.exists():
        raise FileNotFoundError(f"relabel queue not found: {queue_path}; run `hmp relabel queue` first")

    tasks = read_jsonl_list(queue_path, model=RelabelTask)
    updated: list[RelabelTask] = []

    for task in tasks:
        if not task.mask_path:
            updated.append(task)
            continue

        branch_records = generate_all_branch_alphas(
            cfg,
            task_id=task.task_id,
            image_path=task.image_path or "",
            mask_path=task.mask_path,
            trimap_path=task.trimap_path,
            alpha_dir=alpha_dir,
            project_root=root,
            dry_run=dry_run,
            provider=provider,
        )

        fused_alpha = fused_dir / f"{task.task_id}_alpha.png"
        eval_map = eval_map_dir / f"{task.task_id}_eval.png"
        reliable_map = reliable_dir / f"{task.task_id}_reliable.png"

        if dry_run:
            mqe_scores = {}
            quality_score = None
            review_required = True
            branch_source = {}
        else:
            roi_paths = _ensure_roi(task, alpha_dir)
            fuse_alpha_from_paths(
                cfg,
                task_id=task.task_id,
                branch_records=branch_records,
                reliable_map_path=str(reliable_map),
                roi_paths=roi_paths,
                output_alpha_path=fused_alpha,
                output_eval_map_path=eval_map,
                output_branch_source_path=alpha_dir / "branch_source" / f"{task.task_id}_branch_source.png",
                project_root=root,
            )
            # Recompute reliable map from fused alpha for MQE.
            mqe = evaluate_instance(
                item_id=task.item_id,
                instance_id=task.instance_id,
                alpha_path=fused_alpha,
                mask_path=Path(task.mask_path),
                output_reliable_path=reliable_map,
                output_eval_map_path=eval_map,
            )
            mqe_scores = dict(mqe.scores)
            quality_score = mqe.clip_quality_score
            review_required = mqe.review_required
            branch_source = {
                "core": "Bv",
                "boundary": "Bi",
                "diffusion_refine": "Bd",
                "fallback": "Bs",
                "policy": "heuristic_fusion_agent_v1",
            }

        updated.append(
            task.model_copy(
                update={
                    "branch_outputs": {rec.branch: rec.alpha_path for rec in branch_records},
                    "alpha_path": str(fused_alpha),
                    "eval_map_path": str(eval_map),
                    "quality_scores": mqe_scores,
                    "quality_score": quality_score,
                    "branch_source": branch_source,
                    "review_required": review_required,
                    "status": "review" if review_required else "accepted",
                    "steps": _mark_steps_done(task, through=9),
                }
            )
        )

    if dry_run:
        log.info("[dry-run] would process %d relabel tasks", len(updated))
        return queue_path

    fused_dir.mkdir(parents=True, exist_ok=True)
    reliable_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(queue_path, updated, overwrite=overwrite)
    log.info("Processed %d relabel tasks -> %s", len(updated), queue_path)
    return queue_path

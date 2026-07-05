"""Human-in-the-loop review queue export (pipeline step 10)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import RelabelTask

log = get_logger("hmp.matting.hitl_queue")


def build_hitl_queue(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = True,
) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    relabel = cfg.get("relabel", {})
    alpha_dir = resolve_path(root, cfg.get("paths", {}).get("alpha_dir", "data/alpha"))
    queue_path = resolve_path(root, relabel.get("queue_path", str(alpha_dir / "relabel_queue.jsonl")))
    out_path = resolve_path(root, relabel.get("hitl_queue_path", str(alpha_dir / "hitl_queue.jsonl")))

    tasks = read_jsonl_list(queue_path, model=RelabelTask)
    review_items = []
    for task in tasks:
        if not task.review_required and task.status not in {"review", "pending"}:
            continue
        review_items.append(
            {
                "task_id": task.task_id,
                "item_id": task.item_id,
                "instance_id": task.instance_id,
                "image_path": task.image_path,
                "fused_alpha_path": task.expected_outputs.get("fused_alpha") or task.alpha_path,
                "eval_map_path": task.eval_map_path,
                "quality_score": task.quality_score,
                "quality_scores": task.quality_scores,
                "status": task.status,
                "suggested_actions": ["boundary_paint", "trimap_edit", "SAM2_repropagation"],
            }
        )

    if dry_run:
        log.info("[dry-run] would write %d HITL items -> %s", len(review_items), out_path)
        return out_path

    write_jsonl(out_path, review_items, overwrite=overwrite)
    log.info("Wrote %d HITL review items -> %s", len(review_items), out_path)
    return out_path

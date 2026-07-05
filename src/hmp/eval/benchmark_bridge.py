"""Bridge COCONut benchmark outputs into relabel / HITL contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import AnnotationRecord, InstanceAnnotation, RelabelTask

log = get_logger("hmp.eval.benchmark_bridge")


def import_benchmark_annotations(
    benchmark_dir: Path,
    *,
    annotation_path: Path,
    overwrite: bool = True,
) -> Path:
    """Import benchmark `annotations_pred.jsonl` into pipeline annotation path."""
    src = benchmark_dir / "annotations_pred.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"missing {src}")
    records = read_jsonl_list(src, model=AnnotationRecord)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(annotation_path, records, overwrite=overwrite)
    log.info("Imported %d annotation records -> %s", len(records), annotation_path)
    return annotation_path


def benchmark_review_to_hitl(
    benchmark_dir: Path,
    *,
    hitl_path: Path,
    decisions: tuple[str, ...] = ("review", "reject"),
) -> Path:
    """Convert benchmark review queue rows into HITL-compatible JSONL."""
    review_path = benchmark_dir / "review_queue.jsonl"
    if not review_path.exists():
        from .coconut_benchmark import export_benchmark_review_queue

        export_benchmark_review_queue(benchmark_dir, review_path=review_path)

    rows = read_jsonl_list(review_path)
    hitl_rows = [
        {
            "task_id": f"{row['item_id']}_{row['instance_id']}",
            "item_id": row["item_id"],
            "instance_id": row["instance_id"],
            "image_path": row["image_path"],
            "gt_mask_path": row.get("gt_mask_path"),
            "pred_mask_path": row.get("pred_mask_path"),
            "diff_mask_path": row.get("diff_mask_path"),
            "decision": row.get("decision"),
            "quality_scores": row.get("quality_scores", {}),
            "error_tags": row.get("error_tags", []),
            "improvement_hint": row.get("improvement_hint", ""),
            "suggested_actions": ["prompt_correction", "SAM2_repropagation", "boundary_paint"],
            "source": "coconut_benchmark",
        }
        for row in rows
        if row.get("decision") in decisions
    ]
    write_jsonl(hitl_path, hitl_rows, overwrite=True)
    log.info("Wrote %d HITL rows from benchmark -> %s", len(hitl_rows), hitl_path)
    return hitl_path


def apply_iteration_patch(
    cfg: Config,
    patch_path: Path,
    *,
    out_path: Path | None = None,
) -> Path:
    """Merge `next_config_patch.yaml` from coconut-iterate into a config file."""
    import yaml

    base = cfg.to_dict()
    patch = yaml.safe_load(patch_path.read_text(encoding="utf-8")) or {}

    def _merge(dst: dict, src: dict) -> dict:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _merge(dst[key], value)
            else:
                dst[key] = value
        return dst

    merged = _merge(base, patch)
    target = out_path or patch_path.with_name("merged_relabel.yaml")
    target.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("Merged iteration patch -> %s", target)
    return target

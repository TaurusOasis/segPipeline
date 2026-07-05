"""Mask refinement pipeline (Step 12, local-only minimal).

Reads raw annotation JSONL, applies local postprocess (remove small components,
fill holes, optionally keep largest) to each mask, writes refined masks +
``annotations_refined.jsonl`` + ``refine_report.jsonl`` with before/after area
and component counts.

External refiners (CascadePSP, BPR, SAMRefiner, SegRefiner) are configured as
external-command adapters but disabled by default in the MVP — they are added
in a later milestone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import mask_area_ratio, read_binary_mask, write_binary_mask
from ..refine.mask_postprocess import postprocess_from_config
from ..schemas import AnnotationRecord, InstanceAnnotation

log = get_logger("hmp.refine")


def _n_components(mask: np.ndarray) -> int:
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0
    num, _ = cv2.connectedComponents(m, connectivity=8)
    return max(0, num - 1)


def refine_masks(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Run local mask refinement. Returns (annotations_refined, report)."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    in_ann = resolve_path(
        root, paths.get("annotation_path", "data/annotations/annotations_raw.jsonl")
    )
    out_ann = resolve_path(root, paths.get("refined_annotation_path", "data/annotations/annotations_refined.jsonl"))
    out_mask_dir = resolve_path(root, paths.get("masks_refined_dir", "data/masks_refined"))
    report_path = resolve_path(root, cfg.get("refine", {}).get("report_path", "data/annotations/refine_report.jsonl"))

    records = read_jsonl_list(in_ann, model=AnnotationRecord)
    log.info("Refining %d items", len(records))

    if dry_run:
        log.info("[dry-run] would refine masks -> %s", out_mask_dir)
        return out_ann, report_path

    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_records: list[AnnotationRecord] = []
    report: list[dict] = []

    for rec in records:
        new_instances: list[InstanceAnnotation] = []
        for inst in rec.instances:
            if not inst.mask_path:
                new_instances.append(inst)
                continue
            before = read_binary_mask(inst.mask_path)
            after = postprocess_from_config(before, cfg)
            out_path = out_mask_dir / f"{rec.item_id}_{inst.instance_id}.png"
            write_binary_mask(out_path, after)
            new_instances.append(
                inst.model_copy(update={"mask_path": str(out_path), "source": (inst.source or "raw") + "+refined_local"})
            )
            report.append(
                {
                    "item_id": rec.item_id,
                    "instance_id": inst.instance_id,
                    "area_before": mask_area_ratio(before),
                    "area_after": mask_area_ratio(after),
                    "components_before": _n_components(before),
                    "components_after": _n_components(after),
                }
            )
        out_records.append(AnnotationRecord(item_id=rec.item_id, instances=new_instances))

    write_jsonl(out_ann, out_records, overwrite=True)
    write_jsonl(report_path, report, overwrite=True)
    log.info("Wrote %d refined records + report to %s", len(out_records), out_ann)
    return out_ann, report_path
"""Bridge COCONut benchmark outputs into relabel / HITL contracts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import mask_to_bbox_xyxy, read_binary_mask, write_binary_mask, write_uint8_image
from ..eval.label_quality import parse_decision_from_prompt_history
from ..labeling.auto_label_core import label_instance_from_bbox, labeling_runtime_from_config
from ..schemas import AnnotationRecord, BenchmarkRecord, InstanceAnnotation, MediaItem

log = get_logger("hmp.eval.benchmark_bridge")

Decision = Literal["accept", "review", "reject"]


def _instance_decision(inst) -> Decision | None:
    return parse_decision_from_prompt_history(list(inst.prompt_history))


def filter_annotation_records(
    records: list[AnnotationRecord],
    *,
    decisions: tuple[str, ...] | None = None,
) -> list[AnnotationRecord]:
    """Keep instances whose prompt_history decision is in ``decisions``."""
    if not decisions:
        return records
    allowed = set(decisions)
    filtered: list[AnnotationRecord] = []
    for rec in records:
        instances = [inst for inst in rec.instances if _instance_decision(inst) in allowed]
        if instances:
            filtered.append(rec.model_copy(update={"instances": instances}))
    return filtered


def import_benchmark_manifest(
    benchmark_dir: Path,
    *,
    manifest_path: Path,
    overwrite: bool = True,
) -> Path:
    """Import benchmark ``manifest.jsonl`` into pipeline manifest path."""
    src = benchmark_dir / "manifest.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"missing {src}")
    rows = read_jsonl_list(src, model=MediaItem)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_path, rows, overwrite=overwrite)
    log.info("Imported %d manifest rows -> %s", len(rows), manifest_path)
    return manifest_path


def import_benchmark_annotations(
    benchmark_dir: Path,
    *,
    annotation_path: Path,
    overwrite: bool = True,
    decisions: tuple[str, ...] | None = None,
) -> Path:
    """Import benchmark ``annotations_pred.jsonl`` into pipeline annotation path."""
    src = benchmark_dir / "annotations_pred.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"missing {src}")
    records = read_jsonl_list(src, model=AnnotationRecord)
    records = filter_annotation_records(records, decisions=decisions)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(annotation_path, records, overwrite=overwrite)
    log.info(
        "Imported %d annotation records (%s) -> %s",
        len(records),
        f"decisions={decisions}" if decisions else "all",
        annotation_path,
    )
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
    """Merge ``next_config_patch.yaml`` from coconut-iterate into a config file."""
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


def bootstrap_from_benchmark(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    benchmark_dir: Path | None = None,
    import_decisions: tuple[str, ...] = ("accept", "review"),
    hitl_decisions: tuple[str, ...] = ("review", "reject"),
    overwrite: bool = True,
) -> dict[str, Path]:
    """Import benchmark manifest/annotations and export HITL queue for pipeline stages 0-2/10."""
    root = Path(project_root) if project_root else Path.cwd()
    bridge = cfg.get("coconut_bridge", {})
    paths = cfg.get("paths", {})

    if benchmark_dir is None:
        explicit = bridge.get("benchmark_dir")
        if explicit:
            benchmark_dir = resolve_path(root, explicit)
        else:
            mode = bridge.get("mode", "yolo_person__sam2")
            compare_root = resolve_path(root, bridge.get("compare_output_dir", "runs/coconut_compare"))
            benchmark_dir = compare_root / str(mode)

    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"benchmark dir not found: {benchmark_dir}")

    manifest_path = resolve_path(
        root,
        bridge.get("manifest_path", paths.get("manifest_path", "data/manifests/manifest.jsonl")),
    )
    ann_path = resolve_path(
        root,
        bridge.get("annotation_path", paths.get("annotation_path", "data/annotations/annotations_raw.jsonl")),
    )
    hitl_path = resolve_path(
        root,
        bridge.get(
            "hitl_queue_path",
            cfg.get("relabel", {}).get("hitl_queue_path", cfg.get("hitl", {}).get("queue_path", "data/hitl/review_queue.jsonl")),
        ),
    )

    out = {
        "benchmark_dir": benchmark_dir,
        "manifest_path": import_benchmark_manifest(benchmark_dir, manifest_path=manifest_path, overwrite=overwrite),
        "annotation_path": import_benchmark_annotations(
            benchmark_dir,
            annotation_path=ann_path,
            overwrite=overwrite,
            decisions=tuple(bridge.get("import_decisions", import_decisions)),
        ),
        "hitl_path": benchmark_review_to_hitl(
            benchmark_dir,
            hitl_path=hitl_path,
            decisions=tuple(bridge.get("hitl_decisions", hitl_decisions)),
        ),
    }
    log.info("Bootstrapped pipeline inputs from %s", benchmark_dir)
    return out


def export_bad_boundary_queue(
    benchmark_dir: Path,
    *,
    queue_path: Path | None = None,
    include_decisions: tuple[str, ...] = ("review", "reject", "accept"),
) -> Path:
    """Export benchmark instances tagged ``bad_boundary`` for SamHQ re-label."""
    from .coconut_benchmark import backfill_benchmark_records, export_benchmark_review_queue

    report_jsonl = benchmark_dir / "benchmark_records.jsonl"
    if not report_jsonl.exists():
        raise FileNotFoundError(f"missing {report_jsonl}")

    records = read_jsonl_list(report_jsonl, model=BenchmarkRecord)
    records = backfill_benchmark_records(records)
    allowed = set(include_decisions)
    queue_path = queue_path or (benchmark_dir / "bad_boundary_queue.jsonl")
    rows = [
        {
            "item_id": r.item_id,
            "instance_id": r.instance_id,
            "image_path": r.image_path,
            "gt_mask_path": r.gt_mask_path,
            "pred_mask_path": r.pred_mask_path,
            "decision": r.decision,
            "mask_iou": r.mask_iou,
            "boundary_f_score": r.boundary_f_score,
            "error_tags": r.error_tags,
            "improvement_hint": r.improvement_hint,
            "suggested_actions": ["samhq_relabel", "boundary_paint"],
            "teacher": "samhq",
        }
        for r in records
        if r.decision in allowed and "bad_boundary" in r.error_tags
    ]
    write_jsonl(queue_path, rows, overwrite=True)
    log.info("Exported %d bad_boundary items -> %s", len(rows), queue_path)

    review_path = benchmark_dir / "review_queue.jsonl"
    if not review_path.exists():
        export_benchmark_review_queue(benchmark_dir, review_path=review_path)
    return queue_path


def _annotation_index(records: list[AnnotationRecord]) -> dict[tuple[str, str], InstanceAnnotation]:
    index: dict[tuple[str, str], InstanceAnnotation] = {}
    for rec in records:
        for inst in rec.instances:
            index[(rec.item_id, inst.instance_id)] = inst
    return index


def relabel_bad_boundary_instances(
    cfg: Config,
    benchmark_dir: Path,
    *,
    project_root: Optional[Path] = None,
    teacher_key: str = "samhq",
    include_decisions: tuple[str, ...] = ("review", "reject", "accept"),
    max_instances: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Re-run boundary teacher (SamHQ) on bad_boundary instances and patch benchmark outputs."""
    from .coconut_benchmark import _with_derived_fields, backfill_benchmark_records, resummarize_benchmark_dir

    benchmark_dir = Path(benchmark_dir)
    report_jsonl = benchmark_dir / "benchmark_records.jsonl"
    ann_jsonl = benchmark_dir / "annotations_pred.jsonl"
    if not report_jsonl.exists():
        raise FileNotFoundError(f"missing {report_jsonl}")
    if not ann_jsonl.exists():
        raise FileNotFoundError(f"missing {ann_jsonl}")

    records = backfill_benchmark_records(read_jsonl_list(report_jsonl, model=BenchmarkRecord))
    ann_rows = read_jsonl_list(ann_jsonl, model=AnnotationRecord)
    ann_index = _annotation_index(ann_rows)
    allowed = set(include_decisions)
    targets = [
        r
        for r in records
        if r.decision in allowed and "bad_boundary" in r.error_tags
    ]
    if max_instances is not None:
        targets = targets[: int(max_instances)]

    if dry_run:
        log.info("[dry-run] would SamHQ re-label %d bad_boundary instances in %s", len(targets), benchmark_dir)
        return {"targets": len(targets), "benchmark_dir": str(benchmark_dir), "dry_run": True}

    runtime = labeling_runtime_from_config(cfg, segment_mode="samhq", teacher_key=teacher_key)
    pred_mask_dir = benchmark_dir / "pred_masks"
    diff_mask_dir = benchmark_dir / "diff_masks"
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    diff_mask_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    updated = 0
    record_by_key = {(r.item_id, r.instance_id): r for r in records}
    t0 = time.perf_counter()

    for record in targets:
        key = (record.item_id, record.instance_id)
        inst = ann_index.get(key)
        if inst is None:
            log.warning("missing annotation for %s/%s", record.item_id, record.instance_id)
            continue
        image_bgr = cv2.imread(record.image_path)
        if image_bgr is None:
            log.warning("missing image %s", record.image_path)
            continue
        gt_path = Path(record.gt_mask_path)
        gt = read_binary_mask(gt_path) if gt_path.exists() else None
        height, width = image_bgr.shape[:2]
        bbox = list(inst.bbox_xyxy)
        inst_t0 = time.perf_counter()
        result = label_instance_from_bbox(
            image_bgr,
            bbox_xyxy=bbox,
            width=width,
            height=height,
            runtime=runtime,
            cfg=cfg,
            gt_mask=gt,
            multi_person="multi_person" in record.error_tags,
            boundary_f1=float(record.boundary_f_score),
        )
        elapsed_ms = (time.perf_counter() - inst_t0) * 1000.0
        pred = result.mask
        pred_mask_path = Path(record.pred_mask_path) if record.pred_mask_path else pred_mask_dir / f"{record.item_id}_{record.instance_id}_pred.png"
        diff_mask_path = Path(record.diff_mask_path) if record.diff_mask_path else diff_mask_dir / f"{record.item_id}_{record.instance_id}_diff.png"
        write_binary_mask(pred_mask_path, pred)
        if gt is not None:
            write_uint8_image(diff_mask_path, np.asarray(pred, dtype=bool) ^ np.asarray(gt, dtype=bool))

        updated_inst = inst.model_copy(
            update={
                "bbox_xyxy": mask_to_bbox_xyxy(pred) or bbox,
                "mask_path": str(pred_mask_path),
                "source": result.segment_source,
                "prompt_history": list(inst.prompt_history)
                + [
                    {
                        "agent": "boundary_teacher",
                        "teacher": teacher_key,
                        "decision": result.decision,
                        "error_tags": result.error_tags,
                        "improvement_hint": result.improvement_hint,
                    }
                ],
            }
        )
        ann_index[key] = updated_inst

        new_record = BenchmarkRecord(
            **record.model_dump(),
            sam_mode=teacher_key,
            mask_iou=float(result.quality_scores.get("semantic_score", record.mask_iou)),
            boundary_f_score=float(result.quality_scores.get("boundary_score", record.boundary_f_score)),
            decision=result.decision,
            error_tags=result.error_tags,
            improvement_hint=result.improvement_hint,
            elapsed_ms=float(elapsed_ms),
            pred_mask_path=str(pred_mask_path),
            diff_mask_path=str(diff_mask_path) if gt is not None else record.diff_mask_path,
            quality_scores={
                **record.quality_scores,
                **result.quality_scores,
                "boundary_relabel_teacher": teacher_key,
            },
        )
        record_by_key[key] = _with_derived_fields(new_record)
        updated += 1

    merged_records = list(record_by_key.values())
    merged_ann: list[AnnotationRecord] = []
    for rec in ann_rows:
        instances = [ann_index.get((rec.item_id, inst.instance_id), inst) for inst in rec.instances]
        merged_ann.append(rec.model_copy(update={"instances": instances}))

    write_jsonl(report_jsonl, merged_records, overwrite=True)
    write_jsonl(ann_jsonl, merged_ann, overwrite=True)
    summary = resummarize_benchmark_dir(benchmark_dir)
    export_bad_boundary_queue(benchmark_dir, include_decisions=include_decisions)

    elapsed_s = time.perf_counter() - t0
    stats = {
        "targets": len(targets),
        "updated": updated,
        "elapsed_s": round(elapsed_s, 3),
        "accept_rate": summary.get("accept_rate"),
        "mean_boundary_f1": summary.get("mean_boundary_f1"),
        "benchmark_dir": str(benchmark_dir),
        "teacher": teacher_key,
    }
    (benchmark_dir / "boundary_relabel_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("SamHQ boundary re-label: updated=%d/%d -> %s", updated, len(targets), benchmark_dir)
    return stats

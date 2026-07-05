"""Bridge COCONut benchmark outputs into relabel / HITL contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..eval.label_quality import parse_decision_from_prompt_history
from ..schemas import AnnotationRecord, MediaItem

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

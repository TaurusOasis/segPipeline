"""Bridge COCONut benchmark accept masks into ultralytics distillation data."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal, Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.mask_io import combine_instance_masks, read_binary_mask
from ..data.yolo_seg_io import write_data_yaml, write_yolo_label
from ..eval.benchmark_bridge import filter_annotation_records
from ..models.tiers import load_model_tiers
from ..schemas import AnnotationRecord, MediaItem

log = get_logger("hmp.yolo.coconut_distill_bridge")

SplitName = Literal["val", "train"]


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _patch_val_labels(
    *,
    records: list[AnnotationRecord],
    items_by_id: dict[str, MediaItem],
    labels_val: Path,
    class_id: int,
) -> dict[str, object]:
    patched: list[str] = []
    skipped: list[str] = []
    for rec in records:
        item = items_by_id.get(rec.item_id)
        if item is None:
            skipped.append(rec.item_id)
            continue
        masks = []
        for inst in rec.instances:
            if not inst.mask_path:
                skipped.append(f"{rec.item_id}:{inst.instance_id}")
                continue
            masks.append(read_binary_mask(inst.mask_path))
        if not masks:
            skipped.append(rec.item_id)
            continue
        combined = combine_instance_masks(masks)
        label_path = labels_val / f"{rec.item_id}.txt"
        write_yolo_label(label_path, class_id, combined, item.width, item.height)
        patched.append(rec.item_id)
    return {"patched": patched, "skipped": skipped, "patched_count": len(patched)}


def overlay_accept_labels_on_coconut(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    benchmark_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Copy base COCONut YOLO layout and patch val labels with benchmark accept masks."""
    root = Path(project_root) if project_root else Path.cwd()
    bridge = cfg.get("coconut_distill_bridge", {})
    if benchmark_dir is None:
        explicit = bridge.get("benchmark_dir")
        if explicit:
            benchmark_dir = resolve_path(root, explicit)
        else:
            mode = bridge.get("mode", "yolo_person__sam2")
            compare_root = resolve_path(root, bridge.get("compare_output_dir", "runs/coconut_compare"))
            benchmark_dir = compare_root / str(mode)
    benchmark_dir = Path(benchmark_dir)
    ann_src = benchmark_dir / "annotations_pred.jsonl"
    manifest_src = benchmark_dir / "manifest.jsonl"
    if not ann_src.exists():
        raise FileNotFoundError(f"missing {ann_src}")
    if not manifest_src.exists():
        raise FileNotFoundError(f"missing {manifest_src}")

    base_root = Path(bridge.get("base_yolo_root", "/home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2"))
    out_root = resolve_path(root, bridge.get("output_yolo_root", "data/coconut/yolo_accept_overlay"))
    decisions = tuple(bridge.get("import_decisions", ("accept",)))
    class_names = list(bridge.get("class_names", ["person"]))
    class_id = int(bridge.get("class_id", 0))
    split: SplitName = bridge.get("patch_split", "val")  # type: ignore[assignment]

    records = filter_annotation_records(
        read_jsonl_list(ann_src, model=AnnotationRecord),
        decisions=decisions,
    )
    items = read_jsonl_list(manifest_src, model=MediaItem)
    items_by_id = {item.item_id: item for item in items}

    if dry_run:
        log.info(
            "[dry-run] would patch %d accept images into %s from %s",
            len(records),
            out_root,
            benchmark_dir,
        )
        return out_root / "data.yaml"

    if not base_root.exists():
        raise FileNotFoundError(f"base YOLO dataset not found: {base_root}")

    images_train = out_root / "images" / "train"
    images_val = out_root / "images" / "val"
    labels_train = out_root / "labels" / "train"
    labels_val = out_root / "labels" / "val"
    out_root.mkdir(parents=True, exist_ok=True)

    _symlink_or_copy(base_root / "images" / "train", images_train)
    _symlink_or_copy(base_root / "images" / "val", images_val)
    _symlink_or_copy(base_root / "labels" / "train", labels_train)

    base_labels_val = base_root / "labels" / "val"
    labels_val.mkdir(parents=True, exist_ok=True)
    for label in base_labels_val.glob("*.txt"):
        dst = labels_val / label.name
        if not dst.exists():
            _symlink_or_copy(label, dst)

    stats = _patch_val_labels(
        records=records,
        items_by_id=items_by_id,
        labels_val=labels_val,
        class_id=class_id,
    )
    data_yaml = out_root / "data.yaml"
    write_data_yaml(data_yaml, yolo_dir=out_root, class_names=class_names)

    manifest_out = out_root / "accept_overlay_manifest.jsonl"
    write_jsonl(
        manifest_out,
        [
            {
                "item_id": item_id,
                "benchmark_dir": str(benchmark_dir),
                "decisions": list(decisions),
                "patch_split": split,
            }
            for item_id in stats["patched"]  # type: ignore[index]
        ],
        overwrite=True,
    )
    stats_path = out_root / "accept_overlay_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                **stats,
                "benchmark_dir": str(benchmark_dir),
                "base_yolo_root": str(base_root),
                "output_yolo_root": str(out_root),
                "decisions": list(decisions),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "Patched %d val labels with accept masks -> %s (skipped=%d)",
        stats["patched_count"],
        out_root,
        len(stats["skipped"]),  # type: ignore[arg-type]
    )
    return data_yaml


def build_coconut_distill_plan(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    data_yaml: Path | None = None,
) -> dict[str, object]:
    """Return ultralytics distillation launch command and resolved paths."""
    root = Path(project_root) if project_root else Path.cwd()
    bridge = cfg.get("coconut_distill_bridge", {})
    registry = load_model_tiers(cfg)
    distill_teacher = registry.teachers.get(str(bridge.get("distill_teacher", "yolo26x-seg")))
    edge = registry.edge

    if data_yaml is None:
        data_yaml = resolve_path(root, bridge.get("output_yolo_root", "data/coconut/yolo_accept_overlay")) / "data.yaml"

    ultralytics_root = Path(bridge.get("ultralytics_root", "/home/genesis/Train/Code/ultralytics"))
    train_script = ultralytics_root / "scripts/train_yolo26s_seg_coconut_distill.py"
    python_bin = str(bridge.get("python_bin", "/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python"))
    student = str(bridge.get("student_weights", edge.weights or "yolo26s-seg.pt"))
    teacher = str(bridge.get("teacher_weights", distill_teacher.weights if distill_teacher else "yolo26x-seg.pt"))
    name = str(bridge.get("run_name", "yolo26s-seg-coconut-accept-overlay-distill"))
    epochs = int(bridge.get("epochs", 50))
    batch = int(bridge.get("batch", 64))
    device = str(bridge.get("device", "0,1"))

    cmd = (
        f"{python_bin} {train_script} "
        f"--data {Path(data_yaml).resolve()} "
        f"--student {student} "
        f"--teacher {teacher} "
        f"--name {name} "
        f"--epochs {epochs} "
        f"--batch {batch} "
        f"--device {device}"
    )
    return {
        "command": cmd,
        "data_yaml": str(Path(data_yaml).resolve()),
        "train_script": str(train_script),
        "student_weights": student,
        "teacher_weights": teacher,
        "ultralytics_root": str(ultralytics_root),
        "run_name": name,
    }

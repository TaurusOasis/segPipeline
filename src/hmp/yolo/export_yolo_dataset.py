"""Export project masks/annotations to a YOLO segmentation dataset (Step 07).

Reads manifest + annotation JSONL, converts binary masks to polygons, splits
train/val deterministically, copies/symlinks images, and writes ``data.yaml``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..common.jsonl import read_jsonl_list
from ..common.logging import get_logger
from ..config import Config, resolve_path, seed_from_config
from ..data.split_dataset import train_val_split
from ..data.yolo_seg_io import write_data_yaml, write_yolo_label
from ..schemas import AnnotationRecord, MediaItem

log = get_logger("hmp.yolo.export")


def _link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass  # fall back to copy
    shutil.copy2(src, dst)


def export_yolo_dataset(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    class_names: Optional[list[str]] = None,
) -> Path:
    """Build ``data/yolo_seg`` from manifest + annotation JSONL."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    manifest_path = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))
    annotation_path = resolve_path(
        root, paths.get("refined_annotation_path", paths.get("annotation_path", "data/annotations/annotations_raw.jsonl"))
    )
    yolo_dir = resolve_path(root, paths.get("yolo_dir", "data/yolo_seg"))

    e_cfg = cfg.get("yolo_export", {})
    val_ratio = float(e_cfg.get("val_ratio", 0.2))
    seed = int(e_cfg.get("seed", seed_from_config(cfg)))
    symlink = bool(e_cfg.get("symlink", True))
    class_names = class_names or e_cfg.get("class_names", ["person"])
    class_map = {name: i for i, name in enumerate(class_names)}

    items = read_jsonl_list(manifest_path, model=MediaItem)
    anns = {r.item_id: r for r in read_jsonl_list(annotation_path, model=AnnotationRecord)}

    train_items, val_items = train_val_split(items, val_ratio=val_ratio, seed=seed)
    log.info("Split: %d train / %d val", len(train_items), len(val_items))

    if dry_run:
        log.info("[dry-run] would write yolo seg dataset to %s", yolo_dir)
        return yolo_dir

    images_train = yolo_dir / "images" / "train"
    images_val = yolo_dir / "images" / "val"
    labels_train = yolo_dir / "labels" / "train"
    labels_val = yolo_dir / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    def _process(item: MediaItem, images_dir: Path, labels_dir: Path) -> int:
        from ..data.mask_io import combine_instance_masks, read_binary_mask

        rec = anns.get(item.item_id)
        # copy image
        src = Path(item.path)
        dst = images_dir / src.name
        _link_or_copy(src, dst, symlink)

        if rec is None or not rec.instances:
            # empty label file
            (labels_dir / (src.stem + ".txt")).write_text("", encoding="utf-8")
            return 0
        masks = []
        for inst in rec.instances:
            if inst.mask_path:
                masks.append(read_binary_mask(inst.mask_path))
        if not masks:
            (labels_dir / (src.stem + ".txt")).write_text("", encoding="utf-8")
            return 0
        combined = combine_instance_masks(masks)
        cls = class_map.get("person", 0)
        n = write_yolo_label(labels_dir / (src.stem + ".txt"), cls, combined, item.width, item.height)
        return n

    total_polys = 0
    for it in train_items:
        total_polys += _process(it, images_train, labels_train)
    for it in val_items:
        total_polys += _process(it, images_val, labels_val)

    write_data_yaml(yolo_dir / "data.yaml", yolo_dir=yolo_dir, class_names=class_names)
    log.info("Wrote YOLO seg dataset to %s (%d polygons)", yolo_dir, total_polys)
    return yolo_dir
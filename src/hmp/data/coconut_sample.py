"""Build COCONut val manifest for pipeline stage 0 sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..common.jsonl import write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import MediaItem
from .coconut_io import iter_coconut_person_samples, sample_to_media_item

log = get_logger("hmp.data.coconut_sample")


def build_coconut_manifest(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = True,
) -> Path:
    """Sample COCONut val images with person instances into manifest JSONL."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    sample_cfg = cfg.get("coconut_sample", {})
    bcfg = cfg.get("coconut_benchmark", {})

    out_path = resolve_path(
        root,
        sample_cfg.get("manifest_path", paths.get("manifest_path", "data/manifests/coconut_val.jsonl")),
    )
    coconut_root = Path(sample_cfg.get("coconut_root", bcfg.get("coconut_root", "/home/genesis/Train/Dataset/coconut")))
    image_root = Path(sample_cfg.get("image_root", bcfg.get("image_root", "/home/genesis/Train/Dataset/coco2017")))
    json_path = Path(sample_cfg.get("json_path", bcfg.get("json_path", coconut_root / "relabeled_coco_val.json")))
    mask_dir = Path(sample_cfg.get("mask_dir", bcfg.get("mask_dir", coconut_root / "relabeled_coco_val")))
    limit = int(sample_cfg.get("limit", bcfg.get("limit", 64)))
    seed = int(sample_cfg.get("seed", bcfg.get("seed", cfg.get("project", {}).get("seed", 42))))

    if dry_run:
        log.info("[dry-run] would write up to %d COCONut manifest rows -> %s", limit, out_path)
        return out_path

    rows: list[MediaItem] = []
    for sample in iter_coconut_person_samples(
        json_path=json_path,
        mask_dir=mask_dir,
        image_root=image_root,
        image_subdir=str(sample_cfg.get("image_subdir", bcfg.get("image_subdir", "val2017"))),
        limit=limit,
        seed=seed,
    ):
        item = sample_to_media_item(sample)
        extra = dict(item.model_extra or {})
        extra.update(
            {
                "source_dataset": "coconut",
                "coconut_mask_path": str(sample.mask_path),
                "person_count": len(sample.persons),
            }
        )
        rows.append(item.model_copy(update=extra))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows, overwrite=overwrite)
    log.info("Wrote %d COCONut manifest rows -> %s", len(rows), out_path)
    return out_path

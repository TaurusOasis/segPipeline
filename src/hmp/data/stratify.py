"""Video/image stratification and dedup tagging (pipeline step 1)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import MediaItem, StratificationTags

log = get_logger("hmp.data.stratify")


def _blur_score(image_path: Path) -> float:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _brightness_score(image_path: Path) -> float:
    from PIL import Image, UnidentifiedImageError

    try:
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return 0.5
    return float(arr.mean() / 255.0)


def infer_stratification(item: MediaItem) -> StratificationTags:
    path = Path(item.path)
    blur = _blur_score(path)
    bright = _brightness_score(path)
    area_ratio = min(item.width, item.height) / max(item.width, item.height)

    motion_blur = "heavy" if blur < 40 else ("light" if blur < 120 else "none")
    lighting = "low_light" if bright < 0.25 else ("backlit" if bright > 0.85 else "normal")
    person_distance = "near" if area_ratio > 0.75 else ("far" if area_ratio < 0.45 else "mid")
    background_complexity = "plain" if blur > 180 else ("complex" if blur < 80 else "moderate")
    return StratificationTags(
        person_distance=person_distance,
        hair_complexity="mid",
        occlusion="none",
        multi_person=False,
        motion_blur=motion_blur,
        lighting=lighting,
        background_complexity=background_complexity,
    )


def assign_dedup_clusters(items: list[MediaItem]) -> dict[str, int]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for item in items:
        by_hash[item.sha256].append(item.item_id)
    cluster_map: dict[str, int] = {}
    cluster_id = 0
    for ids in by_hash.values():
        for item_id in ids:
            cluster_map[item_id] = cluster_id
        cluster_id += 1
    return cluster_map


def stratify_manifest(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    manifest_path = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))
    out_path = resolve_path(
        root,
        cfg.get("stratification", {}).get("output_path", str(manifest_path)),
    )

    items = read_jsonl_list(manifest_path, model=MediaItem)
    clusters = assign_dedup_clusters(items)

    if dry_run:
        log.info("[dry-run] would stratify %d manifest rows -> %s", len(items), out_path)
        return out_path

    enriched: list[MediaItem] = []
    for item in items:
        tags = infer_stratification(item)
        extra = dict(getattr(item, "model_extra", None) or {})
        extra["stratification"] = tags
        extra["dedup_cluster_id"] = clusters[item.item_id]
        enriched.append(item.model_copy(update={"stratification": tags, **extra}))

    write_jsonl(out_path, enriched, overwrite=True)
    log.info("Stratified %d manifest rows -> %s", len(enriched), out_path)
    return out_path

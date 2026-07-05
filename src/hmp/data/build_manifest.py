"""Build an image manifest JSONL (Step 02).

Scans configured image directories, reads width/height, computes sha256, and
writes one :class:`MediaItem` per image to the manifest path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..common.hashing import sha256_file
from ..common.image_io import iter_image_files, read_image_size
from ..common.jsonl import count_jsonl, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import MediaItem

log = get_logger("hmp.data.build_manifest")


def item_id_from_path(root: Path, path: Path) -> str:
    """Stable item_id: relative path without extension, '/' -> '_'."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    stem = rel.with_suffix("")
    return str(stem).replace("/", "_").replace("\\", "_")


def build_media_items(
    root: Path,
    files: Iterable[Path],
    *,
    media_type: str = "image",
    tags: Optional[list[str]] = None,
) -> list[MediaItem]:
    """Build MediaItem records for a list of image files under ``root``."""
    tags = tags or []
    items: list[MediaItem] = []
    for f in files:
        w, h = read_image_size(f)
        items.append(
            MediaItem(
                item_id=item_id_from_path(root, f),
                media_type=media_type,
                path=str(f),
                width=w,
                height=h,
                sha256=sha256_file(f),
                tags=list(tags),
            )
        )
    return items


def build_manifest(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:
    """Scan image dirs from config and write manifest JSONL.

    Config fields (under ``paths`` and ``manifest``):
      - raw_dir: directory to scan (required)
      - manifest_path: output JSONL (required)
      - extra_dirs: optional list of additional dirs
      - media_type: default "image"
      - tags: optional list
    """
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    raw_dir = resolve_path(root, paths.get("raw_dir", "data/raw"))
    manifest_path = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))

    dirs = [raw_dir]
    for d in (paths.get("extra_dirs") or []):
        dirs.append(resolve_path(root, d))
    media_type = cfg.get("manifest", {}).get("media_type", "image")
    tags = cfg.get("manifest", {}).get("tags", [])

    all_files: list[Path] = []
    for d in dirs:
        files = iter_image_files(d)
        log.info("Scanned %s: %d images", d, len(files))
        all_files.extend(files)

    if dry_run:
        log.info("[dry-run] would write %d items to %s", len(all_files), manifest_path)
        return manifest_path

    if manifest_path.exists() and not overwrite:
        existing = count_jsonl(manifest_path)
        log.info("Manifest already exists (%d items); use --overwrite to rebuild.", existing)
        return manifest_path

    items = build_media_items(raw_dir, all_files, media_type=media_type, tags=tags)
    write_jsonl(manifest_path, items, overwrite=overwrite)
    log.info("Wrote %d items to %s", len(items), manifest_path)
    return manifest_path
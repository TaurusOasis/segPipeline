"""Dataset ingest and manifest enrichment (pipeline step 0)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import MediaItem
from .build_manifest import build_manifest
from .dataset_registry import get_dataset_entry, load_dataset_registry, registry_path_from_config

log = get_logger("hmp.data.ingest")


def _license_meta(entry: dict) -> dict[str, str]:
    urls = entry.get("source_urls") or []
    return {
        "dataset_id": str(entry.get("display_name") or entry.get("role") or "unknown"),
        "role": str(entry.get("role", "unknown")),
        "modality": str(entry.get("modality", "unknown")),
        "license": "verify_before_use",
        "source_url": str(urls[0]) if urls else "",
    }


def enrich_manifest_with_dataset(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    """Tag manifest rows with source_dataset and license metadata."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    manifest_path = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))
    ingest = cfg.get("ingest", {})
    dataset_ids = list(ingest.get("datasets") or ingest.get("dataset_ids") or [])
    if not dataset_ids:
        dataset_ids = ["demo"]

    registry_path = registry_path_from_config(cfg, root)
    registry = load_dataset_registry(registry_path) if registry_path.exists() else {"datasets": {}}

    if not manifest_path.exists():
        log.info("manifest missing; building from raw dirs first")
        build_manifest(cfg, project_root=root, dry_run=dry_run, overwrite=True)

    if dry_run and not manifest_path.exists():
        log.info("[dry-run] would enrich manifest at %s with datasets %s", manifest_path, dataset_ids)
        return manifest_path

    items = read_jsonl_list(manifest_path, model=MediaItem)
    primary_id = dataset_ids[0]
    try:
        entry = get_dataset_entry(registry, primary_id)
        license_meta = _license_meta(entry)
        source_dataset = primary_id
    except KeyError:
        entry = {}
        license_meta = {"dataset_id": primary_id, "role": "demo", "license": "internal"}
        source_dataset = primary_id

    enriched: list[MediaItem] = []
    for item in items:
        tags = list(item.tags)
        if source_dataset not in tags:
            tags.append(source_dataset)
        extra = dict(getattr(item, "model_extra", None) or {})
        extra["source_dataset"] = source_dataset
        extra["license_meta"] = license_meta
        enriched.append(item.model_copy(update={"tags": tags, **extra}))

    if dry_run:
        log.info("[dry-run] would enrich %d manifest rows with source_dataset=%s", len(enriched), source_dataset)
        return manifest_path

    write_jsonl(manifest_path, enriched, overwrite=True)
    log.info("Enriched %d manifest rows with source_dataset=%s", len(enriched), source_dataset)
    return manifest_path

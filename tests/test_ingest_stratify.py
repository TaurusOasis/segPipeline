"""Tests for dataset ingest and stratification."""

from __future__ import annotations

from pathlib import Path

from hmp.common.jsonl import read_jsonl_list, write_jsonl
from hmp.config import Config
from hmp.data.ingest import enrich_manifest_with_dataset
from hmp.data.stratify import stratify_manifest
from hmp.schemas import MediaItem


def _manifest(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.jsonl"
    write_jsonl(
        p,
        [
            MediaItem(
                item_id="img",
                path=str(tmp_path / "img.jpg"),
                width=64,
                height=96,
                sha256="abc",
                tags=["demo"],
            )
        ],
    )
    (tmp_path / "img.jpg").write_bytes(b"fake")
    return p


def test_enrich_manifest_with_dataset(tmp_path):
    manifest = _manifest(tmp_path)
    cfg = Config(
        {
            "paths": {"manifest_path": str(manifest)},
            "ingest": {
                "datasets": ["coco_rem"],
                "registry_path": str(Path(__file__).resolve().parents[1] / "configs/datasets.yaml"),
            },
        }
    )
    out = enrich_manifest_with_dataset(cfg, project_root=tmp_path)
    items = read_jsonl_list(out, model=MediaItem)
    assert items[0].tags[-1] == "coco_rem"


def test_stratify_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    cfg = Config({"paths": {"manifest_path": str(manifest)}, "stratification": {"output_path": str(manifest)}})
    out = stratify_manifest(cfg, project_root=tmp_path)
    items = read_jsonl_list(out, model=MediaItem)
    extra = items[0].model_extra or {}
    assert "stratification" in extra or hasattr(items[0], "stratification")

"""Labeling interface and shared helpers (Step 06).

All real labeling models (SAM3, Grounded-SAM-2, HQ-SAM, Ultralytics
auto_annotate, ...) implement :class:`Labeler`. The interface is intentionally
small: accept a manifest, emit :class:`AnnotationRecord` rows plus mask files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from ..common.jsonl import read_jsonl, write_jsonl
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import AnnotationRecord, MediaItem

log = get_logger("hmp.labeling")


class Labeler(ABC):
    """Abstract base for person-mask labelers.

    Subclasses implement :meth:`label_one` (mask + InstanceAnnotation for a
    single MediaItem). :meth:`run` handles manifest iteration, mask writing and
    annotation JSONL output.
    """

    name: str = "base"

    def __init__(
        self,
        cfg: Config,
        *,
        project_root: Optional[Path] = None,
        mask_dir: Optional[Path] = None,
        annotation_path: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.project_root = Path(project_root) if project_root else Path.cwd()
        paths = cfg.get("paths", {})
        self.mask_dir = mask_dir or resolve_path(self.project_root, paths.get("masks_raw_dir", "data/masks_raw"))
        self.annotation_path = annotation_path or resolve_path(
            self.project_root, paths.get("annotation_path", "data/annotations/annotations_raw.jsonl")
        )

    @abstractmethod
    def label_one(self, item: MediaItem) -> list:
        """Return a list of :class:`InstanceAnnotation` for ``item``.

        Implementations must write the corresponding mask file to
        ``self.mask_dir`` and set ``mask_path`` on each returned instance.
        """

    def read_manifest(self, manifest_path: Path) -> Iterator[MediaItem]:
        return read_jsonl(manifest_path, model=MediaItem)

    def run(
        self,
        manifest_path: Path,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Run the labeler over the manifest and write annotation JSONL."""
        from ..schemas import InstanceAnnotation  # local to avoid cycles

        records: list[AnnotationRecord] = []
        for item in self.read_manifest(manifest_path):
            if dry_run:
                continue
            instances = self.label_one(item)
            records.append(AnnotationRecord(item_id=item.item_id, instances=list(instances)))

        if dry_run:
            log.info("[%s] dry-run: would label items from %s", self.name, manifest_path)
            return self.annotation_path

        self.mask_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.annotation_path, records, overwrite=True)
        total = sum(len(r.instances) for r in records)
        log.info("[%s] wrote %d records (%d instances) to %s", self.name, len(records), total, self.annotation_path)
        return self.annotation_path
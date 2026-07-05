"""Annotation export helpers (Step 06).

Small utilities to convert / summarize annotation JSONL, used by the review
queue and visualization stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..common.jsonl import read_jsonl_list, write_jsonl
from ..common.logging import get_logger
from ..schemas import AnnotationRecord

log = get_logger("hmp.labeling.export")


def summary(annotation_path: Path) -> dict:
    """Return per-provider/total counts for an annotation JSONL file."""
    records = read_jsonl_list(annotation_path, model=AnnotationRecord)
    by_source: dict[str, int] = {}
    total_instances = 0
    items_with_instances = 0
    for r in records:
        n = len(r.instances)
        total_instances += n
        if n:
            items_with_instances += 1
        for inst in r.instances:
            src = inst.source or "unknown"
            by_source[src] = by_source.get(src, 0) + 1
    return {
        "n_items": len(records),
        "n_items_with_instances": items_with_instances,
        "n_instances": total_instances,
        "by_source": by_source,
    }


def filter_with_instances(annotation_path: Path, out_path: Path) -> int:
    """Write a copy of ``annotation_path`` keeping only items with >=1 instance."""
    records = read_jsonl_list(annotation_path, model=AnnotationRecord)
    kept = [r for r in records if r.instances]
    write_jsonl(out_path, kept, overwrite=True)
    return len(kept)
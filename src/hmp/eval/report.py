"""Evaluation report generation (Step 23, minimal).

Produces a Markdown report summarizing manifest / annotation / refine / quality
counts. Later milestones add boundary, YOLO validation, matting and export
sections; missing sections are reported gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..common.jsonl import count_jsonl, read_jsonl_list
from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..schemas import AnnotationRecord

log = get_logger("hmp.eval.report")


def _section(title: str, rows: list[tuple[str, str]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines += ["_no data available_", ""]
        return lines
    lines += ["| key | value |", "| --- | --- |"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def build_report(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Build a Markdown evaluation report from the available JSONL artifacts."""
    root = Path(project_root) if project_root else Path.cwd()
    paths = cfg.get("paths", {})
    manifest = resolve_path(root, paths.get("manifest_path", "data/manifests/manifest.jsonl"))
    ann_raw = resolve_path(root, paths.get("annotation_path", "data/annotations/annotations_raw.jsonl"))
    ann_refined = resolve_path(root, paths.get("refined_annotation_path", "data/annotations/annotations_refined.jsonl"))
    report_jsonl = resolve_path(root, cfg.get("refine", {}).get("report_path", "data/annotations/refine_report.jsonl"))

    runs_dir = resolve_path(root, paths.get("runs_dir", "runs"))
    out = out_path or runs_dir / "eval_report.md"

    lines = ["# hmp Evaluation Report", ""]

    # manifest
    lines += _section("Manifest", [("items", count_jsonl(manifest) if manifest.exists() else 0)])

    # annotations raw
    inst_raw = 0
    if ann_raw.exists():
        for r in read_jsonl_list(ann_raw, model=AnnotationRecord):
            inst_raw += len(r.instances)
    lines += _section("Annotations (raw)", [("items", count_jsonl(ann_raw) if ann_raw.exists() else 0), ("instances", inst_raw)])

    # annotations refined
    inst_ref = 0
    if ann_refined.exists():
        for r in read_jsonl_list(ann_refined, model=AnnotationRecord):
            inst_ref += len(r.instances)
    lines += _section("Annotations (refined)", [("items", count_jsonl(ann_refined) if ann_refined.exists() else 0), ("instances", inst_ref)])

    # refine report aggregate
    rows = []
    if report_jsonl.exists():
        areas = []
        comps = []
        n = 0
        for line in report_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            n += 1
            areas.append(obj.get("area_after", 0.0))
            comps.append(obj.get("components_after", 0))
        if n:
            import statistics

            rows = [
                ("records", n),
                ("mean_area_after", round(statistics.mean(areas), 4) if areas else 0),
                ("mean_components_after", round(statistics.mean(comps), 2) if comps else 0),
            ]
    lines += _section("Refine report", rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote evaluation report to %s", out)
    return out

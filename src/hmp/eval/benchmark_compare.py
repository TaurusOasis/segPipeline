"""Run multiple COCONut benchmark modes and emit a comparison table."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional

import yaml

from ..common.logging import get_logger
from ..config import Config, resolve_path
from .coconut_benchmark import export_benchmark_review_queue, resummarize_benchmark_dir, run_coconut_benchmark

log = get_logger("hmp.eval.benchmark_compare")

DEFAULT_MODES: tuple[tuple[str, str], ...] = (
    ("gt_bbox", "grabcut"),
    ("gt_bbox", "sam2"),
    ("yolo_person", "grabcut"),
    ("yolo_person", "sam2"),
    ("yolo_person", "samhq"),
)


def _parse_modes(raw: object) -> list[tuple[str, str]]:
    if raw is None:
        return list(DEFAULT_MODES)
    modes: list[tuple[str, str]] = []
    for item in raw:  # type: ignore[union-attr]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            modes.append((str(item[0]), str(item[1])))
        elif isinstance(item, str) and "+" in item:
            det, sam = item.split("+", 1)
            modes.append((det.strip(), sam.strip()))
    return modes or list(DEFAULT_MODES)


def _mode_score(row: dict[str, object]) -> float:
    """Rank modes for label production, balancing quality and cost."""
    iou = float(row.get("mean_mask_iou", 0.0))
    bf1 = float(row.get("mean_boundary_f1", 0.0))
    accept = float(row.get("accept_rate", 0.0))
    reject = float(row.get("reject_rate", 0.0))
    latency_penalty = min(float(row.get("mean_elapsed_ms", 0.0)) / 1000.0, 2.0) * 0.02
    return (0.50 * iou) + (0.30 * bf1) + (0.20 * accept) - (0.10 * reject) - latency_penalty


def _is_oracle_mode(row: dict[str, object]) -> bool:
    return str(row.get("sam_mode", "")) in {"oracle", "noisy_oracle"}


def _apply_production_preference(
    rows: list[dict[str, object]],
    compare_cfg: object,
) -> dict[str, object] | None:
    """Prefer end-to-end detector when its score is within margin of the best mode."""
    if not rows:
        return None
    prefer_det = "yolo_person"
    margin = 0.03
    if hasattr(compare_cfg, "get"):
        prefer_det = str(compare_cfg.get("prefer_production_detector", prefer_det))
        margin = float(compare_cfg.get("production_score_margin", margin))
    best_score = float(rows[0]["mode_score"])
    pool = [row for row in rows if float(row["mode_score"]) >= best_score - margin]
    for sam_mode in ("sam2", "samhq", "grabcut"):
        for row in pool:
            if row["detector_mode"] == prefer_det and row["sam_mode"] == sam_mode:
                return row
    for row in pool:
        if row["detector_mode"] == prefer_det:
            return row
    return rows[0]


def _select_best_mode(rows: list[dict[str, object]], compare_cfg: object) -> tuple[dict[str, object] | None, dict[str, object]]:
    allow_oracle = bool(compare_cfg.get("allow_oracle_selection", False)) if hasattr(compare_cfg, "get") else False
    selectable = rows if allow_oracle else [row for row in rows if not _is_oracle_mode(row)]
    if not selectable:
        selectable = rows
    ranked = sorted(selectable, key=lambda r: float(r["mode_score"]), reverse=True)
    best = _apply_production_preference(ranked, compare_cfg)
    policy = {
        "allow_oracle_selection": allow_oracle,
        "oracle_modes_excluded": bool(rows and not allow_oracle and any(_is_oracle_mode(row) for row in rows) and selectable != rows),
        "selection_pool_size": len(selectable),
        "prefer_production_detector": (
            str(compare_cfg.get("prefer_production_detector", "yolo_person"))
            if hasattr(compare_cfg, "get")
            else "yolo_person"
        ),
        "production_score_margin": (
            float(compare_cfg.get("production_score_margin", 0.03))
            if hasattr(compare_cfg, "get")
            else 0.03
        ),
        "top_mode_score": float(ranked[0]["mode_score"]) if ranked else 0.0,
    }
    return best, policy


def _next_actions(best_summary: dict[str, object]) -> list[str]:
    buckets = best_summary.get("error_buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}
    actions: list[str] = []
    if buckets.get("detector_miss", 0):
        actions.append("Tune person detector first: lower yolo_conf, add GroundingDINO fallback, and compare detector recall.")
    if buckets.get("background_leak", 0):
        actions.append("Add negative-point prompts around background/nearby people; keep branch_source for identity-leak audits.")
    if buckets.get("missed_foreground", 0):
        actions.append("Collect correction prompts for missed foreground and replay them through SAM2 correction.")
    if buckets.get("bad_boundary", 0):
        actions.append("Route bad-boundary ROI to HQ-SAM or Bd diffusion refine before alpha training.")
    if buckets.get("needs_scribble", 0):
        actions.append("Send only needs_scribble records to HITL and store scribbles as prompt-agent training data.")
    return actions or ["Increase COCONut sample limit and add harder buckets; current selected mode has no dominant failure bucket."]


def _config_patch(best: dict[str, object], best_summary: dict[str, object]) -> dict[str, object]:
    gates = best_summary.get("quality_gates", {})
    patch: dict[str, object] = {
        "coconut_benchmark": {
            "detector_mode": best["detector_mode"],
            "sam_mode": best["sam_mode"],
            "quality_gates": gates,
            "write_masks": True,
        },
        "labeling": {
            "segment_mode": best["sam_mode"],
            "quality_gates": gates,
        },
    }
    buckets = best_summary.get("error_buckets", {})
    if isinstance(buckets, dict) and buckets.get("background_leak", 0):
        patch["local_postprocess"] = {"keep_largest_component": True}
    return patch


def _compare_markdown(rows: list[dict[str, object]], best: dict[str, object] | None, plan: dict[str, object]) -> str:
    header = "| rank | detector | sam | instances | score | mask IoU | boundary F1 | accept | review | reject | ms/inst |"
    sep = "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = []
    for idx, row in enumerate(rows, start=1):
        body.append(
            f"| {idx} | `{row['detector_mode']}` | `{row['sam_mode']}` | {row['instances']} | "
            f"{float(row['mode_score']):.4f} | {float(row['mean_mask_iou']):.4f} | "
            f"{float(row['mean_boundary_f1']):.4f} | {float(row.get('accept_rate', 0.0)):.2%} | "
            f"{float(row.get('review_rate', 0.0)):.2%} | {float(row.get('reject_rate', 0.0)):.2%} | "
            f"{float(row['mean_elapsed_ms']):.1f} |"
        )
    lines = ["# COCONut Benchmark Compare", "", header, sep, *body, ""]
    if best is not None:
        policy = plan.get("selection_policy", {})
        lines += [
            "## Selected Next Mode",
            "",
            f"- detector: `{best['detector_mode']}`",
            f"- sam: `{best['sam_mode']}`",
            f"- score: **{float(best['mode_score']):.4f}**",
            f"- output: `{best['output_dir']}`",
            f"- oracle excluded: **{bool(policy.get('oracle_modes_excluded', False))}**",
            "",
            "## Iteration Actions",
            "",
        ]
        lines += [f"- {a}" for a in plan.get("next_actions", [])]
        lines += ["", "Patch: `next_config_patch.yaml`", "Plan: `iteration_plan.json`", ""]
    return "\n".join(lines)


def _summary_is_complete(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(summary.get("decision_counts")) and int(summary.get("instances", 0)) > 0


def run_coconut_compare(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    bcfg = cfg.get("coconut_benchmark", {})
    compare_cfg = cfg.get("coconut_compare", {})
    modes = _parse_modes(compare_cfg.get("modes", bcfg.get("compare_modes")))
    out_root = resolve_path(root, compare_cfg.get("output_dir", bcfg.get("compare_output_dir", "runs/coconut_compare")))
    compare_json = out_root / "compare_summary.json"
    compare_md = out_root / "compare_summary.md"
    iteration_json = out_root / "iteration_plan.json"
    config_patch_yaml = out_root / "next_config_patch.yaml"

    if dry_run:
        log.info("[dry-run] would compare %d modes -> %s", len(modes), out_root)
        return compare_md

    out_root.mkdir(parents=True, exist_ok=True)
    skip_existing = bool(compare_cfg.get("skip_existing", True))
    force_rerun = bool(compare_cfg.get("force_rerun", False))
    rows: list[dict[str, object]] = []
    for detector_mode, sam_mode in modes:
        mode_dir = out_root / f"{detector_mode}__{sam_mode}"
        summary_path = mode_dir / "benchmark_summary.json"
        records_path = mode_dir / "benchmark_records.jsonl"
        if skip_existing and not force_rerun and _summary_is_complete(summary_path):
            log.info("reuse completed mode %s+%s from %s", detector_mode, sam_mode, summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        elif records_path.exists() and not force_rerun:
            log.info("resummarize legacy mode %s+%s from %s", detector_mode, sam_mode, records_path)
            gates = cfg.get("coconut_benchmark", {}).get("quality_gates", {})
            if hasattr(gates, "to_dict"):
                gates = gates.to_dict()
            worst_k = int(cfg.get("coconut_benchmark", {}).get("worst_k", 10))
            summary = resummarize_benchmark_dir(
                mode_dir,
                quality_gates={k: float(v) for k, v in gates.items()} if gates else None,
                worst_k=worst_k,
            )
        else:
            run_cfg = deepcopy(cfg.to_dict())
            run_cfg.setdefault("coconut_benchmark", {})
            run_cfg["coconut_benchmark"]["detector_mode"] = detector_mode
            run_cfg["coconut_benchmark"]["sam_mode"] = sam_mode
            run_cfg["coconut_benchmark"]["output_dir"] = str(mode_dir)
            child_cfg = Config(run_cfg)
            run_coconut_benchmark(child_cfg, project_root=root, dry_run=False)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "detector_mode": detector_mode,
                "sam_mode": sam_mode,
                "instances": summary.get("instances", 0),
                "mean_mask_iou": summary.get("mean_mask_iou", 0.0),
                "mean_boundary_f1": summary.get("mean_boundary_f1", 0.0),
                "mean_bbox_iou": summary.get("mean_bbox_iou", 0.0),
                "accept_rate": summary.get("accept_rate", 0.0),
                "review_rate": summary.get("review_rate", 0.0),
                "reject_rate": summary.get("reject_rate", 0.0),
                "decision_counts": summary.get("decision_counts", {}),
                "error_buckets": summary.get("error_buckets", {}),
                "mean_elapsed_ms": summary.get("mean_elapsed_ms", 0.0),
                "instances_per_second": summary.get("instances_per_second", 0.0),
                "output_dir": str(summary_path.parent),
                "summary_path": str(summary_path),
                "mode_score": 0.0,
            }
        )
        export_path = mode_dir / "review_queue.jsonl"
        export_benchmark_review_queue(mode_dir, review_path=export_path)

    for row in rows:
        row["mode_score"] = _mode_score(row)
    rows = sorted(rows, key=lambda r: float(r["mode_score"]), reverse=True)
    best, selection_policy = _select_best_mode(rows, compare_cfg)
    best_summary: dict[str, object] = {}
    if best is not None:
        best_summary = json.loads(Path(str(best["summary_path"])).read_text(encoding="utf-8"))
    plan = {
        "selected_mode": best,
        "selection_policy": selection_policy,
        "next_actions": _next_actions(best_summary) if best else [],
        "next_config_patch": _config_patch(best, best_summary) if best else {},
        "ranked_modes": rows,
    }
    compare_json.write_text(json.dumps({"modes": rows, "selected_mode": best}, indent=2), encoding="utf-8")
    iteration_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    config_patch_yaml.write_text(yaml.safe_dump(plan["next_config_patch"], sort_keys=False), encoding="utf-8")
    compare_md.write_text(_compare_markdown(rows, best, plan), encoding="utf-8")
    log.info("COCONut compare: %d modes -> %s", len(rows), compare_md)
    return compare_md

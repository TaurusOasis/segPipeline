"""hmp command-line interface (Typer).

The CLI is the single registry of commands. Each command callback lazy-imports
its implementation module *inside* the callback body, so importing ``hmp.cli``
never pulls heavy / GPU dependencies. Heavy work lives in domain modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .common.logging import configure_logging, get_logger

app = typer.Typer(
    name="hmp",
    help="Human matting pipeline: data -> labels -> refine -> YOLO -> matting -> export.",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------
@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only show warnings+."),
) -> None:
    level = "DEBUG" if verbose else ("WARNING" if quiet else "INFO")
    configure_logging(level)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the hmp version and exit."""
    typer.echo(__version__)


# ---------------------------------------------------------------------------
# config group (Step 00)
# ---------------------------------------------------------------------------
config_app = typer.Typer(help="Inspect loaded configuration.")
app.add_typer(config_app, name="config")


@config_app.command("models")
def config_models(
    config: Path | None = typer.Option(None, "--config", "-c", help="Optional config overlay."),
) -> None:
    """Print edge vs GPU teacher model tiers."""
    from .config import Config, load_config
    from .models.tiers import load_model_tiers

    cfg = load_config(config) if config else Config({})
    registry = load_model_tiers(cfg, project_root=Path.cwd())
    payload = {
        "edge": {
            "name": registry.edge.name,
            "role": registry.edge.role,
            "weights": registry.edge.weights,
            "deploy_target": registry.edge.deploy_target,
        },
        "teachers": {
            key: {
                "kind": spec.kind,
                "backend": spec.backend,
                "weights": spec.weights,
                "notes": spec.notes,
            }
            for key, spec in registry.teachers.items()
        },
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@config_app.command("show")
def config_show(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML config file."),
) -> None:
    """Load and print the resolved config as JSON."""
    from .config import load_config

    cfg = load_config(config)
    typer.echo(json.dumps(cfg.to_dict(), indent=2, default=str))


def _load_cfg(config: Optional[Path]) -> "object":
    """Helper used by command callbacks to load config + return Config + root."""
    from .config import load_config

    if config is None:
        raise typer.BadParameter("--config is required")
    cfg = load_config(config)
    get_logger("hmp.cli").info("Loaded config: %s", config)
    return cfg


# ---------------------------------------------------------------------------
# manifest group (Step 02)
# ---------------------------------------------------------------------------
manifest_app = typer.Typer(help="Build and inspect the data manifest.")
app.add_typer(manifest_app, name="manifest")


@manifest_app.command("build")
def manifest_build(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would be written."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Rebuild even if manifest exists."),
) -> None:
    """Scan image directories and build the manifest JSONL."""
    from .data.build_manifest import build_manifest

    cfg = _load_cfg(config)
    out = build_manifest(cfg, project_root=Path.cwd(), dry_run=dry_run, overwrite=overwrite)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# frames group (Step 03)
# ---------------------------------------------------------------------------
frames_app = typer.Typer(help="Extract frames from videos.")
app.add_typer(frames_app, name="frames")


@frames_app.command("extract")
def frames_extract(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Rebuild manifest instead of appending."),
) -> None:
    """Extract video frames and update the manifest."""
    from .data.extract_frames import extract_frames

    cfg = _load_cfg(config)
    out = extract_frames(cfg, project_root=Path.cwd(), dry_run=dry_run, overwrite=overwrite)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# label group (Step 06+)
# ---------------------------------------------------------------------------
label_app = typer.Typer(help="Automatic person-mask labeling.")
app.add_typer(label_app, name="label")


@label_app.command("dummy")
def label_dummy(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the dummy centered-rectangle labeler."""
    from .labeling.dummy_labeler import DummyLabeler

    cfg = _load_cfg(config)
    from .config import resolve_path

    manifest_path = resolve_path(Path.cwd(), cfg.get("paths", {}).get("manifest_path", "data/manifests/manifest.jsonl"))
    labeler = DummyLabeler(cfg, project_root=Path.cwd())
    out = labeler.run(manifest_path, dry_run=dry_run, overwrite=True)
    typer.echo(str(out))


@label_app.command("yolo-sam2")
def label_yolo_sam2(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    segment_mode: str = typer.Option("sam2", "--segment-mode", help="sam2 | samhq | grabcut"),
    teacher: str = typer.Option("", "--teacher", help="Teacher key from models.yaml (sam2, samhq)."),
) -> None:
    """Run YOLO edge detect + GPU segment teacher (SAM2/SamHQ) or GrabCut ablation."""
    from .config import resolve_path
    from .labeling.yolo_sam2_labeler import YoloSam2Labeler

    cfg = _load_cfg(config)
    manifest_path = resolve_path(Path.cwd(), cfg.get("paths", {}).get("manifest_path", "data/manifests/manifest.jsonl"))
    teacher_key = teacher or (segment_mode if segment_mode in {"sam2", "samhq", "grabcut"} else "sam2")
    labeler = YoloSam2Labeler(
        cfg,
        project_root=Path.cwd(),
        segment_mode=segment_mode,  # type: ignore[arg-type]
        teacher_key=teacher_key,
    )
    out = labeler.run(manifest_path, dry_run=dry_run, overwrite=True)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# dataset group (Step 07)
# ---------------------------------------------------------------------------
dataset_app = typer.Typer(help="Export datasets in target formats.")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("export-yolo")
def dataset_export_yolo(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Export project masks to a YOLO segmentation dataset."""
    from .yolo.export_yolo_dataset import export_yolo_dataset

    cfg = _load_cfg(config)
    out = export_yolo_dataset(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@dataset_app.command("ingest")
def dataset_ingest(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Enrich manifest with dataset registry metadata (pipeline step 0)."""
    from .data.ingest import enrich_manifest_with_dataset

    cfg = _load_cfg(config)
    out = enrich_manifest_with_dataset(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# yolo group (distillation bridge)
# ---------------------------------------------------------------------------
yolo_app = typer.Typer(help="YOLO export and distillation bridge to ultralytics/.")
app.add_typer(yolo_app, name="yolo")


@yolo_app.command("export-accept-coconut")
def yolo_export_accept_coconut(
    config: Path = typer.Option(..., "--config", "-c"),
    benchmark_dir: Path | None = typer.Option(None, "--benchmark-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Patch COCONut YOLO val labels with benchmark accept masks."""
    from .yolo.coconut_distill_bridge import overlay_accept_labels_on_coconut

    cfg = _load_cfg(config)
    out = overlay_accept_labels_on_coconut(
        cfg,
        project_root=Path.cwd(),
        benchmark_dir=benchmark_dir,
        dry_run=dry_run,
    )
    typer.echo(str(out))


@yolo_app.command("distill-plan")
def yolo_distill_plan(
    config: Path = typer.Option(..., "--config", "-c"),
    data_yaml: Path | None = typer.Option(None, "--data-yaml", help="Optional overlay data.yaml path."),
) -> None:
    """Print ultralytics COCONut distillation command for accept-overlay labels."""
    from .yolo.coconut_distill_bridge import build_coconut_distill_plan

    cfg = _load_cfg(config)
    plan = build_coconut_distill_plan(cfg, project_root=Path.cwd(), data_yaml=data_yaml)
    typer.echo(json.dumps(plan, indent=2))


@dataset_app.command("stratify")
def dataset_stratify(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Assign stratification tags and dedup clusters (pipeline step 1)."""
    from .data.stratify import stratify_manifest

    cfg = _load_cfg(config)
    out = stratify_manifest(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@dataset_app.command("coconut-sample")
def dataset_coconut_sample(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    overwrite: bool = typer.Option(True, "--overwrite", help="Replace existing manifest."),
) -> None:
    """Sample COCONut val images into manifest JSONL (pipeline step 0)."""
    from .data.coconut_sample import build_coconut_manifest

    cfg = _load_cfg(config)
    out = build_coconut_manifest(cfg, project_root=Path.cwd(), dry_run=dry_run, overwrite=overwrite)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# refine group (Step 12 local-only minimal)
# ---------------------------------------------------------------------------
refine_app = typer.Typer(help="Refine masks (local postprocess + external refiners).")
app.add_typer(refine_app, name="refine")


@refine_app.command("masks")
def refine_masks(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run local mask refinement."""
    from .refine.refine_pipeline import refine_masks as _run

    cfg = _load_cfg(config)
    ann, report = _run(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(f"{ann}\t{report}")


# ---------------------------------------------------------------------------
# matting group (Step 17 trimap minimal)
# ---------------------------------------------------------------------------
matting_app = typer.Typer(help="Matting: trimaps, alpha teachers, student training.")
app.add_typer(matting_app, name="matting")


@matting_app.command("make-trimap")
def matting_make_trimap(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate trimaps from refined masks."""
    from .matting.trimap import make_trimap_from_annotation

    cfg = _load_cfg(config)
    out = make_trimap_from_annotation(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@matting_app.command("make-adaptive-trimap")
def matting_make_adaptive_trimap(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate adaptive trimaps and ROI sidecars (pipeline step 6)."""
    from .matting.adaptive_trimap import make_adaptive_trimap_from_annotation

    cfg = _load_cfg(config)
    out = make_adaptive_trimap_from_annotation(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@matting_app.command("fuse-alpha")
def matting_fuse_alpha(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Fuse multi-teacher alpha branches for queued tasks (pipeline step 9)."""
    from .matting.process_queue import process_relabel_queue

    cfg = _load_cfg(config)
    out = process_relabel_queue(cfg, project_root=Path.cwd(), dry_run=dry_run, provider="mock")
    typer.echo(str(out))


@matting_app.command("alpha-teacher")
def matting_alpha_teacher(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    provider: str = typer.Option("mock", "--provider", help="mock or external"),
) -> None:
    """Generate Bv/Bi/Bd/Bs branch alphas for queued tasks (pipeline step 7)."""
    from .matting.process_queue import process_relabel_queue

    cfg = _load_cfg(config)
    out = process_relabel_queue(cfg, project_root=Path.cwd(), dry_run=dry_run, provider=provider)
    typer.echo(str(out))


@matting_app.command("process-queue")
def matting_process_queue(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    provider: str = typer.Option("mock", "--provider", help="mock or external"),
) -> None:
    """Run steps 7-9: branch alphas, MQE, and fusion for relabel queue."""
    from .matting.process_queue import process_relabel_queue

    cfg = _load_cfg(config)
    out = process_relabel_queue(cfg, project_root=Path.cwd(), dry_run=dry_run, provider=provider)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# relabel group (mask-to-matte queue)
# ---------------------------------------------------------------------------
relabel_app = typer.Typer(help="Build mask-to-matte relabeling queues.")
app.add_typer(relabel_app, name="relabel")


@relabel_app.command("queue")
def relabel_queue(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite"),
) -> None:
    """Build a JSONL queue for alpha relabeling from masks and trimaps."""
    from .matting.relabel_queue import build_relabel_queue

    cfg = _load_cfg(config)
    out = build_relabel_queue(cfg, project_root=Path.cwd(), dry_run=dry_run, overwrite=overwrite)
    typer.echo(str(out))


@relabel_app.command("hitl-queue")
def relabel_hitl_queue(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Export HITL review items for failed/low-quality regions (pipeline step 10)."""
    from .matting.hitl_queue import build_hitl_queue

    cfg = _load_cfg(config)
    out = build_hitl_queue(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@relabel_app.command("export-labels")
def relabel_export_labels(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Export final alpha label manifest (pipeline step 11)."""
    from .matting.export_labels import export_alpha_labels

    cfg = _load_cfg(config)
    out = export_alpha_labels(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# eval group (Step 23 report minimal)
# ---------------------------------------------------------------------------
eval_app = typer.Typer(help="Evaluation reports.")
app.add_typer(eval_app, name="eval")


@eval_app.command("report")
def eval_report(
    config: Path = typer.Option(..., "--config", "-c"),
) -> None:
    """Build a Markdown evaluation report from available artifacts."""
    from .eval.report import build_report

    cfg = _load_cfg(config)
    out = build_report(cfg, project_root=Path.cwd())
    typer.echo(str(out))


@eval_app.command("mqe")
def eval_mqe(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run rule-based MQE / QA scoring (pipeline step 8)."""
    from .eval.mqe import evaluate_from_config

    cfg = _load_cfg(config)
    out = evaluate_from_config(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@eval_app.command("alpha-metrics")
def eval_alpha_metrics(
    pred_dir: Path = typer.Option(..., "--pred-dir", help="Directory of predicted alpha mattes (PNG, [0,1])."),
    gt_dir: Path = typer.Option(..., "--gt-dir", help="Directory of GT alpha mattes (PNG, [0,1])."),
    trimap_dir: Optional[Path] = typer.Option(None, "--trimap-dir", help="Optional trimap dir; 0/128/255 or bool unknown."),
    with_connectivity: bool = typer.Option(False, "--connectivity", help="Also compute Xu et al. connectivity error (slower)."),
    out: Path = typer.Option(Path("alpha_metrics.json"), "--out", help="Where to write the JSON summary."),
) -> None:
    """Compute SAD/MAD/MSE/gradient (and optionally connectivity) over matched alpha pairs.

    Files are matched by stem (e.g. ``a_0001.png`` in pred matches ``a_0001.png``
    in gt). Unmatched files are skipped and reported.
    """
    import json as _json

    import numpy as np
    from PIL import Image

    from .eval.alpha_metrics import aggregate_alpha_metrics

    def _load_gray(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0

    def _load_trimap(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert("L"))

    pred_paths = {p.stem: p for p in sorted(pred_dir.glob("*.png"))}
    gt_paths = {p.stem: p for p in sorted(gt_dir.glob("*.png"))}
    common = sorted(set(pred_paths) & set(gt_paths))
    missing_pred = sorted(set(gt_paths) - set(pred_paths))
    missing_gt = sorted(set(pred_paths) - set(gt_paths))
    if not common:
        typer.echo(f"No matched alpha pairs between {pred_dir} and {gt_dir}", err=True)
        raise typer.Exit(1)

    preds, gts, trimaps = [], [], []
    for stem in common:
        preds.append(_load_gray(pred_paths[stem]))
        gts.append(_load_gray(gt_paths[stem]))
        if trimap_dir is not None:
            tp = trimap_dir / f"{stem}.png"
            trimaps.append(_load_trimap(tp) if tp.exists() else None)
    tri_list = trimaps if trimap_dir is not None else None

    summary = aggregate_alpha_metrics(preds, gts, tri_list, with_connectivity=with_connectivity)
    summary["n_pairs"] = len(common)
    summary["missing_in_pred"] = missing_pred
    summary["missing_in_gt"] = missing_gt
    out.write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"alpha-metrics: {len(common)} pairs -> {out}")
    typer.echo(_json.dumps({k: v for k, v in summary.items() if k.startswith(("sad", "mad", "mse", "grad", "conn"))}, indent=2))


@eval_app.command("coconut-benchmark")
def eval_coconut_benchmark(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run COCONut auto-label vs GT benchmark."""
    from .eval.coconut_benchmark import run_coconut_benchmark

    cfg = _load_cfg(config)
    out = run_coconut_benchmark(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@eval_app.command("coconut-compare")
def eval_coconut_compare(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run multi-mode COCONut benchmark comparison (detector x SAM)."""
    from .eval.benchmark_compare import run_coconut_compare

    cfg = _load_cfg(config)
    out = run_coconut_compare(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@eval_app.command("coconut-iterate")
def eval_coconut_iterate(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force_rerun: bool = typer.Option(False, "--force-rerun", help="Re-run all benchmark modes even if cached."),
) -> None:
    """Run COCONut AI-label comparison and emit an iteration plan."""
    from .eval.benchmark_compare import run_coconut_compare

    cfg = _load_cfg(config)
    if force_rerun:
        cfg_dict = cfg.to_dict()
        cfg_dict.setdefault("coconut_compare", {})
        cfg_dict["coconut_compare"]["force_rerun"] = True
        from .config import Config

        cfg = Config(cfg_dict)
    out = run_coconut_compare(cfg, project_root=Path.cwd(), dry_run=dry_run)
    typer.echo(str(out))


@eval_app.command("coconut-resummarize")
def eval_coconut_resummarize(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Existing benchmark output directory."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Optional config for quality gates."),
) -> None:
    """Backfill QA fields and rebuild benchmark summary from records JSONL."""
    from .config import Config, load_config
    from .eval.coconut_benchmark import resummarize_benchmark_dir

    gates = None
    worst_k = 10
    if config is not None:
        cfg = load_config(config)
        raw = cfg.get("coconut_benchmark", {}).get("quality_gates", {})
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        gates = {k: float(v) for k, v in raw.items()} if raw else None
        worst_k = int(cfg.get("coconut_benchmark", {}).get("worst_k", 10))
    summary = resummarize_benchmark_dir(benchmark_dir, quality_gates=gates, worst_k=worst_k)
    typer.echo(json.dumps({"instances": summary["instances"], "accept_rate": summary["accept_rate"]}, indent=2))


@eval_app.command("coconut-export-review")
def eval_coconut_export_review(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Benchmark output directory."),
) -> None:
    """Export review/reject instances from a benchmark run."""
    from .eval.coconut_benchmark import export_benchmark_review_queue

    out = export_benchmark_review_queue(benchmark_dir)
    typer.echo(str(out))


@eval_app.command("coconut-visualize")
def eval_coconut_visualize(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Benchmark output directory."),
    max_items: int = typer.Option(12, "--max-items", help="Number of worst records to draw."),
    tile_width: int = typer.Option(220, "--tile-width", help="Width of each visual tile."),
) -> None:
    """Build a worst-case contact sheet from benchmark masks."""
    from .eval.coconut_benchmark import write_benchmark_contact_sheet

    out = write_benchmark_contact_sheet(benchmark_dir, max_items=max_items, tile_width=tile_width)
    typer.echo(str(out))


@eval_app.command("coconut-import-annotations")
def eval_coconut_import_annotations(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Benchmark output directory."),
    config: Path = typer.Option(..., "--config", "-c"),
    overwrite: bool = typer.Option(True, "--overwrite"),
    decisions: str = typer.Option("", "--decisions", help="Comma-separated accept/review/reject filter."),
) -> None:
    """Import benchmark predictions into pipeline annotation path."""
    from .config import resolve_path
    from .eval.benchmark_bridge import import_benchmark_annotations

    cfg = _load_cfg(config)
    ann_path = resolve_path(
        Path.cwd(),
        cfg.get("paths", {}).get("annotation_path", "data/annotations/annotations_raw.jsonl"),
    )
    decision_tuple = tuple(d.strip() for d in decisions.split(",") if d.strip()) or None
    out = import_benchmark_annotations(
        benchmark_dir,
        annotation_path=ann_path,
        overwrite=overwrite,
        decisions=decision_tuple,
    )
    typer.echo(str(out))


@eval_app.command("coconut-export-hitl")
def eval_coconut_export_hitl(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Benchmark output directory."),
    config: Path = typer.Option(..., "--config", "-c"),
) -> None:
    """Export benchmark review/reject rows into HITL queue JSONL."""
    from .config import resolve_path
    from .eval.benchmark_bridge import benchmark_review_to_hitl

    cfg = _load_cfg(config)
    hitl_path = resolve_path(
        Path.cwd(),
        cfg.get("hitl", {}).get("queue_path", "data/hitl/review_queue.jsonl"),
    )
    out = benchmark_review_to_hitl(benchmark_dir, hitl_path=hitl_path)
    typer.echo(str(out))


@eval_app.command("coconut-export-bad-boundary")
def eval_coconut_export_bad_boundary(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Benchmark output directory."),
    out: Path | None = typer.Option(None, "--out", help="Optional output JSONL path."),
    decisions: str = typer.Option(
        "review,reject,accept",
        "--decisions",
        help="Comma-separated decisions to include.",
    ),
) -> None:
    """Export bad_boundary instances for SamHQ boundary re-label."""
    from .eval.benchmark_bridge import export_bad_boundary_queue

    decision_tuple = tuple(d.strip() for d in decisions.split(",") if d.strip())
    path = export_bad_boundary_queue(
        benchmark_dir,
        queue_path=out,
        include_decisions=decision_tuple or ("review", "reject", "accept"),
    )
    typer.echo(str(path))


@eval_app.command("coconut-relabel-boundary")
def eval_coconut_relabel_boundary(
    benchmark_dir: Path = typer.Option(..., "--benchmark-dir", help="Benchmark output directory."),
    config: Path = typer.Option(..., "--config", "-c"),
    teacher: str = typer.Option("samhq", "--teacher", help="Boundary teacher key from models.yaml."),
    max_instances: int = typer.Option(0, "--max-instances", help="Limit re-label count (0 = all)."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    decisions: str = typer.Option(
        "review,reject,accept",
        "--decisions",
        help="Comma-separated decisions eligible for re-label.",
    ),
) -> None:
    """Re-run SamHQ on bad_boundary benchmark instances and patch outputs."""
    from .eval.benchmark_bridge import relabel_bad_boundary_instances

    cfg = _load_cfg(config)
    decision_tuple = tuple(d.strip() for d in decisions.split(",") if d.strip())
    stats = relabel_bad_boundary_instances(
        cfg,
        benchmark_dir,
        teacher_key=teacher,
        include_decisions=decision_tuple or ("review", "reject", "accept"),
        max_instances=max_instances or None,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(stats, indent=2))


@eval_app.command("coconut-apply-patch")
def eval_coconut_apply_patch(
    config: Path = typer.Option(..., "--config", "-c"),
    patch: Path = typer.Option(..., "--patch", help="Path to next_config_patch.yaml from coconut-iterate."),
    out: Path | None = typer.Option(None, "--out", help="Merged config output path."),
) -> None:
    """Merge coconut-iterate config patch into the base relabel config."""
    from .eval.benchmark_bridge import apply_iteration_patch

    cfg = _load_cfg(config)
    merged = apply_iteration_patch(cfg, patch, out_path=out)
    typer.echo(str(merged))


# ---------------------------------------------------------------------------
# agents group
# ---------------------------------------------------------------------------
agents_app = typer.Typer(help="RL / heuristic prompt and fusion agents.")
app.add_typer(agents_app, name="agents")


@agents_app.command("prompt")
def agents_prompt(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Plan prompts for queued tasks (pipeline step 3, dry-run scaffold)."""
    from .agents.prompt_agent import plan_prompts

    cfg = _load_cfg(config)
    _ = cfg
    decision = plan_prompts(bbox_xyxy=[10, 10, 100, 200], width=640, height=480)
    typer.echo(json.dumps({"dry_run": dry_run, "decision": decision.__dict__}, default=str))


@agents_app.command("fusion")
def agents_fusion(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run fusion agent on processed queue (pipeline step 9 via process-queue)."""
    from .matting.process_queue import process_relabel_queue

    cfg = _load_cfg(config)
    out = process_relabel_queue(cfg, project_root=Path.cwd(), dry_run=dry_run, provider="mock")
    typer.echo(str(out))


# ---------------------------------------------------------------------------
# pipeline group
# ---------------------------------------------------------------------------
pipeline_app = typer.Typer(help="Inspect the 12-stage relabeling pipeline.")
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("stages")
def pipeline_stages() -> None:
    """Print the canonical 0-11 pipeline stages."""
    from .pipeline.stages import PIPELINE_STAGES

    for stage in PIPELINE_STAGES:
        typer.echo(f"{stage.index}\t{stage.name}\t{stage.title}")


@pipeline_app.command("run-relabel")
def pipeline_run_relabel(
    config: Path = typer.Option(..., "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    provider: str = typer.Option(
        "mock",
        "--provider",
        help="mock | yolo_grabcut | yolo_sam2 | yolo_samhq (edge=yolo26s-seg, teacher=GPU SAM/SamHQ)",
    ),
    from_stage: int = typer.Option(0, "--from-stage"),
    to_stage: int = typer.Option(11, "--to-stage"),
) -> None:
    """Run the end-to-end 12-stage relabeling pipeline."""
    from .pipeline.run_relabel import run_relabel_pipeline

    cfg = _load_cfg(config)
    results = run_relabel_pipeline(
        cfg,
        project_root=Path.cwd(),
        dry_run=dry_run,
        provider=provider,
        from_stage=from_stage,
        to_stage=to_stage,
    )
    for index, name, out in results:
        typer.echo(f"{index}\t{name}\t{out}")


@pipeline_app.command("bootstrap-from-benchmark")
def pipeline_bootstrap_from_benchmark(
    config: Path = typer.Option(..., "--config", "-c"),
    benchmark_dir: Path | None = typer.Option(None, "--benchmark-dir", help="Override coconut_bridge.benchmark_dir."),
) -> None:
    """Import COCONut benchmark manifest/annotations and export HITL queue."""
    from .eval.benchmark_bridge import bootstrap_from_benchmark

    cfg = _load_cfg(config)
    out = bootstrap_from_benchmark(cfg, project_root=Path.cwd(), benchmark_dir=benchmark_dir)
    for key, path in out.items():
        typer.echo(f"{key}\t{path}")


# ---------------------------------------------------------------------------
# adapters group (external research-repo subprocess adapters, Step 4/5/6/7)
# ---------------------------------------------------------------------------
adapters_app = typer.Typer(help="Inspect and dry-run external research-repo adapters.")
app.add_typer(adapters_app, name="adapters")


def _parse_kv(items: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``--flag key=value`` strings into a dict."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"{flag} value must be 'key=value', got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@adapters_app.command("list")
def adapters_list(
    group: Optional[str] = typer.Option(None, "--group", help="Filter by group (masklet, mask_refine, ...)."),
) -> None:
    """List all registered external-repo adapter integrations."""
    from .adapters import load_registry

    reg = load_registry()
    specs = reg.by_group(group) if group else [reg.specs[n] for n in sorted(reg.names())]
    for spec in specs:
        typer.echo(
            f"{spec.name}\t{spec.group}\tp{spec.priority}\t{spec.license_review}\t"
            f"{','.join(spec.expected_outputs)}"
        )


@adapters_app.command("dry-run")
def adapters_dry_run(
    name: str = typer.Option(..., "--name", help="Integration name (e.g. samrefiner)."),
    input: list[str] = typer.Option(None, "--input", help="key=value input mapping (repeatable)."),
    output: list[str] = typer.Option(None, "--output", help="key=value output mapping (repeatable)."),
    param: list[str] = typer.Option(None, "--param", help="key=value param mapping (repeatable)."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Adapter workdir (default runs/adapters/<name>)."),
    command_template: Optional[str] = typer.Option(
        None, "--command-template", help="Override the default command template (comma-separated argv)."
    ),
) -> None:
    """Resolve an adapter's command + env + outputs without executing it.

    Inputs/outputs/params use ``key=value`` repeated flags, e.g.
    ``--input image=a.png --output refined_mask=out.png``.
    """
    import json as _json

    from .adapters import build_adapter

    if workdir is None:
        workdir = Path("runs/adapters") / name
    inputs = _parse_kv(input, "--input")
    outputs = _parse_kv(output, "--output")
    params = _parse_kv(param, "--param")
    tmpl = command_template.split(",") if command_template else None
    adapter = build_adapter(
        name,
        workdir=str(workdir),
        command_template=tmpl,
    )
    # Auto-supply the default repo python if the template references it.
    full_params = dict(params)
    if "{param_repo_python}" in " ".join(adapter.command_template):
        full_params.setdefault("repo_python", adapter.env_overlay.get("REPO_PYTHON", "python"))
    res = adapter.dry_run(inputs, outputs, params=full_params)
    typer.echo(_json.dumps({
        "name": res.name,
        "command": res.command,
        "env": res.env,
        "outputs": res.outputs,
        "dry_run": res.dry_run,
        "spec": {
            "group": adapter.spec.group,
            "url": adapter.spec.url,
            "adapter_target": adapter.spec.adapter_target,
            "expected_outputs": adapter.spec.expected_outputs,
            "license_review": adapter.spec.license_review,
        },
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()

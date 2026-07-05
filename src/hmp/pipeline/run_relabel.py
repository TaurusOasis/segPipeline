"""Run the full 12-stage mask-to-matte relabeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..common.logging import get_logger
from ..config import Config, resolve_path
from ..data.build_manifest import build_manifest
from ..data.ingest import enrich_manifest_with_dataset
from ..data.stratify import stratify_manifest
from ..labeling.labeler_factory import make_labeler
from ..matting.adaptive_trimap import make_adaptive_trimap_from_annotation
from ..matting.export_labels import export_alpha_labels
from ..matting.hitl_queue import build_hitl_queue
from ..matting.process_queue import process_relabel_queue
from ..matting.relabel_queue import build_relabel_queue
from ..refine.refine_pipeline import refine_masks

log = get_logger("hmp.pipeline.run_relabel")


@dataclass(frozen=True)
class StageSpec:
    index: int
    name: str
    runner: Callable[..., object]
    kwargs: dict


def _ordered_stages(cfg: Config, root: Path, *, provider: str) -> list[StageSpec]:
    manifest_path = resolve_path(root, cfg.get("paths", {}).get("manifest_path", "data/manifests/manifest.jsonl"))
    labeler = make_labeler(cfg, project_root=root, provider=provider)
    return [
        StageSpec(0, "data_source_sampling", enrich_manifest_with_dataset, {"cfg": cfg, "project_root": root}),
        StageSpec(1, "video_preprocess_bucket", stratify_manifest, {"cfg": cfg, "project_root": root}),
        StageSpec(2, "human_discovery", labeler.run, {"manifest_path": manifest_path}),
        StageSpec(5, "masklet_refinement", refine_masks, {"cfg": cfg, "project_root": root}),
        StageSpec(6, "matting_critical_roi", make_adaptive_trimap_from_annotation, {"cfg": cfg, "project_root": root}),
        StageSpec(6, "build_relabel_queue", build_relabel_queue, {"cfg": cfg, "project_root": root}),
        StageSpec(7, "multi_branch_alpha_mqe_rl_fusion", process_relabel_queue, {"cfg": cfg, "project_root": root, "provider": provider}),
        StageSpec(10, "human_in_the_loop", build_hitl_queue, {"cfg": cfg, "project_root": root}),
        StageSpec(11, "final_label_output", export_alpha_labels, {"cfg": cfg, "project_root": root}),
    ]


def run_relabel_pipeline(
    cfg: Config,
    *,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
    provider: str = "mock",
    from_stage: int = 0,
    to_stage: int = 11,
    skip_manifest_build: bool = False,
) -> list[tuple[int, str, object]]:
    """Execute relabeling stages. Steps 7-9 run inside ``process_relabel_queue``."""
    root = Path(project_root) if project_root else Path.cwd()
    results: list[tuple[int, str, object]] = []

    if not skip_manifest_build and from_stage <= 0:
        if dry_run:
            log.info("[dry-run] would build manifest before stage 0")
        else:
            build_manifest(cfg, project_root=root, overwrite=True)

    if from_stage <= 3 <= to_stage:
        if provider in {"yolo_sam2", "yolo_grabcut"}:
            log.info("[stage 3] rl_prompt_agent executed inside %s labeler", provider)
            results.append((3, "rl_prompt_agent", f"via_{provider}_labeler"))
        else:
            log.info("[stage 3] rl_prompt_agent is recorded in relabel queue prompt_history")
            results.append((3, "rl_prompt_agent", "recorded_in_queue"))
    if from_stage <= 4 <= to_stage:
        if provider in {"yolo_sam2", "yolo_grabcut"}:
            log.info("[stage 4] sam2_vos_masklet executed inside %s labeler", provider)
            results.append((4, "sam2_vos_masklet", f"via_{provider}_labeler"))
        else:
            log.info("[stage 4] sam2_vos_masklet skipped in CPU demo (covered by labeler/mock masklet)")
            results.append((4, "sam2_vos_masklet", "skipped_cpu_demo"))

    for spec in _ordered_stages(cfg, root, provider=provider):
        if spec.index < from_stage or spec.index > to_stage:
            continue
        log.info("Running stage %d: %s", spec.index, spec.name)
        kwargs = dict(spec.kwargs)
        kwargs["dry_run"] = dry_run
        out = spec.runner(**kwargs)
        results.append((spec.index, spec.name, out))
    return results

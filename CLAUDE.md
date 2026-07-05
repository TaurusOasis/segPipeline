# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hmp` (Human Matting Pipeline) is a modular, reproducible training pipeline for
human segmentation / video matting / RK3576 deployment. The first milestone is a
reliable, testable, swappable end-to-end engineering skeleton — *not* SOTA
quality. The current code-target memory and layer boundaries live in
`CODE_TARGETS_MEM_zh.md` and `configs/code_targets.yaml`. The full design lives
in `PROJECT_PLAN_human_matting_pipeline.md`; the 12-stage RL + diffusion relabeling design in `PIPELINE_zh.md` / `PIPELINE_v2_zh.md`;
the per-module map in `CODEMAP_zh.md`; the dataset registry in
`configs/datasets.yaml` + `DATASET_STRATEGY_zh.md`.

Pinned boundary: offline GPU teachers do auto-labeling, video masklets,
temporal QA, alpha generation, and distillation targets; `YOLO26s-seg` remains
the edge hard-instance-segmentation student; video tracking/wrapper and matting
student are separate later layers.

## Environments

Two conda envs are used on this machine (paths already allow-listed in
`.claude/settings.local.json`):

- **`hmp-py310`** — light env, base + dev deps. Used for the CLI and the unit
  test suite. Binary: `/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/{hmp,python}`.
- **`yolo26-cu133`** — GPU env with `ultralytics` + torch. Required for real
  YOLO/SAM2 benchmark runs (`coconut-iterate`, `coconut-compare` with `sam2`).
  Binary: `/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python`.

## Commands

```bash
# Install (dev, light)
pip install -e ".[dev]"

# Tests (no GPU / no weights / no external repos)
PYTHONPATH=src pytest -q
# single test
PYTHONPATH=src pytest tests/test_relabel_queue.py -x

# CLI (use the hmp-py310 env binary)
hmp --help
hmp pipeline stages
hmp config show --config configs/demo_relabel.yaml
hmp pipeline run-relabel --config configs/demo_relabel.yaml --provider mock   # CPU, no GPU deps
hmp pipeline run-relabel --config configs/demo_relabel.yaml --provider yolo_sam2  # needs yolo26-cu133
hmp eval coconut-iterate --config configs/coconut_benchmark.yaml   # needs yolo26-cu133

# Full CPU end-to-end demo (generates fixtures + runs all stages with mock alpha teachers)
bash scripts/run_demo_relabel_pipeline.sh
# COCONut benchmark (compare modes) — run inside yolo26-cu133
bash scripts/run_coconut_compare.sh
```

Heavy optional deps are split into `requirements/*.txt` (`yolo`, `labeling`,
`matting`, `export`); `pyproject.toml` only declares light runtime deps.

## Architecture

### Lazy imports & the no-GPU rule

The core `hmp` package **must import without GPU libraries**. Heavy deps
(torch, ultralytics, SAM, RKNN, cleanvision…) are lazy-imported *inside the
function/module that uses them*, never at package top level. The CLI
(`src/hmp/cli.py`) is the single registry of commands; each callback
lazy-imports its implementation inside the callback body, so `import hmp.cli`
never pulls heavy deps. External research repos are wrapped as adapters /
subprocess command templates (see `alpha_branches` command templates in
`configs/pipeline.yaml`) and are **never** vendored into `src/` — they live in
`external/` (gitignored).

Unit tests must never require GPU, weights, or external repos. Use the
`mock`/`DummyLabeler`/`mock_sam2` provider, or `--dry-run`, to keep tests light.

### Config

Configs are plain YAML loaded into `hmp.config.Config` — an attribute-access
wrapper over a nested dict (`cfg.paths.raw_dir`, `cfg.get("labeling", {})`).
There is no schema for config files themselves; the typed contracts are the
JSONL models in `hmp.schemas` (`MediaItem`, `AnnotationRecord` + `InstanceAnnotation`,
`QualityRecord`, `RelabelStep`). These JSONL formats are the **stable on-disk
contracts** between stages. `Config` and the schemas use pydantic `extra="allow"`
so configs can grow new keys without breaking.

### 12-stage pipeline

Canonical stage registry: `src/hmp/pipeline/stages.py` (`PIPELINE_STAGES` +
`build_step_plan`, steps 0–11). The orchestrator that actually runs them is
`src/hmp/pipeline/run_relabel.py:run_relabel_pipeline`, which maps each stage to
a runner function (`enrich_manifest_with_dataset`, `stratify_manifest`,
`labeler.run`, `refine_masks`, `make_adaptive_trimap_from_annotation`,
`build_relabel_queue`, `process_relabel_queue`, `build_hitl_queue`,
`export_alpha_labels`). Stages 3/4/8/9 (RL prompt agent, SAM2 masklet, MQE, RL
fusion) are folded into the surrounding runners as heuristic/mock
implementations — `PIPELINE_STAGES` lists tool options and `hmp` commands per
stage but not all stages have a dedicated runner.

The `--provider` flag selects the step-2 labeler via
`labeling/labeler_factory.py`: `mock` (DummyLabeler) | `yolo_grabcut` |
`yolo_sam2`.

### Shared labeling core

`labeling/auto_label_core.py` exposes `label_instance_from_bbox(...)` →
`InstanceLabelResult`, which runs plan_prompts → segment_from_prompt →
postprocess → decision_and_tags. **Production labeling and the COCONut
benchmark share this function** to keep QA logic from diverging — when changing
labeling behavior, both paths change together.

### Quality gates

`eval/label_quality.py` centralizes quality decisions:
`quality_gates_from_config` reads the unified `labeling.quality_gates` /
`coconut_benchmark.quality_gates` config block; `decision_and_tags` returns
accept/review/reject + error_tags + hint. `matting/relabel_queue.py` reads the
decision out of `prompt_history.decision` to set `review_required` / `status`.
When touching QA thresholds or the accept/review/reject logic, update the shared
gates rather than adding branch-specific checks.

### COCONut benchmark loop

The benchmark chain (see `PIPELINE_v2_zh.md` and `CODEMAP_zh.md`) is:
`coconut-benchmark` → `coconut-compare`/`coconut-iterate` (4 detector×SAM modes +
iteration plan) → `coconut-resummarize` (recompute QA) → `coconut-export-review`
(review_queue.jsonl) → `coconut-import-annotations` (→ annotations_raw.jsonl) →
`coconut-export-hitl` → `coconut-apply-patch` (merge `next_config_patch.yaml`).
Core modules: `eval/coconut_benchmark.py`, `eval/benchmark_compare.py`,
`eval/benchmark_bridge.py`.

Detector modes: `gt_bbox` / `jitter_bbox` / `center_prior` / `yolo_person`.
SAM modes: `grabcut` / `sam2` / `oracle` / `noisy_oracle`.

### Data flow (relabel)

```
dataset coconut-sample → manifest.jsonl
label yolo-sam2        → annotations_raw.jsonl (+ prompt_history.decision)
refine masks           → annotations_refined.jsonl
matting trimap/adaptive-trimap → trimaps/, roi/
relabel queue          → relabel_queue.jsonl  (QA-aware review_required)
pipeline run-relabel   → branch alphas → fused → MQE → HITL queue → alpha_labels.jsonl
```

### Module layout (`src/hmp/`)

`common/` (jsonl, image_io, video_io, hashing, logging, subprocess_utils) ·
`data/` (ingest, build_manifest, stratify, coconut_io/sample, dataset_registry,
yolo_seg_io, extract_frames) · `labeling/` (base, dummy_labeler, labeler_factory,
auto_label_core, yolo_person_detector, sam2_adapter, mock_sam2, yolo_sam2_labeler,
export_annotations) · `refine/` (refine_pipeline, boundary, mask_postprocess) ·
`matting/` (trimap, adaptive_trimap, alpha_branches, alpha_teacher, alpha_fusion,
process_queue, relabel_queue, hitl_queue, export_labels) · `agents/`
(prompt_agent, fusion_agent) · `eval/` (mqe, label_quality, boundary_metrics,
coconut_benchmark, benchmark_compare, benchmark_bridge, report) · `pipeline/`
(stages, run_relabel) · `yolo/` (export_yolo_dataset).

## Conventions

- Every CLI command supports `--dry-run` or a mock mode — preserve this when
  adding commands.
- New heavy deps go into a `requirements/*.txt` file and are lazy-imported, not
  added to `pyproject.toml` runtime deps.
- `data/`, `runs/`, `external/`, weights (`*.pt`/`*.pth`/`*.onnx`/`*.rknn`) are
  gitignored — do not commit generated artifacts. Note `sam2_b.pt` is committed
  at the repo root as a convenience but normally weights are excluded.
- The CLAUDE.md / design docs are Chinese-suffixed (`*_zh.md`); English is the
  default for code comments and the README.

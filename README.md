# Human Matting Pipeline (`hmp`)

A modular, reproducible training pipeline for human segmentation, video matting,
and RK3576 deployment:

```text
raw images/videos
  -> data cleaning
  -> automatic person mask labeling
  -> mask/boundary refinement
  -> quality scoring and review queue
  -> YOLO26x-seg teacher training
  -> YOLO26s-seg student distillation
  -> alpha / trimap / video matting training
  -> ONNX / RKNN deployment for RK3576
```

The first milestone is **not** SOTA quality — it is a reliable, testable,
swappable, end-to-end engineering skeleton.

## Design principles

- The core package is importable **without** GPU libraries.
- Heavy dependencies (torch, ultralytics, SAM, RKNN, ...) are optional and
  lazy-imported inside the modules that use them.
- External research repos are wrapped as adapters / subprocess command
  templates, never vendored into `src/`.
- Every CLI command supports `--dry-run` or a mock mode.
- Unit tests never require GPU, weights, or external repos.

## Install (development)

```bash
conda create -n hmp-py310 python=3.10 -y
conda activate hmp-py310
pip install -e ".[dev]"
```

Heavy optional dependencies are split into `requirements/*.txt` so the base
install stays light:

```bash
pip install -r requirements/yolo.txt      # ultralytics + torch (lazy)
pip install -r requirements/labeling.txt  # SAM adapters deps
pip install -r requirements/matting.txt   # matting deps
pip install -r requirements/export.txt    # onnx / rknn
```

## Usage

```bash
hmp --help
hmp manifest build --config configs/project.yaml
hmp label dummy --config configs/labeling.yaml
hmp dataset export-yolo --config configs/data.yaml
hmp relabel queue --config configs/relabel.yaml
hmp pipeline run-relabel --config configs/demo_relabel.yaml --provider mock
hmp matting process-queue --config configs/pipeline.yaml --provider mock
hmp relabel export-labels --config configs/pipeline.yaml
hmp eval coconut-iterate --config configs/coconut_benchmark.yaml
hmp pipeline stages
```

See `PROJECT_PLAN_human_matting_pipeline.md` for the full step-by-step plan,
`PIPELINE_zh.md` for the 12-stage RL + diffusion relabeling design, and
`TASKS_human_matting_pipeline.md` for the checklist. See
`OPEN_SOURCE_INTEGRATION_TARGETS_zh.md` and
`configs/reference_integrations.yaml` for the concrete open-source adapter
targets and reference-code registry.

Full CPU relabel demo:

```bash
bash scripts/run_demo_relabel_pipeline.sh
```

For real training data, see `DATASET_STRATEGY_zh.md` and the machine-readable
dataset registry in `configs/datasets.yaml`. They separate direct alpha matte
datasets from segmentation/video-mask datasets that must be relabeled before
they can supervise alpha losses.

COCONut can be used as the local sample benchmark for AI-assisted labeling:

```bash
bash scripts/run_coconut_compare.sh
```

It writes predicted masks, GT binary masks, diff masks, quality buckets, and an
iteration plan under `runs/coconut_compare/`.
Existing benchmark runs can be backfilled or turned into active-labeling review
queues with:

```bash
hmp eval coconut-resummarize --benchmark-dir runs/coconut_benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-export-review --benchmark-dir runs/coconut_benchmark
hmp eval coconut-visualize --benchmark-dir runs/coconut_benchmark
```

`oracle` / `noisy_oracle` modes are useful sanity checks, but `coconut-iterate`
excludes them from the selected next production mode by default when a real
SAM/GrabCut mode is available.

## Tests

```bash
pytest
```

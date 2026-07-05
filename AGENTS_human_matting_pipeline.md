# AGENTS.md / CLAUDE.md

Use this file as coding-agent guidance for Claude Code, Codex, or other code-generation agents working on the human-matting-pipeline repository.

## Project Goal

Build a modular training pipeline for human segmentation, video matting, and RK3576 deployment:

```text
raw data
  -> curation
  -> auto labeling
  -> mask refinement
  -> quality scoring
  -> YOLO26x-seg teacher training
  -> YOLO26s-seg distillation
  -> alpha / video matting training
  -> ONNX / RKNN export
```

## Hard Rules

- Do not implement all stages in one pass.
- One task should be small enough for one commit.
- Keep the core Python package importable without GPU libraries.
- Heavy dependencies must be optional and lazy-imported.
- Do not commit datasets, weights, checkpoints, ONNX, RKNN, or cloned external repos.
- Do not vendor external research repositories into `src/`.
- External research code should be called through adapters or subprocess command templates.
- Every new CLI command must support `--dry-run` or a mock mode unless impossible.
- Unit tests must not require GPU, model weights, SAM, YOLO, RKNN, or external repos.
- Use tiny synthetic images, masks, and videos for tests.
- Do not rely on notebooks for pipeline logic.
- Do not overwrite user data unless an explicit `--overwrite` flag is passed.

## Preferred Implementation Style

- Use `src/` layout with package name `hmp`.
- Use Typer for CLI.
- Use YAML configs.
- Use type hints for public functions.
- Keep modules small and focused.
- Use deterministic random seeds.
- Write clear error messages with optional dependency install hints.
- All paths should come from config or CLI arguments.
- Use JSONL for manifests, annotations, and quality records.

## Definition of Done

A task is done only when:

```text
pytest passes
hmp --help works
new CLI command has dry-run or mock path
tests cover the new functionality
README or relevant docs are updated
no heavy dependency is imported at package import time
```

## Branching / Commit Pattern

Use small commits like:

```text
feat: add manifest schema
feat: add video frame extraction
feat: add dummy labeler
feat: export yolo segmentation dataset
feat: add yolo teacher training wrapper
feat: add official distillation wrapper
feat: add trimap generation
feat: add onnx export wrapper
```

## MVP Order

Build in this order:

```text
00 repo skeleton
01 schemas/jsonl
02 image manifest
03 video frames
04 mask IO
05 boundary metrics
06 dummy labeler
07 YOLO seg export
08 curation adapters
09 Ultralytics auto annotate adapter
10 SAM3/Grounded-SAM-2 command adapters
11 HQ-SAM adapter
12 refine pipeline
13 review queue
14 YOLO teacher wrapper
15 official distill wrapper
16 custom KD loss skeleton
17 trimap generation
18 alpha teacher adapters
19 RVM/MatAnyone adapters
20 matting metrics
21 ONNX export
22 RKNN export
23 evaluation report
24 pipeline orchestrator
25 CPU-only demo pipeline
```

## External Adapter Pattern

External models should be configured like this:

```yaml
sam3:
  enabled: true
  command: >
    python external/sam3/run_label.py
    --manifest {manifest_path}
    --prompt {prompt}
    --output {output_dir}
```

The adapter should:

```text
validate input paths
format the command safely
support dry-run
run subprocess
validate expected outputs
parse results into project JSONL format
```

## Testing Pattern

Use fake external scripts in tests instead of real heavy models.

Example:

```text
tests/fixtures/fake_sam3.py
```

The fake script should create small masks and JSON outputs so the adapter can be tested end-to-end.

## Things Not to Do

- Do not clone SAM3, Grounded-SAM-2, RVM, MatAnyone, or RKNN into the package source.
- Do not add CUDA-only dependencies to `requirements/base.txt`.
- Do not make tests depend on internet access.
- Do not hardcode absolute local paths.
- Do not build a notebook-only pipeline.
- Do not merge all experimental KD code into the official distillation wrapper.


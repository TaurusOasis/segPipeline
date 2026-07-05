# Human Segmentation + Video Matting Pipeline — Project Plan

This document is designed for Claude Code / Codex-style coding agents. The project should be built step by step, with each step small enough to become one clean commit or pull request.

The goal is to build a reproducible training pipeline for:

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

The first milestone is not SOTA quality. The first milestone is a reliable, testable, swappable, end-to-end engineering skeleton.

---

## 0. Build Philosophy

### 0.1 Keep the core project lightweight

The core package must not require heavy model dependencies at import time.

Good:

```python
try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("Install optional dependency: pip install -r requirements/yolo.txt") from e
```

Bad:

```python
from ultralytics import YOLO  # at top level of common package
```

### 0.2 Use adapters for external research repos

External repositories such as SAM3, Grounded-SAM-2, HQ-SAM, CascadePSP, MatAnyone, RVM, ViTMatte, and RKNN should be wrapped as adapters.

The core project should own:

```text
configuration
manifests
schemas
format conversion
evaluation
quality scoring
pipeline orchestration
```

External repos should own:

```text
foundation model inference
matting model training
specialized refinement models
vendor-specific export tools
```

### 0.3 Every stage must support dry-run or mock mode

This is critical for Claude Code / Codex iteration. Most CI tests should run without GPU models, weights, SAM, YOLO, or RKNN installed.

### 0.4 One task equals one commit

Each step should have:

```text
Goal
Files to edit
Implementation notes
Command to run
Acceptance criteria
Suggested agent prompt
```

---

## 1. Target Repository Layout

```text
human-matting-pipeline/
│
├── README.md
├── PROJECT_PLAN.md
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml
├── .gitignore
│
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   ├── yolo.txt
│   ├── labeling.txt
│   ├── refine.txt
│   ├── matting.txt
│   └── export.txt
│
├── configs/
│   ├── project.yaml
│   ├── data.yaml
│   ├── curation.yaml
│   ├── labeling.yaml
│   ├── refine.yaml
│   ├── yolo_teacher.yaml
│   ├── yolo_student.yaml
│   ├── distill.yaml
│   ├── matting.yaml
│   └── export.yaml
│
├── src/
│   └── hmp/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── schemas.py
│       │
│       ├── common/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   ├── paths.py
│       │   ├── hashing.py
│       │   ├── image_io.py
│       │   ├── video_io.py
│       │   └── subprocess_utils.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── build_manifest.py
│       │   ├── extract_frames.py
│       │   ├── split_dataset.py
│       │   ├── mask_io.py
│       │   ├── coco_io.py
│       │   ├── yolo_seg_io.py
│       │   └── visualization.py
│       │
│       ├── curation/
│       │   ├── __init__.py
│       │   ├── cleanvision_adapter.py
│       │   ├── fastdup_adapter.py
│       │   ├── cleanlab_adapter.py
│       │   ├── quality_scores.py
│       │   └── review_queue.py
│       │
│       ├── labeling/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── dummy_labeler.py
│       │   ├── ultralytics_auto_annotate.py
│       │   ├── sam3_adapter.py
│       │   ├── grounded_sam2_adapter.py
│       │   ├── hq_sam_adapter.py
│       │   └── export_annotations.py
│       │
│       ├── refine/
│       │   ├── __init__.py
│       │   ├── boundary.py
│       │   ├── mask_postprocess.py
│       │   ├── cascade_psp_adapter.py
│       │   ├── bpr_adapter.py
│       │   ├── samrefiner_adapter.py
│       │   ├── segrefiner_adapter.py
│       │   └── refine_pipeline.py
│       │
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── mask_metrics.py
│       │   ├── boundary_metrics.py
│       │   ├── temporal_metrics.py
│       │   ├── matting_metrics.py
│       │   └── report.py
│       │
│       ├── yolo/
│       │   ├── __init__.py
│       │   ├── export_yolo_dataset.py
│       │   ├── train_teacher.py
│       │   ├── train_student.py
│       │   ├── distill_official.py
│       │   ├── custom_seg_kd.py
│       │   └── validate_yolo.py
│       │
│       ├── matting/
│       │   ├── __init__.py
│       │   ├── trimap.py
│       │   ├── alpha_teacher.py
│       │   ├── rvm_adapter.py
│       │   ├── matanyone_adapter.py
│       │   └── train_matting.py
│       │
│       ├── export/
│       │   ├── __init__.py
│       │   ├── export_onnx.py
│       │   ├── export_rknn.py
│       │   ├── calibration.py
│       │   └── compare_outputs.py
│       │
│       └── pipelines/
│           ├── __init__.py
│           ├── stage_a_prepare_data.py
│           ├── stage_b_auto_label.py
│           ├── stage_c_refine_masks.py
│           ├── stage_d_export_yolo.py
│           ├── stage_e_train_teacher.py
│           ├── stage_f_distill_student.py
│           ├── stage_g_train_matting.py
│           └── run_all.py
│
├── scripts/
│   ├── prepare_env.sh
│   ├── download_weights.sh
│   ├── run_demo_pipeline.sh
│   └── export_rknn.sh
│
├── tests/
│   ├── fixtures/
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_manifest.py
│   ├── test_mask_io.py
│   ├── test_boundary_metrics.py
│   ├── test_yolo_export.py
│   ├── test_trimap.py
│   └── test_cli.py
│
├── notebooks/
│   ├── inspect_dataset.ipynb
│   ├── inspect_masks.ipynb
│   ├── compare_teacher_student.ipynb
│   └── matting_eval.ipynb
│
├── external/
│   └── .gitkeep
│
├── data/
│   ├── raw/
│   ├── frames/
│   ├── manifests/
│   ├── annotations/
│   ├── masks_raw/
│   ├── masks_refined/
│   ├── yolo_seg/
│   ├── alpha/
│   └── calibration/
│
└── runs/
    └── .gitkeep
```

`data/`, `runs/`, model weights, exported ONNX/RKNN files, and external vendor repos should normally be ignored by git.

---

## 2. Data Contracts

### 2.1 Manifest JSONL

One row per image or video frame.

```json
{
  "item_id": "video001_f000123",
  "media_type": "image",
  "path": "data/frames/video001/frame_000123.jpg",
  "width": 1920,
  "height": 1080,
  "sha256": "abc...",
  "source_video": "data/raw/video001.mp4",
  "frame_index": 123,
  "timestamp_ms": 4100,
  "split": null,
  "tags": ["raw", "video_frame"]
}
```

### 2.2 Annotation JSONL

One row per image or frame, with zero or more person instances.

```json
{
  "item_id": "video001_f000123",
  "instances": [
    {
      "instance_id": "person_0",
      "category": "person",
      "bbox_xyxy": [320, 120, 840, 1040],
      "mask_path": "data/masks_raw/video001_f000123_person_0.png",
      "score": 0.93,
      "source": "sam3",
      "track_id": "track_001"
    }
  ]
}
```

### 2.3 Quality Score JSONL

One row per item.

```json
{
  "item_id": "video001_f000123",
  "scores": {
    "blur_score": 0.12,
    "duplicate_cluster_size": 4,
    "mask_area_ratio": 0.24,
    "boundary_f_score": 0.81,
    "teacher_student_iou": 0.89,
    "temporal_warp_error": 0.04
  },
  "decision": "keep",
  "reason": "good_mask_and_boundary"
}
```

### 2.4 YOLO Segmentation Dataset Output

```text
data/yolo_seg/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

YOLO segmentation label line:

```text
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

All polygon coordinates must be normalized to `[0, 1]`.

---

## 3. CLI Target

The final CLI should support this shape:

```bash
hmp manifest build --config configs/project.yaml
hmp frames extract --config configs/project.yaml

hmp curate cleanvision --config configs/curation.yaml
hmp curate fastdup --config configs/curation.yaml

hmp label dummy --config configs/labeling.yaml
hmp label ultralytics-auto --config configs/labeling.yaml
hmp label sam3 --config configs/labeling.yaml
hmp label grounded-sam2 --config configs/labeling.yaml

hmp refine masks --config configs/refine.yaml
hmp review export --config configs/project.yaml

hmp dataset export-yolo --config configs/data.yaml

hmp yolo train-teacher --config configs/yolo_teacher.yaml
hmp yolo train-student --config configs/yolo_student.yaml
hmp yolo distill-official --config configs/distill.yaml
hmp yolo distill-custom --config configs/distill.yaml
hmp yolo validate --config configs/yolo_student.yaml

hmp matting make-trimap --config configs/matting.yaml
hmp matting alpha-teacher --config configs/matting.yaml
hmp matting train --config configs/matting.yaml

hmp export onnx --config configs/export.yaml
hmp export rknn --config configs/export.yaml
hmp export compare --config configs/export.yaml

hmp eval report --config configs/project.yaml
hmp pipeline run-all --config configs/project.yaml
```

---

## 4. MVP Cut Line

### MVP must include

```text
project skeleton
config loader
manifest builder
video frame extraction
mask IO
boundary metrics
dummy labeler
YOLO segmentation dataset export
YOLO26 teacher train wrapper
YOLO26 official distill wrapper
trimap generator
ONNX export wrapper
basic evaluation report
```

### MVP may stub or external-command-wrap

```text
SAM3
Grounded-SAM-2
HQ-SAM
CascadePSP
BPR
SAMRefiner
SegRefiner
RVM
MatAnyone
RKNN
```

### MVP must not require

```text
GPU in unit tests
large model weights in repo
external research repos vendored into src/
manual notebook-only steps
```

---

# 5. Step-by-step Agent Tasks

Each step below can be given directly to Claude Code or Codex.

---

## Step 00 — Initialize repository skeleton

### Goal

Create the Python project structure, basic CLI, config loading, logging, and tests.

### Files

```text
pyproject.toml
README.md
.gitignore
configs/project.yaml
requirements/base.txt
requirements/dev.txt
src/hmp/__init__.py
src/hmp/cli.py
src/hmp/config.py
src/hmp/common/logging.py
tests/test_config.py
tests/test_cli.py
```

### Suggested Agent Prompt

```text
Create the initial Python project skeleton for human-matting-pipeline.

Requirements:
- Use src layout with package name hmp.
- Use Typer for CLI.
- Add a console script named hmp.
- Add YAML config loading in src/hmp/config.py.
- Add basic logging utility.
- Add pyproject.toml.
- Add .gitignore for data, runs, weights, external repos, ONNX/RKNN artifacts.
- Add tests for config loading and CLI import.
- Do not add heavy ML dependencies.

Acceptance:
- `hmp --help` works after editable install.
- `python -m hmp.cli --help` works.
- `pytest` passes.
```

---

## Step 01 — Implement core schemas and JSONL utilities

### Goal

Define stable data contracts for media items, annotations, and quality records.

### Files

```text
src/hmp/schemas.py
src/hmp/common/jsonl.py
tests/test_schemas.py
```

### Suggested Agent Prompt

```text
Implement core project schemas and JSONL utilities.

Requirements:
- Use Pydantic or dataclasses with explicit validation.
- Define MediaItem, InstanceAnnotation, AnnotationRecord, QualityRecord.
- Validate bbox_xyxy order and non-negative width/height.
- Implement read_jsonl and write_jsonl utilities.
- Add tests for valid and invalid records.

Acceptance:
- Can read and write manifest JSONL.
- Can read and write annotation JSONL.
- Can read and write quality score JSONL.
- `pytest` passes.
```

---

## Step 02 — Build image manifest

### Goal

Scan image folders and create a manifest.

### Files

```text
src/hmp/common/hashing.py
src/hmp/common/image_io.py
src/hmp/data/build_manifest.py
src/hmp/cli.py
tests/test_manifest.py
```

### Suggested Agent Prompt

```text
Implement image manifest building.

Requirements:
- Recursively scan configured image directories.
- Support jpg, jpeg, png, webp.
- Read width and height.
- Compute sha256.
- Create stable item_id from relative path or file stem.
- Write JSONL manifest.
- Add CLI command: hmp manifest build.
- Support --dry-run and --overwrite.
- Add tests using generated fixture images.

Acceptance:
- Manifest contains path, width, height, sha256, item_id, media_type.
- Existing output is not overwritten unless --overwrite is passed.
- `pytest` passes.
```

---

## Step 03 — Extract video frames

### Goal

Create frames from videos and append them to the same manifest format.

### Files

```text
src/hmp/common/video_io.py
src/hmp/data/extract_frames.py
src/hmp/cli.py
tests/test_extract_frames.py
```

### Suggested Agent Prompt

```text
Implement video frame extraction.

Requirements:
- Use OpenCV.
- Support extraction by fps, every_n_frames, or max_frames_per_video.
- Write frames to data/frames/<video_stem>/frame_000001.jpg.
- Create MediaItem records with source_video, frame_index, timestamp_ms.
- Add CLI command: hmp frames extract.
- Add a test using a synthetic tiny video.

Acceptance:
- Frames are extracted deterministically.
- Frame manifest is valid JSONL.
- `pytest` passes.
```

---

## Step 04 — Implement mask IO and postprocessing

### Goal

Handle binary instance masks reliably.

### Files

```text
src/hmp/data/mask_io.py
src/hmp/refine/mask_postprocess.py
tests/test_mask_io.py
```

### Suggested Agent Prompt

```text
Implement binary mask IO and simple postprocessing.

Requirements:
- read_binary_mask(path) -> bool numpy array.
- write_binary_mask(path, mask).
- mask_to_bbox_xyxy(mask).
- mask_area_ratio(mask).
- remove_small_components(mask, min_area).
- fill_holes(mask).
- combine_instance_masks(list[mask]).
- Add tests with synthetic numpy masks.

Acceptance:
- Correct bbox for synthetic masks.
- Small components are removed.
- Holes are filled.
- `pytest` passes.
```

---

## Step 05 — Implement boundary metrics

### Goal

Add boundary-aware evaluation for segmentation and later KD/matting decisions.

### Files

```text
src/hmp/refine/boundary.py
src/hmp/eval/boundary_metrics.py
tests/test_boundary_metrics.py
```

### Suggested Agent Prompt

```text
Implement boundary metrics for binary masks.

Requirements:
- mask_to_boundary(mask, pixel_width=None, dilation_ratio=0.02).
- boundary_band(mask, width).
- boundary_iou(pred, gt).
- boundary_precision_recall_fscore(pred, gt).
- Handle empty masks safely.
- Add tests with perfect, shifted, empty, and partially overlapping masks.

Acceptance:
- Perfect mask gives boundary_iou=1.
- Shifted mask gives lower boundary score.
- Empty mask handling is explicit and tested.
- `pytest` passes.
```

---

## Step 06 — Add dummy labeler and labeling abstraction

### Goal

Create the abstraction that all real labeling models will follow.

### Files

```text
src/hmp/labeling/base.py
src/hmp/labeling/dummy_labeler.py
src/hmp/labeling/export_annotations.py
src/hmp/cli.py
tests/test_dummy_labeler.py
```

### Suggested Agent Prompt

```text
Create auto-labeling abstraction and a dummy labeler.

Requirements:
- Define Labeler interface that accepts a manifest and outputs AnnotationRecord JSONL plus mask files.
- Implement DummyLabeler that creates one rectangular person mask in the center of each image.
- Add CLI command: hmp label dummy.
- Output masks to data/masks_raw.
- Output annotations to data/annotations/annotations_raw.jsonl.
- Add tests using generated fixture images.

Acceptance:
- Dummy labeler produces valid masks and annotations.
- The output can be read by project schemas.
- `pytest` passes.
```

---

## Step 07 — Export YOLO segmentation dataset

### Goal

Convert project masks and annotations to YOLO segmentation format.

### Files

```text
src/hmp/data/yolo_seg_io.py
src/hmp/yolo/export_yolo_dataset.py
src/hmp/data/split_dataset.py
src/hmp/data/visualization.py
src/hmp/cli.py
tests/test_yolo_export.py
```

### Suggested Agent Prompt

```text
Implement YOLO segmentation dataset export.

Requirements:
- Read manifest JSONL and annotation JSONL.
- Convert binary masks to polygons using OpenCV contours.
- Normalize polygon coordinates to [0, 1].
- Support class map: person -> 0.
- Split train/val deterministically with seed.
- Copy or symlink images according to config.
- Write data/yolo_seg/images/train, images/val, labels/train, labels/val.
- Write data/yolo_seg/data.yaml.
- Optionally write visualization samples.
- Add tests with synthetic masks.

Acceptance:
- YOLO labels are valid.
- Empty annotations produce empty label files or configured skip behavior.
- data.yaml is generated.
- `pytest` passes.
```

---

## Step 08 — Add data curation adapters

### Goal

Provide optional adapters for CleanVision and fastdup without making them required dependencies.

### Files

```text
src/hmp/curation/cleanvision_adapter.py
src/hmp/curation/fastdup_adapter.py
src/hmp/curation/quality_scores.py
src/hmp/cli.py
tests/test_curation_adapters.py
```

### Suggested Agent Prompt

```text
Implement optional data curation adapters.

Requirements:
- Add CleanVision adapter with lazy import.
- Add fastdup adapter with lazy import.
- Both adapters read manifest JSONL.
- Both write QualityRecord JSONL.
- Add CLI commands:
  - hmp curate cleanvision
  - hmp curate fastdup
- Add --mock mode that writes deterministic fake scores for tests.
- If optional dependency is missing, print actionable install message.

Acceptance:
- Importing hmp does not require cleanvision or fastdup.
- Mock mode works in tests.
- Missing dependency error is clear.
- `pytest` passes.
```

---

## Step 09 — Add Ultralytics auto_annotate adapter

### Goal

Use YOLO detector + SAM/SAM2/SAM3 through Ultralytics auto annotation.

### Files

```text
src/hmp/labeling/ultralytics_auto_annotate.py
src/hmp/cli.py
tests/test_ultralytics_auto_annotate_adapter.py
```

### Suggested Agent Prompt

```text
Implement Ultralytics auto_annotate adapter.

Requirements:
- Lazy import ultralytics.
- Read config fields: image_dir, det_model, sam_model, output_dir, conf, iou.
- Add CLI command: hmp label ultralytics-auto.
- Add --dry-run to validate paths and print the planned call.
- Do not require ultralytics in unit tests.
- Add test using monkeypatch or mock to simulate auto_annotate.
- Convert or record output locations in project annotation format when possible.

Acceptance:
- Dry-run works without GPU.
- Adapter does not import ultralytics at package import time.
- Mocked adapter test passes.
```

---

## Step 10 — Add external command adapters for SAM3 and Grounded-SAM-2

### Goal

Support external research repositories without binding to unstable internal APIs.

### Files

```text
src/hmp/common/subprocess_utils.py
src/hmp/labeling/sam3_adapter.py
src/hmp/labeling/grounded_sam2_adapter.py
src/hmp/cli.py
tests/test_external_labeling_adapters.py
```

### Suggested Agent Prompt

```text
Implement external command adapters for SAM3 and Grounded-SAM-2.

Requirements:
- Read command template from YAML.
- Pass image_dir or manifest path, text prompt, output directory, and optional model weights.
- Support --dry-run.
- Execute command through a safe subprocess wrapper.
- Validate expected output files exist.
- Parse a simple annotation JSON format into AnnotationRecord JSONL.
- Add tests using a fake Python script that pretends to be SAM3/Grounded-SAM-2.

Acceptance:
- Adapter can call fake external command and parse outputs.
- Missing outputs produce clear errors.
- `pytest` passes.
```

---

## Step 11 — Add HQ-SAM adapter

### Goal

Generate higher-quality person masks from boxes or existing rough masks.

### Files

```text
src/hmp/labeling/hq_sam_adapter.py
src/hmp/cli.py
tests/test_hq_sam_adapter.py
```

### Suggested Agent Prompt

```text
Implement HQ-SAM adapter as an external-command adapter.

Requirements:
- Accept project AnnotationRecord JSONL with bbox prompts.
- Call configured external HQ-SAM command.
- Expected output: refined masks and annotation JSON.
- Validate output masks and dimensions.
- Write annotations_hq_sam.jsonl.
- Support --dry-run and fake-script test.

Acceptance:
- Mock external HQ-SAM script can refine masks in tests.
- Output annotations remain compatible with project schemas.
- `pytest` passes.
```

---

## Step 12 — Implement mask refinement pipeline

### Goal

Run postprocessing and optional external refiners on masks.

### Files

```text
src/hmp/refine/refine_pipeline.py
src/hmp/refine/cascade_psp_adapter.py
src/hmp/refine/bpr_adapter.py
src/hmp/refine/samrefiner_adapter.py
src/hmp/refine/segrefiner_adapter.py
src/hmp/cli.py
tests/test_refine_pipeline.py
```

### Suggested Agent Prompt

```text
Implement mask refinement pipeline.

Requirements:
- Read annotation JSONL.
- For each mask, run configured local postprocess operations:
  - remove_small_components
  - fill_holes
  - keep_largest_component optionally
- Optionally run external refiners through command adapters:
  - CascadePSP
  - BPR
  - SAMRefiner
  - SegRefiner
- Compute boundary and area scores before/after.
- Write masks to data/masks_refined.
- Write annotations_refined.jsonl.
- Write refine_report.jsonl.
- Add tests using local postprocess and fake external refiner.

Acceptance:
- Local refine mode works without external repos.
- Refine report records before/after metrics.
- `pytest` passes.
```

---

## Step 13 — Implement quality scoring and review queue

### Goal

Create a queue of samples to keep, refine, review, or drop.

### Files

```text
src/hmp/curation/quality_scores.py
src/hmp/curation/review_queue.py
src/hmp/cli.py
tests/test_review_queue.py
```

### Suggested Agent Prompt

```text
Implement quality scoring and review queue generation.

Requirements:
- Merge scores from curation, masks, boundary metrics, and optional teacher disagreement.
- Configurable thresholds:
  - min_mask_area_ratio
  - max_mask_area_ratio
  - min_boundary_f_score
  - max_duplicate_cluster_size
  - max_blur_score
- Assign decision: keep, refine, review, drop.
- Write review_queue.jsonl and review_summary.json.
- Add CLI command: hmp review build.
- Add tests with synthetic QualityRecord inputs.

Acceptance:
- Threshold decisions are deterministic.
- Summary counts are correct.
- `pytest` passes.
```

---

## Step 14 — Add YOLO26 teacher training wrapper

### Goal

Train YOLO26x-seg teacher from exported YOLO segmentation dataset.

### Files

```text
src/hmp/yolo/train_teacher.py
src/hmp/yolo/validate_yolo.py
src/hmp/cli.py
configs/yolo_teacher.yaml
tests/test_yolo_train_wrapper.py
```

### Suggested Agent Prompt

```text
Implement YOLO26 teacher training wrapper.

Requirements:
- Lazy import ultralytics.
- Read config fields: model, data, imgsz, epochs, batch, device, workers, project, name, pretrained.
- Add CLI command: hmp yolo train-teacher.
- Add --dry-run that prints the YOLO train parameters.
- Add validation wrapper hmp yolo validate.
- Unit tests should mock ultralytics.YOLO and verify train arguments.

Acceptance:
- Dry-run works without ultralytics installed.
- Mocked train call receives expected config.
- `pytest` passes.
```

### Example real config

```yaml
model: yolo26x-seg.pt
data: data/yolo_seg/data.yaml
imgsz: 1024
epochs: 200
batch: 4
device: 0
workers: 8
project: runs/yolo_teacher
name: yolo26x_human_teacher
```

---

## Step 15 — Add official YOLO26 distillation wrapper

### Goal

Use Ultralytics official `distill_model` as the first student training path.

### Files

```text
src/hmp/yolo/distill_official.py
src/hmp/cli.py
configs/distill.yaml
tests/test_yolo_distill_official.py
```

### Suggested Agent Prompt

```text
Implement official YOLO distillation wrapper.

Requirements:
- Lazy import ultralytics.
- Student model path is configurable.
- Teacher model path is configurable through distill_model.
- Support dis parameter.
- Pass normal train parameters to YOLO.train.
- Add CLI command: hmp yolo distill-official.
- Add --dry-run.
- Unit tests mock ultralytics.YOLO and verify distill_model and dis are passed.

Acceptance:
- Dry-run prints student, teacher, data, imgsz, epochs, batch.
- Mocked train call includes distill_model and dis.
- `pytest` passes.
```

### Example real config

```yaml
student_model: yolo26s-seg.pt
teacher_model: runs/yolo_teacher/yolo26x_human_teacher/weights/best.pt
data: data/yolo_seg/data.yaml
imgsz: 1024
epochs: 200
batch: 16
device: 0
dis: 6.0
project: runs/yolo_student
name: yolo26s_distilled_official
```

---

## Step 16 — Add custom segmentation KD skeleton

### Goal

Prepare hooks for mask-logit KD, proto KD, boundary KD, and temporal KD.

This step should not try to fully solve all KD details. It should create a clean extension point.

### Files

```text
src/hmp/yolo/custom_seg_kd.py
src/hmp/yolo/kd_losses.py
src/hmp/cli.py
tests/test_custom_seg_kd_losses.py
```

### Suggested Agent Prompt

```text
Implement a custom segmentation KD skeleton.

Requirements:
- Add kd_losses.py with pure PyTorch functions:
  - mask_logits_kd_loss(student_logits, teacher_logits, valid_mask=None)
  - proto_kd_loss(student_proto, teacher_proto)
  - boundary_band_loss(student_logits, teacher_or_gt_mask, band_width)
  - high_conf_teacher_mask_loss(student_logits, teacher_logits, teacher_conf, threshold)
- Add tests using small tensors.
- Add custom_seg_kd.py as a placeholder trainer entry that explains where to integrate with Ultralytics trainer.
- Add CLI command: hmp yolo distill-custom --dry-run.
- Do not patch Ultralytics internals yet.

Acceptance:
- KD losses are unit-tested.
- Dry-run explains enabled KD terms and expected teacher/student paths.
- `pytest` passes.
```

### Later implementation notes

The full implementation can be done in a later branch by subclassing or patching the Ultralytics segmentation trainer. Keep this step as a controlled foundation.

---

## Step 17 — Generate trimaps from masks

### Goal

Create trimaps for image/video matting from binary masks.

### Files

```text
src/hmp/matting/trimap.py
src/hmp/cli.py
tests/test_trimap.py
```

### Suggested Agent Prompt

```text
Implement trimap generation from binary masks.

Requirements:
- trimap values: 0 background, 128 unknown, 255 foreground.
- Generate unknown band by dilating and eroding binary mask.
- Configurable kernel size or pixel radius.
- Batch process masks from annotation JSONL.
- Write trimaps to data/alpha/trimaps.
- Add CLI command: hmp matting make-trimap.
- Add tests with synthetic masks.

Acceptance:
- Interior foreground is 255.
- Exterior background is 0.
- Boundary band is 128.
- `pytest` passes.
```

---

## Step 18 — Add alpha teacher adapters

### Goal

Support MAM / ViTMatte / Matte Anything / VideoMaMa / GVM through external command adapters.

### Files

```text
src/hmp/matting/alpha_teacher.py
src/hmp/cli.py
tests/test_alpha_teacher_adapter.py
```

### Suggested Agent Prompt

```text
Implement alpha teacher external-command adapter.

Requirements:
- Support provider names: mam, vitmatte, matte-anything, videomama, gvm.
- Read command template from YAML.
- Inputs: image/frame path, mask path, optional trimap path.
- Outputs: alpha matte png or npy.
- Validate dimensions.
- Write alpha_teacher_annotations.jsonl.
- Support --dry-run and fake-script test.

Acceptance:
- Fake alpha teacher script produces alpha files parsed by the adapter.
- Missing alpha outputs produce clear errors.
- `pytest` passes.
```

---

## Step 19 — Add RVM / MatAnyone training adapters

### Goal

Support video matting student training while keeping external repo code outside the core package.

### Files

```text
src/hmp/matting/rvm_adapter.py
src/hmp/matting/matanyone_adapter.py
src/hmp/matting/train_matting.py
src/hmp/cli.py
tests/test_matting_train_adapters.py
```

### Suggested Agent Prompt

```text
Implement video matting training adapters.

Requirements:
- Add external-command adapters for RVM and MatAnyone.
- Read dataset paths, alpha paths, trimap paths, output directory, model config, checkpoint config from YAML.
- Support --dry-run.
- Use fake external command tests.
- Do not import RVM or MatAnyone packages directly in core code.

Acceptance:
- Dry-run prints full planned external training command.
- Fake command creates a mock checkpoint and adapter validates it.
- `pytest` passes.
```

---

## Step 20 — Implement matting metrics

### Goal

Evaluate alpha matte quality and temporal stability.

### Files

```text
src/hmp/eval/matting_metrics.py
src/hmp/eval/temporal_metrics.py
tests/test_matting_metrics.py
tests/test_temporal_metrics.py
```

### Suggested Agent Prompt

```text
Implement basic matting and temporal metrics.

Requirements:
- Alpha metrics:
  - SAD
  - MSE
  - gradient loss style metric
  - connectivity placeholder or documented TODO
- Temporal metrics:
  - frame-to-frame alpha difference
  - optional optical-flow-based warping placeholder
- Metrics should work with numpy arrays.
- Add tests with synthetic alpha mattes.

Acceptance:
- Perfect alpha has zero SAD/MSE.
- Perturbed alpha has non-zero error.
- `pytest` passes.
```

---

## Step 21 — Add ONNX export wrapper

### Goal

Export YOLO26s-seg and matting student models to ONNX.

### Files

```text
src/hmp/export/export_onnx.py
src/hmp/export/compare_outputs.py
src/hmp/cli.py
tests/test_export_onnx_wrapper.py
```

### Suggested Agent Prompt

```text
Implement ONNX export wrapper.

Requirements:
- For YOLO models, call Ultralytics export through lazy import.
- For external matting models, support external command template.
- Configurable imgsz, opset, dynamic, simplify, half.
- Add --dry-run.
- Add output existence validation.
- Add mocked tests.

Acceptance:
- Dry-run prints expected export command or YOLO export args.
- Mocked YOLO export passes expected arguments.
- `pytest` passes.
```

---

## Step 22 — Add RKNN export wrapper

### Goal

Prepare RK3576 deployment path with calibration support.

### Files

```text
src/hmp/export/export_rknn.py
src/hmp/export/calibration.py
src/hmp/cli.py
tests/test_rknn_export_wrapper.py
```

### Suggested Agent Prompt

```text
Implement RKNN export wrapper.

Requirements:
- Treat RKNN conversion as external command or optional Python dependency.
- Read target platform, input ONNX, output RKNN, quantization flag, calibration dataset from YAML.
- Generate calibration image list from manifest with configurable sampling.
- Support --dry-run.
- Add fake-command tests.
- Do not require RKNN toolkit in default tests.

Acceptance:
- Calibration list is generated deterministically.
- Dry-run prints target platform and command.
- Fake converter output is validated.
- `pytest` passes.
```

---

## Step 23 — Build evaluation report

### Goal

Produce one HTML or Markdown report summarizing the run.

### Files

```text
src/hmp/eval/report.py
src/hmp/cli.py
tests/test_eval_report.py
```

### Suggested Agent Prompt

```text
Implement evaluation report generation.

Requirements:
- Read quality scores, refine reports, YOLO validation metrics if present, matting metrics if present.
- Produce Markdown report with tables and links to sample visualizations.
- Support missing sections gracefully.
- Add CLI command: hmp eval report.
- Add tests with small fake metrics files.

Acceptance:
- Report is generated even if only partial inputs exist.
- Counts and averages are correct in tests.
- `pytest` passes.
```

---

## Step 24 — Add pipeline orchestrator

### Goal

Run stages in sequence from a single config.

### Files

```text
src/hmp/pipelines/stage_a_prepare_data.py
src/hmp/pipelines/stage_b_auto_label.py
src/hmp/pipelines/stage_c_refine_masks.py
src/hmp/pipelines/stage_d_export_yolo.py
src/hmp/pipelines/stage_e_train_teacher.py
src/hmp/pipelines/stage_f_distill_student.py
src/hmp/pipelines/stage_g_train_matting.py
src/hmp/pipelines/run_all.py
src/hmp/cli.py
tests/test_pipeline_orchestrator.py
```

### Suggested Agent Prompt

```text
Implement pipeline orchestrator.

Requirements:
- Each stage is a callable function that reads config and returns output paths.
- `hmp pipeline run-all` can run selected stages by name.
- Support --dry-run to print planned stages and inputs/outputs.
- Support --resume by checking output files.
- Add tests using dummy/mock stages.

Acceptance:
- Dry-run lists stages in order.
- Resume skips completed mock stages.
- Failed stage stops pipeline with clear error.
- `pytest` passes.
```

---

## Step 25 — Add demo pipeline

### Goal

Provide a tiny demo that runs without GPU models.

### Files

```text
scripts/run_demo_pipeline.sh
configs/demo.yaml
tests/test_demo_pipeline.py
```

### Suggested Agent Prompt

```text
Create a CPU-only demo pipeline.

Requirements:
- Generate or use tiny fixture images.
- Build manifest.
- Run dummy labeler.
- Refine masks with local postprocessing.
- Export YOLO segmentation dataset.
- Generate trimaps.
- Generate evaluation report.
- Add script scripts/run_demo_pipeline.sh.
- Add a test that runs the demo on temporary directory.

Acceptance:
- Demo completes without GPU or model weights.
- Outputs manifest, annotations, masks, YOLO labels, trimaps, report.
- `pytest` passes.
```

---

# 6. Suggested Config Files

## 6.1 `configs/project.yaml`

```yaml
project:
  name: human-matting-pipeline
  root: .
  seed: 42

paths:
  raw_dir: data/raw
  frames_dir: data/frames
  manifest_path: data/manifests/manifest.jsonl
  annotation_path: data/annotations/annotations_raw.jsonl
  refined_annotation_path: data/annotations/annotations_refined.jsonl
  masks_raw_dir: data/masks_raw
  masks_refined_dir: data/masks_refined
  yolo_dir: data/yolo_seg
  alpha_dir: data/alpha
  runs_dir: runs

logging:
  level: INFO
```

## 6.2 `configs/labeling.yaml`

```yaml
input:
  manifest_path: data/manifests/manifest.jsonl
  image_dir: data/frames

output:
  annotation_path: data/annotations/annotations_raw.jsonl
  mask_dir: data/masks_raw

provider: dummy

ultralytics_auto:
  det_model: yolo26x.pt
  sam_model: sam3.pt
  conf: 0.25
  iou: 0.7

sam3:
  prompt: person
  command: >
    python external/sam3/run_label.py
    --manifest {manifest_path}
    --prompt {prompt}
    --output {output_dir}

grounded_sam2:
  prompt: person
  command: >
    python external/Grounded-SAM-2/run_video_label.py
    --manifest {manifest_path}
    --prompt {prompt}
    --output {output_dir}
```

## 6.3 `configs/refine.yaml`

```yaml
input:
  annotation_path: data/annotations/annotations_raw.jsonl
  mask_dir: data/masks_raw

output:
  annotation_path: data/annotations/annotations_refined.jsonl
  mask_dir: data/masks_refined
  report_path: data/annotations/refine_report.jsonl

local_postprocess:
  remove_small_components: true
  min_component_area: 64
  fill_holes: true
  keep_largest_component: false

boundary:
  band_width: 5
  dilation_ratio: 0.02

external_refiners:
  cascade_psp:
    enabled: false
    command: >
      python external/CascadePSP/refine.py
      --input {input_mask}
      --image {image_path}
      --output {output_mask}
  bpr:
    enabled: false
  samrefiner:
    enabled: false
  segrefiner:
    enabled: false
```

## 6.4 `configs/yolo_teacher.yaml`

```yaml
model: yolo26x-seg.pt
data: data/yolo_seg/data.yaml
imgsz: 1024
epochs: 200
batch: 4
device: 0
workers: 8
project: runs/yolo_teacher
name: yolo26x_human_teacher
pretrained: true
```

## 6.5 `configs/distill.yaml`

```yaml
student_model: yolo26s-seg.pt
teacher_model: runs/yolo_teacher/yolo26x_human_teacher/weights/best.pt
data: data/yolo_seg/data.yaml
imgsz: 1024
epochs: 200
batch: 16
device: 0
workers: 8
project: runs/yolo_student
name: yolo26s_distilled_official

official_kd:
  enabled: true
  dis: 6.0

custom_kd:
  enabled: false
  mask_logits_weight: 2.0
  proto_weight: 0.5
  boundary_weight: 3.0
  high_conf_weight: 0.5
  temporal_weight: 0.2
  teacher_conf_threshold: 0.5
  boundary_band_width: 5
```

## 6.6 `configs/matting.yaml`

```yaml
input:
  manifest_path: data/manifests/manifest.jsonl
  annotation_path: data/annotations/annotations_refined.jsonl
  mask_dir: data/masks_refined

trimap:
  output_dir: data/alpha/trimaps
  radius: 12

alpha_teacher:
  provider: none
  output_dir: data/alpha/teacher_alpha
  command: >
    python external/matting_teacher/run.py
    --image {image_path}
    --mask {mask_path}
    --trimap {trimap_path}
    --output {alpha_path}

student:
  provider: rvm
  output_dir: runs/matting_student
  command: >
    python external/RobustVideoMatting/train.py
    --config {external_config}
```

## 6.7 `configs/export.yaml`

```yaml
yolo:
  weights: runs/yolo_student/yolo26s_distilled_official/weights/best.pt
  format: onnx
  imgsz: 1024
  opset: 12
  dynamic: false
  simplify: true
  output_dir: runs/export/yolo26s

matting:
  checkpoint: runs/matting_student/best.pth
  onnx_path: runs/export/matting_student/model.onnx
  command: >
    python external/matting_export/export_onnx.py
    --checkpoint {checkpoint}
    --output {onnx_path}

rknn:
  enabled: false
  target_platform: rk3576
  quantized: true
  calibration_manifest: data/calibration/calibration.txt
  output_path: runs/export/rknn/yolo26s.rknn
  command: >
    python scripts/convert_rknn.py
    --onnx {onnx_path}
    --target {target_platform}
    --calibration {calibration_manifest}
    --output {output_path}
```

---

# 7. Agent Working Rules

Put this in `AGENTS.md` and `CLAUDE.md`.

```markdown
# Agent Instructions

## Scope
Build this project incrementally. Do not attempt to implement all model integrations in one pass.

## Hard Rules
- Keep the package importable without GPU libraries.
- Heavy dependencies must be optional and lazy-imported.
- Do not commit data, model weights, ONNX, RKNN, checkpoints, or external cloned repos.
- Every new CLI command must support --dry-run or a mock mode unless impossible.
- Every feature must include tests.
- Use small synthetic images/masks/videos in tests.
- Do not rely on notebooks for core functionality.
- External research repos should be called via adapters or subprocess commands.
- Do not modify vendor repos from the core project.

## Preferred Style
- Type hints for public functions.
- Clear error messages with install hints for optional dependencies.
- Deterministic outputs where possible.
- Config-driven paths and thresholds.
- Small modules with limited responsibilities.

## Definition of Done
- `pytest` passes.
- `hmp --help` works.
- The new command has a dry-run or mock path.
- README or relevant docs updated.
```

---

# 8. Recommended Milestones

## Milestone 1 — CPU-only skeleton

Includes Steps 00–07 and Step 25.

Deliverable:

```text
A demo pipeline that runs without GPU and produces:
manifest
synthetic masks
YOLO labels
trimaps
basic report
```

Acceptance:

```bash
bash scripts/run_demo_pipeline.sh
pytest
```

---

## Milestone 2 — Real auto labeling

Includes Steps 08–13.

Deliverable:

```text
Real or external-command-based labeling from:
Ultralytics auto_annotate
SAM3
Grounded-SAM-2
HQ-SAM
local mask refinement
review queue
```

Acceptance:

```bash
hmp label ultralytics-auto --config configs/labeling.yaml --dry-run
hmp label sam3 --config configs/labeling.yaml --dry-run
hmp refine masks --config configs/refine.yaml
hmp review build --config configs/curation.yaml
```

---

## Milestone 3 — YOLO teacher/student training

Includes Steps 14–16.

Deliverable:

```text
YOLO26x-seg teacher wrapper
YOLO26s-seg official distillation wrapper
custom KD loss skeleton
validation wrapper
```

Acceptance:

```bash
hmp yolo train-teacher --config configs/yolo_teacher.yaml --dry-run
hmp yolo distill-official --config configs/distill.yaml --dry-run
hmp yolo distill-custom --config configs/distill.yaml --dry-run
pytest
```

---

## Milestone 4 — Matting training path

Includes Steps 17–20.

Deliverable:

```text
trimap generation
alpha teacher adapter
RVM / MatAnyone training adapter
matting metrics
```

Acceptance:

```bash
hmp matting make-trimap --config configs/matting.yaml
hmp matting alpha-teacher --config configs/matting.yaml --dry-run
hmp matting train --config configs/matting.yaml --dry-run
pytest
```

---

## Milestone 5 — Deployment path

Includes Steps 21–24.

Deliverable:

```text
ONNX export
RKNN export wrapper
calibration list generation
output comparison
full pipeline orchestrator
final report
```

Acceptance:

```bash
hmp export onnx --config configs/export.yaml --dry-run
hmp export rknn --config configs/export.yaml --dry-run
hmp pipeline run-all --config configs/project.yaml --dry-run
pytest
```

---

# 9. First Real Training Path

After the engineering skeleton is built, the first real training run should be:

```text
1. Prepare 500–2,000 images/frames.
2. Build manifest.
3. Run real auto labeling with Ultralytics auto_annotate or SAM3/Grounded-SAM-2.
4. Run local mask refine.
5. Export YOLO segmentation dataset.
6. Train YOLO26x-seg teacher for a short smoke test, e.g. 5–10 epochs.
7. Distill YOLO26s-seg with official distill_model for a short smoke test.
8. Generate trimaps from refined masks.
9. Run matting training adapter in dry-run or short debug mode.
10. Export YOLO26s ONNX.
```

Smoke-test commands:

```bash
hmp manifest build --config configs/project.yaml
hmp label ultralytics-auto --config configs/labeling.yaml
hmp refine masks --config configs/refine.yaml
hmp dataset export-yolo --config configs/data.yaml
hmp yolo train-teacher --config configs/yolo_teacher.yaml
hmp yolo distill-official --config configs/distill.yaml
hmp matting make-trimap --config configs/matting.yaml
hmp export onnx --config configs/export.yaml
hmp eval report --config configs/project.yaml
```

---

# 10. Later Research Extensions

Only after the MVP is stable, add these in separate branches.

## 10.1 Segmentation-specific KD

Add full integration with Ultralytics trainer:

```text
mask logits KD
proto KD
boundary KD
high-confidence teacher consistency
video temporal KD
```

Implementation approach:

```text
1. Keep pure KD losses in hmp.yolo.kd_losses.
2. Add feature extraction hooks to teacher and student.
3. Start with mask logits KD only.
4. Add boundary KD.
5. Add proto KD.
6. Add temporal KD only after video data path is stable.
```

## 10.2 MQE-like quality evaluator

Add a learned or rule-based quality evaluator:

```text
semantic mask quality
boundary quality
teacher disagreement
temporal consistency
alpha confidence
```

Use it for:

```text
sample keep/drop decisions
review queue ranking
hard case mining
refinement queue
```

## 10.3 Better boundary refinement

Add external adapters for:

```text
SAMRefiner
CascadePSP
BPR
SegRefiner
HQ-SAM refinement mode
```

## 10.4 Better alpha teacher

Add external adapters for:

```text
MAM
ViTMatte
Matte Anything
VideoMaMa
GVM
```

## 10.5 RK3576 optimization

Add:

```text
INT8 calibration dataset selection
NPU output comparison
mask proto postprocessing optimization
matting student quantization checks
latency benchmark script
memory benchmark script
```

---

# 11. Definition of Final Success

The project is considered successful when it can run this full loop:

```text
raw videos/images
  -> manifest
  -> automatic person masks
  -> refined masks
  -> quality report
  -> YOLO segmentation dataset
  -> YOLO26x teacher
  -> YOLO26s distilled student
  -> trimaps / alpha teacher data
  -> video matting student
  -> ONNX / RKNN export
  -> evaluation report
```

Minimum final metrics to track:

```text
YOLO mask mAP
person recall
Boundary IoU
Boundary F-score
teacher/student mask IoU
student/teacher boundary gap
alpha SAD
alpha MSE
temporal alpha stability
ONNX/RKNN output difference
RK3576 latency
RK3576 memory usage
```

The most important product-level metric is not COCO-style mask AP alone. For this project, prioritize:

```text
human boundary quality
temporal stability
matting downstream quality
RK3576 deployment feasibility
```

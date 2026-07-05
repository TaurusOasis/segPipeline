# Step-by-step Task Checklist for Claude Code / Codex

## Milestone 1: CPU-only skeleton

- [ ] Step 00: Initialize repository skeleton
- [ ] Step 01: Implement core schemas and JSONL utilities
- [ ] Step 02: Build image manifest
- [ ] Step 03: Extract video frames
- [ ] Step 04: Implement mask IO and postprocessing
- [ ] Step 05: Implement boundary metrics
- [ ] Step 06: Add dummy labeler and labeling abstraction
- [ ] Step 07: Export YOLO segmentation dataset
- [ ] Step 25: Add CPU-only demo pipeline

Acceptance:

```bash
bash scripts/run_demo_pipeline.sh
pytest
```

## Milestone 2: Real labeling and data quality

- [ ] Step 08: Add CleanVision / fastdup curation adapters
- [ ] Step 09: Add Ultralytics auto_annotate adapter
- [ ] Step 10: Add SAM3 and Grounded-SAM-2 external command adapters
- [ ] Step 11: Add HQ-SAM adapter
- [ ] Step 12: Implement mask refinement pipeline
- [ ] Step 13: Implement quality scoring and review queue

Acceptance:

```bash
hmp label ultralytics-auto --config configs/labeling.yaml --dry-run
hmp label sam3 --config configs/labeling.yaml --dry-run
hmp refine masks --config configs/refine.yaml
hmp review build --config configs/curation.yaml
pytest
```

## Milestone 3: YOLO teacher/student

- [ ] Step 14: Add YOLO26x-seg teacher training wrapper
- [ ] Step 15: Add official YOLO26 distillation wrapper
- [ ] Step 16: Add custom segmentation KD loss skeleton

Acceptance:

```bash
hmp yolo train-teacher --config configs/yolo_teacher.yaml --dry-run
hmp yolo distill-official --config configs/distill.yaml --dry-run
hmp yolo distill-custom --config configs/distill.yaml --dry-run
pytest
```

## Milestone 4: Matting path

- [ ] Step 17: Generate trimaps from masks
- [ ] Step 18a: Build mask-to-matte relabeling queue
- [ ] Step 18b: Add alpha teacher adapters
- [ ] Step 19: Add RVM / MatAnyone training adapters
- [ ] Step 20: Implement matting and temporal metrics

Acceptance:

```bash
hmp matting make-trimap --config configs/matting.yaml
hmp relabel queue --config configs/relabel.yaml --dry-run
hmp matting alpha-teacher --config configs/matting.yaml --dry-run
hmp matting train --config configs/matting.yaml --dry-run
pytest
```

## Milestone 5: Deployment and orchestration

- [ ] Step 21: Add ONNX export wrapper
- [ ] Step 22: Add RKNN export wrapper
- [ ] Step 23: Build evaluation report
- [ ] Step 24: Add pipeline orchestrator

Acceptance:

```bash
hmp export onnx --config configs/export.yaml --dry-run
hmp export rknn --config configs/export.yaml --dry-run
hmp pipeline run-all --config configs/project.yaml --dry-run
pytest
```

## First real training smoke test

- [ ] Prepare 500–2,000 images/frames
- [ ] Build manifest
- [ ] Auto-label person masks
- [ ] Refine masks locally
- [ ] Export YOLO segmentation dataset
- [ ] Train YOLO26x-seg for 5–10 epochs
- [ ] Distill YOLO26s-seg for 5–10 epochs
- [ ] Generate trimaps
- [ ] Run matting adapter in dry-run or debug mode
- [ ] Export YOLO26s ONNX
- [ ] Generate evaluation report

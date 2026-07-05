# 代码目标 MEM：三层架构与后续修改边界

最后更新：2026-07-05

本文是后续 Codex / Claude / 人工开发的“目标记忆”。当一个新需求涉及图像分割、视频分割、video matting、temporal memory、端侧部署时，先按本文判断它应该落在哪一层，避免把所有能力塞进 `YOLO26s-seg` 端侧模型。

机器可读版本见 [configs/code_targets.yaml](configs/code_targets.yaml)。

## 当前固定决策

1. `YOLO26s-seg` 是端侧主分割 student，只负责实时 hard instance segmentation。
2. SAM2、SAM-HQ/HQ-SAM2、SAMRefiner、Cutie、RAFT/GMFlow、MatAnyone、VideoMaMa 等重模型都属于离线 GPU 数据引擎或 teacher 系统。
3. 视频能力先在数据侧形成 `track_id + stable masklet`，再考虑端侧 lightweight wrapper。
4. Video matting 是独立阶段：稳定 masklet 先生成 alpha 数据，再训练独立 lightweight matting student；不要让 YOLO26s-seg 直接输出 alpha。
5. COCONut / COCO-ReM 是 segmentation-core 与 benchmark 数据，不是 alpha matte 数据。

## 三层目标

### Layer 1：离线 GPU 数据引擎 / Teacher 系统

职责：

- 生成高质量 instance mask label
- 清洗伪标签并打质量分
- 生成视频 `masklet + track_id`
- 生成 alpha matte、eval_map、branch_source、train_weight
- 训练大 teacher 或 teacher ensemble
- 给端侧 student 提供 hard labels、soft targets、mask/boundary distillation 信号

允许使用：

- GroundingDINO / Grounded-SAM-2 / YOLO26x-seg
- SAM2 / SAM-HQ / HQ-SAM2 / SAMRefiner
- Cutie / XMem
- RAFT / GMFlow / DINOv2 / LightGlue
- MatAnyone / MatAnyone2 / MaGGIe / SEMat
- VideoMaMa / DiffMatte / SDMatte

主要代码目标：

- adapter contract：external repo command template、env、input/output mapping、dry-run
- class-aware proposal：类别来自 detector / grounding / classifier，mask 来自 SAM/refiner
- QA/tiering：gold/silver/bronze/reject/human-review
- provenance：proposal_source、mask_source、refiner、quality、train_weight、human_checked

### Layer 2：端侧 Student

职责：

- `YOLO26s-seg` 在线推理
- 输出 object-level hard mask、bbox、class、confidence
- 支持 ONNX / RKNN / latency report

明确不做：

- 不直接训练 alpha matte
- 不承担高精度边界标注
- 不内置 SAM2/Cutie/RAFT/VideoMaMa 这类重模型
- 不把 temporal memory 直接塞进 student 主干

训练路线：

1. gold + high-confidence silver 训练 baseline
2. YOLO26x/l-seg 与 refined labels 训练 teacher
3. class / box / mask / boundary / feature distillation 到 YOLO26s-seg
4. 用 `quality_score` / `train_weight` 对 pseudo labels 加权

### Layer 3：端侧可选视频 wrapper 与独立 matting student

职责：

- Runtime A：逐帧 YOLO26s-seg
- Runtime B：YOLO26s-seg + mask IoU / bbox Kalman / appearance embedding tracker
- Runtime C：keyframe YOLO + lightweight mask propagation / periodic refresh
- 后续：RGB + mask + previous alpha 的 lightweight matting student

端侧 wrapper 可以做：

- mask IoU matching
- bbox center / size motion model
- short-term missing-frame recovery
- confidence smoothing / hysteresis
- periodic re-detection

端侧 wrapper 不优先做：

- RAFT/GMFlow 级别的大 optical flow
- SAM2/Cutie 级别的 VOS memory
- DINOv2/LightGlue 级别的重特征匹配
- diffusion refine

## 版本路线

| 版本 | 目标 | 主要产物 | 当前优先级 |
| --- | --- | --- | --- |
| V0 | 图像目标分割数据引擎 | `image_seg_dataset_v1`、YOLO labels、quality/train_weight | 最高 |
| V1 | YOLO26s-seg 图像分割 student | baseline/distilled `.pt`、ONNX/RKNN candidate、latency report | 高 |
| V2 | 视频分割数据引擎 | `masklet_tracks/`、track metadata、temporal QA report | 中 |
| V3 | 视频 matting 数据引擎 | alpha/eval_map/branch_source/train_weight | 中后 |
| V4 | 端侧 video wrapper / matting student | tracker runtime、flicker report、lightweight alpha model | 后续 |

## Phase 0：标签标准先行

在继续扩大自动标注前，先补齐 label spec。建议新增或完善：

- `configs/class_map.yaml`
- `LABEL_SPEC_zh.md`
- `configs/qa_schema.yaml`
- YOLO segmentation dataset template

至少定义：

- `class_id` / `class_name`
- include / exclude rule
- occlusion rule
- part-vs-whole rule
- accessory rule
- ignore region rule
- minimum object size
- crowd rule
- polygon simplification rule

人体相关建议先分三层语义：

- `person_core`：人体主体，不含大部分手持物
- `person_full`：人体 + 衣服 + 头发 + 明确附着在人身上的配件
- `person_alpha`：只用于 matting 阶段，不给 YOLO26s-seg 直接训练

## V0 图像 Auto Label Engine 目标

流程：

```text
raw images
  -> class-aware detection / grounding
  -> SAM2 / SAM-HQ mask proposal
  -> SAMRefiner / local refinement
  -> canonical mask / polygon
  -> quality score
  -> human audit only for hard cases
  -> YOLO-format segmentation label
```

必须落盘：

- `images/`
- `labels_yolo_seg/`
- `masks_png/`
- `polygons_json/`
- `quality_json/`
- `ignore_regions/`
- `train_weight/`
- `auto_label_report_v1`
- `human_review_queue_v1`

质量字段建议：

```json
{
  "class_score": 0.94,
  "box_score": 0.91,
  "mask_iou_agreement": 0.87,
  "boundary_score": 0.79,
  "edge_alignment": 0.82,
  "area_ratio_score": 0.93,
  "overlap_conflict": 0.03,
  "small_object_risk": 0.12,
  "teacher_disagreement": 0.18,
  "final_quality": 0.86,
  "label_tier": "silver"
}
```

训练分级：

- `quality >= 0.90`：gold / high-weight
- `0.75 <= quality < 0.90`：silver / normal-weight
- `0.55 <= quality < 0.75`：bronze / low-weight or distillation-only
- `quality < 0.55`：reject or human-review

## V1 YOLO26s-seg Student 目标

Baseline：

- 数据：gold + high-confidence silver
- 模型：`yolo26s-seg`
- 输入：640 或目标端侧分辨率
- 损失：原生 YOLO segmentation loss

评估：

- `mAP_box`
- `mAP_mask`
- `AP_small / AP_medium / AP_large`
- boundary score
- edge latency
- false positive / false negative
- bad-case gallery

Distillation：

```text
L_total =
  L_yolo_hard
  + lambda_cls * L_cls_distill
  + lambda_box * L_box_distill
  + lambda_mask * L_mask_distill
  + lambda_boundary * L_boundary
  + lambda_feat * L_feature
```

关键要求：蒸馏必须覆盖 mask 和 boundary，不只蒸 class / box。

## V2 视频 Segmentation 数据引擎目标

流程：

```text
video
  -> shot split
  -> keyframe detection / prompt
  -> SAM2 / Cutie / XMem propagation
  -> track_id assignment
  -> temporal QA
  -> failed frames to repair queue
  -> video segmentation dataset
```

每帧记录：

```json
{
  "video_id": "xxx",
  "frame_id": 128,
  "track_id": "person_003",
  "class_id": 0,
  "bbox": [0, 0, 10, 10],
  "mask_path": "masks/person_003/000128.png",
  "visible": true,
  "occluded": false,
  "reentry": false,
  "quality": {
    "mask_score": 0.88,
    "temporal_score": 0.91,
    "identity_score": 0.95
  }
}
```

Temporal QA 信号：

- optical-flow warp IoU / boundary error
- area jump / center jump
- local feature identity check
- detector re-acquisition disagreement
- instance swap risk

## V3 视频 Matting 数据引擎目标

流程：

```text
stable masklet
  -> adaptive ROI / trimap
  -> A_video: MatAnyone / MatAnyone2
  -> A_image: SEMat / MaGGIe / Matting Anything
  -> A_diff: VideoMaMa / DiffMatte
  -> MQE / temporal QA
  -> fused alpha
  -> train_weight / eval_map
```

必须落盘：

- `frame.jpg`
- `mask.png`
- `trimap_or_roi.png`
- `alpha.png` / `alpha.exr`
- `eval_map.png`
- `branch_source.png`
- `track_id.json`
- `quality.json`
- `train_weight.png`

规则：

- hard mask 与 alpha 不混用。
- `foreground_core` / `background_core` 必须 clamp diffusion 输出。
- diffusion 只用于 ROI refine，不覆盖全图。
- 没有稳定 `track_id` 的视频不进入 matting teacher。

## 代码修改定位表

| 需求类型 | 应修改位置 | 不应修改位置 |
| --- | --- | --- |
| class ontology / ignore rule | `configs/class_map.yaml`、`LABEL_SPEC_zh.md`、schema | SAM adapter 内硬编码类别 |
| class-aware proposal | `src/hmp/labeling/`、future `src/hmp/adapters/detection/` | SAM mask refinement |
| mask QA / tiering | `src/hmp/eval/label_quality.py`、benchmark summary | YOLO training loop 内散写规则 |
| COCONut hard-case mining | `src/hmp/eval/coconut_benchmark.py`、`benchmark_bridge.py` | matting alpha exporter |
| YOLO seg export | `src/hmp/data/yolo_seg_io.py`、future `src/hmp/yolo/` | alpha label schema |
| distillation wrappers | future `src/hmp/yolo/` | `labeling/auto_label_core.py` |
| video masklet | future `src/hmp/adapters/vos/`、`src/hmp/schemas.py` | edge student model |
| temporal QA | future `src/hmp/eval/temporal_*` | YOLO mask head |
| alpha teacher | `src/hmp/matting/`、future `src/hmp/adapters/matting/` | YOLO labels |
| edge temporal wrapper | future `src/hmp/runtime/` or `src/hmp/export/` | offline VOS adapter |

## 最近的代码优先级

1. 把 COCONut benchmark 的 `review_priority / area_buckets / tag_metrics` 用于 hard-case queue。
2. 增加 image segmentation dataset export 的 `quality_json` 与 `train_weight`。
3. 抽象 external adapter base contract，支持 command template + dry-run + output validation。
4. 接入 SAM2/SAM-HQ/SAMRefiner 作为离线 mask teacher/refiner，而不是端侧依赖。
5. 新增 YOLO baseline/distill wrapper 的 dry-run skeleton。
6. 再进入 video masklet schema 和 temporal QA。
7. 最后进入 video matting alpha teacher 和端侧 lightweight wrapper。

## 验收命令

轻量验收：

```bash
PYTHONPATH=src pytest -q
hmp pipeline stages
hmp eval coconut-resummarize --benchmark-dir runs/coconut_benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-export-review --benchmark-dir runs/coconut_benchmark
```

真实 GPU 验收：

```bash
PYTHONPATH=src yolo26-cu133/bin/python -m hmp.cli eval coconut-iterate -c configs/coconut_benchmark.yaml
```

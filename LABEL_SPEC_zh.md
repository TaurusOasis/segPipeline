# 标签规范：Image Segmentation 到 Video Matting 的边界

最后更新：2026-07-05

本文是 Phase 0 的标签规范，和 [CODE_TARGETS_MEM_zh.md](CODE_TARGETS_MEM_zh.md) 配套使用。目标是先把图像 hard instance segmentation 做干净，再进入 video masklet 和 alpha matting；不要让 `YOLO26s-seg` 同时承担 alpha、视频记忆和高精度 teacher 的职责。

机器可读配置：

- [configs/class_map.yaml](configs/class_map.yaml)：类别、语义层、include/exclude/ignore 规则
- [configs/qa_schema.yaml](configs/qa_schema.yaml)：质量字段、tier、train_weight 映射
- [configs/code_targets.yaml](configs/code_targets.yaml)：三层代码目标与版本路线

## 固定原则

1. SAM / SAM2 / SAM-HQ / SAMRefiner 只负责 mask quality，不决定类别。
2. 类别必须来自 detector / grounding / classifier / human label spec。
3. `YOLO26s-seg` 训练只吃 hard instance segmentation label。
4. `person_alpha` 是 soft alpha matte 目标，只能进入 matting pipeline，不能导出到 YOLO segmentation label。
5. COCONut / COCO-ReM 可做 segmentation-core 与 benchmark，不可直接作为 alpha supervision。
6. 每个自动标签都必须保留 provenance：`proposal_source`、`mask_source`、`refiner`、`quality_score`、`label_tier`、`train_weight`、`human_checked`。

## 人体三层语义

| 语义层 | 用途 | 是否可进 YOLO | 说明 |
| --- | --- | --- | --- |
| `person_core` | core mask / refinement seed / trimap foreground core | 可以作为辅助 hard mask | 人体主体，偏保守，不包含大多数手持物 |
| `person_full` | 默认 YOLO hard segmentation target | 是 | 人体、衣服、头发、帽子、明确穿戴或附着在人身上的配件 |
| `person_alpha` | matting alpha target | 否 | 软透明边界、头发半透明、motion blur、eval_map / train_weight |

当前 YOLO 导出默认使用 `person_full -> class_id=0 -> person`。

## Include / Exclude 规则

默认 `person_full` 包含：

- 可见人体皮肤、头发、衣服、鞋
- 帽子、围巾、口罩、眼镜等明确穿戴在人身上的物体
- 与身体自然连接且视觉上不可分割的服饰边缘
- 被遮挡后仍可见的独立人体区域

默认排除：

- 手持物：手机、包、球、伞、工具、杯子
- 交通工具、家具、椅子、自行车、滑板、动物
- 与人体重叠但属于其它类别的物体
- 镜面反射、影子、屏幕中人像、海报中人像，除非任务明确要求

默认 ignore：

- crowd 区域中无法稳定分实例的人
- 极小人像或严重 motion blur，低于配置阈值时不参与 hard-label 训练
- 多人强粘连且自动系统无法稳定分离的区域
- 遮挡后 re-entry 身份不确定的 track 片段

## Occlusion / Overlap 规则

- 可见部分按 instance 标注，不补不可见身体。
- `person` 与附着衣物/头发合并到同一 instance。
- 手持物默认不并入人，即使与手部接触。
- 多人重叠时每个人保留独立 instance；不能稳定分离时进入 `human-review` 或 ignore。
- person 与 backpack/bike/chair 等重叠时，优先保持人体可见区域，不吞掉其它物体。

## Polygon / Mask 规则

- YOLO segmentation label 使用 normalized polygon。
- mask 到 polygon 时保留主连通区域和合理分离的可见人体区域；碎片、小噪声按阈值删除。
- 孔洞策略由类别决定：人体衣物内部的小孔默认填充；真实大洞或肢体间空隙不强制填充。
- polygon 简化不能破坏细长结构和手指/头发边界；低质量边界进入 review 或 teacher refine。

## 质量分与训练权重

每个 instance 至少写入：

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
  "label_tier": "silver",
  "train_weight": 0.75
}
```

Tier 约定：

| tier | final_quality | train_weight | 用途 |
| --- | ---: | ---: | --- |
| `gold` | `>= 0.90` | `1.00` | 人工确认或高可信公开精标，高权重训练 |
| `silver` | `>= 0.75` | `0.75` | 默认训练样本 |
| `bronze` | `>= 0.55` | `0.35` | 低权重或 distillation-only |
| `reject` | `< 0.55` | `0.00` | 丢弃或进入人工 review |

## 文件产物

V0 图像 segmentation 数据引擎结束时应有：

```text
image_seg_dataset_v1/
  images/
  labels_yolo_seg/
  masks_png/
  polygons_json/
  quality_json/
  ignore_regions/
  train_weight/
  reports/
```

V2/V3 后续扩展时再增加：

```text
masklet_tracks/
track_meta.json
temporal_quality_report.json
alpha/
eval_map/
branch_source/
matting_quality_report.json
```

## 后续代码落点

| 需求 | 修改位置 |
| --- | --- |
| 新类别 / 语义层 | `configs/class_map.yaml` + 本文 |
| QA 字段 / tier / train_weight | `configs/qa_schema.yaml` |
| YOLO 导出过滤 | `src/hmp/data/yolo_seg_io.py` / `src/hmp/yolo/` |
| 自动标注 provenance | `src/hmp/schemas.py` / `src/hmp/labeling/` |
| benchmark hard-case queue | `src/hmp/eval/coconut_benchmark.py` |
| alpha 规则 | `src/hmp/matting/`，不可写进 YOLO label exporter |


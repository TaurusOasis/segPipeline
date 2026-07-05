# 人体 Matting 二次重标注 Pipeline（12 阶段）

本文档是 `segPipeline` 的 canonical pipeline 设计，对应机器可读定义：

- 阶段注册表：`src/hmp/pipeline/stages.py`
- 统一配置：`configs/pipeline.yaml`
- 数据策略：`DATASET_STRATEGY_zh.md`
- 数据集 registry：`configs/datasets.yaml`

核心原则不变：**hard mask ≠ alpha matte**。Pipeline 的目标是把视频/图像里的 person instance，经过 masklet、trimap、多 teacher alpha、质量评估与 HITL，最终输出可监督训练的 alpha label。

## 总览

```text
0. 数据源采样
   SA-V / YouTube / OpenImages视频 / 内部真实视频
   + COCO-ReM / COCONut / COCO-Matting / HIM / HHM 作为 image teacher 与质量先验
   ↓
1. 视频预处理与采样分层
   shot detection / duplicate removal / resolution filtering / motion filtering
   按人像距离、头发、遮挡、多人、运动模糊、光照、背景复杂度分桶
   ↓
2. 人体发现与目标指定
   person detector / GroundingDINO / pose / face / person classifier
   输出 person bbox、keypoints、instance candidate、target id
   ↓
3. RL Prompt Agent
   选择关键帧、box / positive point / negative point / mask prompt
   决定是否需要人工补 scribble
   ↓
4. 视频 masklet 生成
   SAM2 / VOS tracker / XMem / Cutie
   支持 keyframe prompt、mask prompt、box prompt、positive/negative points
   输出 per-instance masklet + track_id
   ↓
5. masklet 修正与 refinement
   SAM2 correction loop / SAMRefiner / HQ-SAM / COCO-ReM-style boundary refinement
   temporal consistency check / identity switch check / occlusion handling
   ↓
6. Matting-critical region 生成
   adaptive trimap + ROI detector + edge/hair/motion-aware unknown band
   输出 foreground core / background core / unknown ROI
   ↓
7. 多 teacher alpha 生成
   Bv: video matting branch，保证时序稳定
   Bi: image matting branch，保证头发、手指、衣物边缘细节
   Bd: diffusion refine branch，处理 motion blur / 半透明 / 复杂边缘
   Bs: segmentation core branch，保证人体语义完整
   ↓
8. Learned quality evaluation
   MQE / matting quality evaluator
   + rule-based QA: core hole、edge alignment、temporal flicker、instance swap
   输出 pixel-wise reliable map + clip-level quality score
   ↓
9. RL Fusion & Repair Agent
   reliable core 用 video branch
   fine boundary 用 image/diffusion branch
   failed region rerun diffusion / 送人工修边 / 丢弃
   输出 fused_alpha + eval_map + branch_source
   ↓
10. Human-in-the-loop 修正
   只修 MQE/规则检测失败的帧与区域
   scribble / point / mask correction → SAM2 re-propagation
   boundary paint / trimap edit → alpha teacher re-run
   ↓
11. 最终 label 输出
    alpha.png / alpha.exr / mask.png / trimap_or_roi.png / eval_map.png
    bbox.json / instance_id / video_track_id / prompt_history / branch_source
    quality_score / license/meta
```

## 阶段与 hmp 命令映射

| 阶段 | 名称 | 当前 hmp 入口 | 状态 |
| --- | --- | --- | --- |
| 0 | data_source_sampling | `hmp dataset ingest` | 已实现 registry enrich |
| 1 | video_preprocess_bucket | `hmp dataset stratify` | 已实现 heuristic 分桶 + dedup cluster |
| 2 | human_discovery | `hmp label dummy` / `ultralytics-auto --dry-run` | skeleton + adapter 占位 |
| 3 | rl_prompt_agent | `hmp agents prompt` / `hmp relabel queue` | heuristic prompt policy 已接入 queue |
| 4 | sam2_vos_masklet | external SAM2/VOS adapter | CPU demo 由 labeler 代替 |
| 5 | masklet_refinement | `hmp refine masks` | 本地 refine 已实现 |
| 6 | matting_critical_roi | `hmp matting make-adaptive-trimap` | adaptive + ROI 已实现 |
| 7-9 | multi_branch_alpha + MQE + RL fusion | `hmp matting process-queue --provider mock` | mock Bv/Bi/Bd/Bs + fusion + MQE 已实现 |
| 10 | human_in_the_loop | `hmp relabel hitl-queue` | 失败项导出已实现 |
| 11 | final_label_output | `hmp relabel export-labels` | AlphaLabelRecord manifest 已实现 |
| 全流程 | run-relabel | `hmp pipeline run-relabel --provider mock` | CPU 端到端已打通 |

查看阶段列表：

```bash
hmp pipeline stages
```

一键跑通 CPU demo（mock Bv/Bi/Bd/Bs，无需 GPU）：

```bash
bash scripts/run_demo_relabel_pipeline.sh
# 或
hmp pipeline run-relabel --config configs/demo_relabel.yaml --provider mock
```

分步命令：

```bash
hmp dataset ingest --config configs/pipeline.yaml
hmp dataset stratify --config configs/pipeline.yaml
hmp label dummy --config configs/pipeline.yaml
hmp refine masks --config configs/pipeline.yaml
hmp matting make-adaptive-trimap --config configs/pipeline.yaml
hmp relabel queue --config configs/pipeline.yaml
hmp matting process-queue --config configs/pipeline.yaml --provider mock
hmp relabel hitl-queue --config configs/pipeline.yaml
hmp relabel export-labels --config configs/pipeline.yaml
```

## 数据合约

### RelabelTask（queue JSONL）

每个 person instance 一行，包含：

- 输入：`image_path` / `source_video` / `frame_index`、`mask_path`、`masklet_path`、`trimap_or_roi_path`
- 分支：`branch_outputs.Bv/Bi/Bd/Bs`
- 生命周期：`steps[0..11]`
- 输出：`expected_outputs`（alpha_png、alpha_exr、fused_alpha、masklet、eval_map、bbox、IDs、prompt_history、branch_source、license）

生成 queue：

```bash
hmp relabel queue --config configs/pipeline.yaml
```

### AlphaLabelRecord（最终 label manifest）

最终落盘字段见 `src/hmp/schemas.py::AlphaLabelRecord`：

```json
{
  "item_id": "video001_f000123",
  "instance_id": "person_0",
  "image_path": "data/frames/video001/frame_000123.jpg",
  "source_video": "data/raw/video001.mp4",
  "frame_index": 123,
  "alpha_path": "data/alpha/mattes/video001_f000123_person_0_alpha.png",
  "alpha_exr_path": "data/alpha/mattes/video001_f000123_person_0_alpha.exr",
  "mask_path": "data/masks_refined/video001_f000123_person_0.png",
  "masklet_path": "data/alpha/masklets/video001_f000123_person_0_masklet.json",
  "trimap_path": "data/alpha/adaptive_trimaps/video001_f000123_person_0_trimap.png",
  "roi_path": "data/alpha/roi/video001_f000123_person_0_unknown_roi.png",
  "trimap_or_roi_path": "data/alpha/adaptive_trimaps/video001_f000123_person_0_trimap.png",
  "eval_map_path": "data/alpha/eval_maps/video001_f000123_person_0_eval.png",
  "bbox_path": "data/alpha/bboxes/video001_f000123_person_0.json",
  "video_track_id": "track_001",
  "target_id": "target_0",
  "branch_source": {"Bv": "152034", "Bi": "8421", "Bd": "913", "Bs": "0"},
  "quality_score": 0.91,
  "quality_scores": {"core_score": 0.95, "boundary_score": 0.88, "temporal_score": 0.90},
  "prompt_history": [{"tool": "SAM2", "prompt": "box", "frame": 12}],
  "license_meta": {"dataset": "sa_v", "license": "CC-BY-4.0"},
  "review_status": "accepted"
}
```

## 分支策略（Step 7 / 9）

| 分支 | 角色 | 优先区域 | 典型 provider |
| --- | --- | --- | --- |
| Bv | video matting | foreground core、时序稳定 | RVM, MatAnyone, VideoMaMa |
| Bi | image matting | unknown ROI、头发/手指/衣物 | SEMat, ViTMatte, MatteAnything |
| Bd | diffusion refine | 半透明、motion blur、复杂边、ROI refinement | VideoMaMa, DiffMatte, DiffusionMat, SDMatte |
| Bs | segmentation core | core hole 修复、语义完整性 | COCONut, COCO-ReM, HQ-SAM |

Fusion 默认策略（`src/hmp/matting/alpha_fusion.py`）：

1. `foreground_core` → 优先 Bv，缺失则 Bs
2. `unknown_roi` → 按 Bi → Bd → Bv → Bs 优先级
3. `reliable_map < 0.35` → eval_map=255，送 HITL 或 reject
4. 失败区域可 fallback 到 Bs 半透明修复

## 质量评估（Step 8）

Rule-based QA（已实现）：

| 规则 | 检测内容 | 失败标记 |
| --- | --- | --- |
| core_hole | core 区域 alpha 填充不足 | `failed_rules: core_hole` |
| edge_alignment | boundary band 与 mask 不对齐 | `edge_alignment` |
| temporal_flicker | 相邻帧 alpha 抖动过大 | `temporal_flicker` |
| instance_swap | core 几乎为空，疑似 ID 切换 | `instance_swap` |

```bash
hmp eval mqe --config configs/pipeline.yaml --dry-run
```

后续 learned MQE 应复用同一 `MqeRecord` schema，只替换 scorer 实现。

## Adaptive trimap（Step 6）

不再只用 fixed erode/dilate：

- 基于 distance transform 的局部 unknown band
- `hair_priority` / `motion_blur` tag 会加宽 unknown 区域
- 同时输出 `foreground_core`、`background_core`、`unknown_roi`

```bash
hmp matting make-adaptive-trimap --config configs/pipeline.yaml
```

legacy fixed trimap 仍保留：

```bash
hmp matting make-trimap --config configs/relabel.yaml
```

## COCONut 迭代评测闭环

COCONut 的 person label 不是 alpha matte，但非常适合作为 segmentation-core / prompt / SAM2 correction 的 sample benchmark：

```bash
hmp eval coconut-benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-iterate --config configs/coconut_benchmark.yaml
```

输出内容：

- `benchmark_records.jsonl`：每个 person instance 的 AI 标注 vs GT 指标
- `pred_masks/`、`gt_masks/`、`diff_masks/`：预测、GT、差异图
- `contact_sheet_worst.png`：最差样本的 image / GT / pred / diff 四列可视化
- `benchmark_summary.json/md`：IoU、boundary F1、accept/review/reject、错误分桶
- `compare_summary.md`、`iteration_plan.json`、`next_config_patch.yaml`：多模式比较和下一轮配置建议
- `review_queue.jsonl`：将 review/reject 样本转成 active-labeling 队列

建议用法：

1. 小样本先跑 `gt_bbox + oracle/noisy_oracle` 验证数据读取和指标。
2. 再跑 `gt_bbox + sam2` 评估 prompt/SAM2 masklet 质量上限。
3. 最后跑 `yolo_person + sam2` 评估真实自动标注链路。
4. 将 `review/reject` 的 COCONut 样本作为 prompt agent / detector / refinement 的 hard-case queue。

`oracle/noisy_oracle` 只作为 sanity check。`hmp eval coconut-iterate` 默认不会在存在真实模式时把 oracle 选为下一轮生产配置，除非显式设置 `coconut_compare.allow_oracle_selection: true`。

已有旧结果也可以直接补摘要和 review queue：

```bash
hmp eval coconut-resummarize --benchmark-dir runs/coconut_benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-export-review --benchmark-dir runs/coconut_benchmark
hmp eval coconut-visualize --benchmark-dir runs/coconut_benchmark
```

## 推荐落地顺序

1. 用 `configs/datasets.yaml` 选定 SA-V + COCO-ReM/COCONut 采样池
2. `hmp manifest build` + `hmp frames extract`
3. 跑 person labeling + `hmp refine masks`
4. `hmp matting make-adaptive-trimap`
5. `hmp relabel queue` 生成 branch/MQE/fusion 任务
6. 外部 alpha teacher adapter 按 Bv/Bi/Bd/Bs dry-run 接入
7. `hmp eval mqe` + `hmp matting fuse-alpha`
8. HITL 只处理 `review_required=true` 或 `eval_map=255` 区域
9. 导出 `AlphaLabelRecord` manifest 供 matting / distillation 训练
10. 用 `hmp eval coconut-iterate` 持续回归 segmentation-core / prompt / SAM2 correction 效果

## 模块边界

```text
src/hmp/pipeline/          # 阶段定义、run_relabel 编排
src/hmp/data/              # ingest、stratify、manifest
src/hmp/matting/           # trimap、branches、fusion、process_queue、export
src/hmp/eval/mqe.py        # 质量评估
src/hmp/labeling/          # detector / SAM adapter（待扩展）
src/hmp/refine/            # masklet refinement
external/                  # 重型 teacher 推理，不 vendoring 到 src/
```

每个模块保持 adapter 模式：**core 只拥有 schema、config、orchestration；模型推理在外部 repo**。

# 开源标注引擎落地目标与参考代码

本文把“不要从零写，而是拼成离线数据标注引擎”的分析落到本仓库的工程目标。当前三层代码目标记忆见 [CODE_TARGETS_MEM_zh.md](CODE_TARGETS_MEM_zh.md)。原则是：外部研究项目只通过 adapter / subprocess / command template 接入，不把大仓库源码散落进 `src/`，所有输出都回到 `hmp` 的 JSONL schema、质量评估和 review queue。

## 总目标

输入图像或视频，输出可训练的人体 alpha label：

```text
raw image/video
  -> person candidates
  -> prompt history
  -> per-instance mask / video masklet
  -> refined masklet
  -> adaptive trimap / unknown ROI
  -> Bv/Bi/Bd/Bs alpha candidates
  -> eval_map / train_weight / quality_score
  -> fused_alpha
  -> review_queue or accepted AlphaLabelRecord
```

最终每个样本至少落盘：

- `image_path` / `source_video` / `frame_index`
- `mask.png` 或 `masklet/`
- `trimap_or_roi.png`
- `alpha.png` / `alpha.exr`
- `eval_map.png`
- `branch_source`
- `train_weight.png` 或等价权重图
- `bbox.json`
- `instance_id` / `video_track_id` / `target_id`
- `prompt_history`
- `quality_score`
- `license_meta`

## 当前已经落地的基线

本仓库已经把 COCONut 作为 segmentation-core / prompt / correction 的 sample benchmark：

```bash
hmp eval coconut-benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-iterate --config configs/coconut_benchmark.yaml
hmp eval coconut-export-review --benchmark-dir runs/coconut_benchmark
```

输出：

- `benchmark_records.jsonl`：AI mask 与 COCONut GT 的逐实例对比
- `pred_masks/`、`gt_masks/`、`diff_masks/`
- `benchmark_summary.json/md`
- `contact_sheet_worst.png`
- `review_queue.jsonl`
- `compare_summary.md`、`iteration_plan.json`、`next_config_patch.yaml`

新一轮迭代重点是把 COCONut hard cases 变成 prompt agent / mask refinement / detector 参数调优的闭环，而不是把 COCONut binary mask 直接当 alpha。

## 最小可落地开源栈

| 模块 | 优先参考代码 | 本项目目标 |
| --- | --- | --- |
| 数据管理 | FiftyOne, COCO API, PySceneDetect, Decord | 数据浏览、采样、shot split、抽帧、bad case 管理 |
| 人体发现 | Grounded-SAM-2, GroundingDINO, YOLO, Detectron2, MMPose | 输出 person bbox、score、keypoints、target id、prompt seed |
| 视频 masklet | SAM2, Cutie, XMem | 输出 per-instance masklet、track_id、occlusion/re-entry metadata |
| mask refine | SAMRefiner, HQ-SAM/HQ-SAM2, CascadePSP | 粗 mask 修边，减少孔洞、边界错位、多人粘连 |
| adaptive ROI | OpenCV, scikit-image, PyMatting, SCHP, RAFT/GMFlow | 输出 foreground core、background core、unknown ROI |
| video alpha | MatAnyone, MatAnyone2, RVM, MaGGIe | Bv 分支，强调目标人物和时序稳定 |
| image alpha | SEMat, Matting Anything/MAM, Matte Anything, ViTMatte | Bi 分支，强调头发、手指、衣物边缘 |
| diffusion refine | VideoMaMa, DiffMatte, SDMatte, DiffusionMat | Bd 分支，只在 ROI 内修复杂边界和 motion blur |
| QA/MQE | MMagic, PyMatting, RAFT, GMFlow, FiftyOne, Cleanlab | eval_map、temporal flicker、branch disagreement、review queue |
| HITL | CVAT, Label Studio, FiftyOne, Gradio, napari | 人只修低分 ROI，修正后回灌 SAM2 / teacher |
| RL/主动学习 | AlignSAM idea, Gymnasium, Stable-Baselines3, CleanRL | prompt agent、fusion agent、active-labeling decision，不直接生成 alpha |

## 分阶段交付目标

### Phase 1：MVP 离线标注闭环

目标：输入一个真实视频或一批 COCONut/OpenImages 图片，输出可审核的 mask/alpha 候选和 review queue。

必须落盘：

- `manifest.jsonl`
- `person_candidates.jsonl`
- `prompt_history.jsonl`
- `masklet/` 或 `pred_masks/`
- `refined_masks/`
- `adaptive_trimaps/` 与 `unknown_roi/`
- `alpha_raw/Bv`、`alpha_raw/Bi`
- `eval_maps/`
- `fused_alpha/`
- `review_queue.jsonl`

优先接入：

- Grounded-SAM-2 或 YOLO + SAM2
- SAMRefiner 或 HQ-SAM
- MatAnyone 或 MaGGIe
- SEMat 或 Matting Anything
- RAFT/GMFlow 先做 temporal QA，可先降级到 frame-diff baseline
- FiftyOne / CVAT 任选一个做低分样本审核

验收命令目标：

```bash
hmp eval coconut-iterate --config configs/coconut_benchmark.yaml
hmp relabel queue --config configs/pipeline.yaml
hmp matting process-queue --config configs/pipeline.yaml --provider mock
hmp relabel export-labels --config configs/pipeline.yaml
pytest
```

### Phase 2：diffusion refine

目标：只在 `unknown_roi`、hair、motion blur、transparent/soft boundary 区域采用 diffusion 输出，禁止 diffusion 覆盖整个人体 core。

必须实现的保护：

- foreground core clamp
- background core clamp
- ROI-only refine
- identity check
- temporal consistency check
- failed region 自动进入 repair queue

优先参考：

- VideoMaMa：video mask-to-matte diffusion branch
- DiffMatte / SDMatte：keyframe/image ROI refine

### Phase 3：learned MQE

目标：从 rule-based QA 升级到 pixel-wise reliable map，并把 `eval_map` 变成训练权重。

必须输出：

- `eval_map.png`
- `temporal_error.png`
- `branch_disagreement.png`
- `quality_score.json`
- `train_weight.png`

优先参考：

- MatAnyone2 的 MQE / reliable map 思路
- MMagic matting metrics
- PyMatting classical metrics
- FiftyOne bad-case clustering

### Phase 4：RL / active labeling

目标：减少人工修边成本，而不是让 RL 直接预测 alpha。

动作空间：

- Prompt Agent：选 keyframe、box、positive/negative point、mask prompt、是否请求 scribble
- Fusion Agent：在 ROI 内选择 Bv/Bi/Bd/Bs/reject/human-fix
- Active Labeling Agent：在固定人工预算下选择 clip/frame/region

优先参考：

- AlignSAM-style RL prompting
- Gymnasium
- Stable-Baselines3
- CleanRL

## 与当前代码的映射

| 工程目标 | 当前入口 / 计划入口 |
| --- | --- |
| 数据集 registry | `configs/datasets.yaml` |
| 开源参考 registry | `configs/reference_integrations.yaml` |
| COCONut benchmark | `src/hmp/eval/coconut_benchmark.py` |
| 多模式迭代比较 | `src/hmp/eval/benchmark_compare.py` |
| prompt planning | `src/hmp/agents/prompt_agent.py` |
| shared auto-label core | `src/hmp/labeling/auto_label_core.py` |
| YOLO/SAM2 labeler | `src/hmp/labeling/yolo_sam2_labeler.py` |
| adaptive trimap / alpha path | `src/hmp/matting/` |
| MQE schema | `src/hmp/schemas.py::MqeRecord` |
| final label schema | `src/hmp/schemas.py::AlphaLabelRecord` |
| HITL bridge | `src/hmp/eval/benchmark_bridge.py` |

后续 adapter 建议放置：

```text
src/hmp/adapters/
  detection/
    grounded_sam2.py
    groundingdino.py
    yolo.py
    mmpose.py
  vos/
    sam2.py
    cutie.py
    xmem.py
  mask_refine/
    samrefiner.py
    hq_sam.py
    cascadepsp.py
  matting/
    matanyone.py
    maggie.py
    semat.py
    mam.py
    rvm.py
  diffusion/
    videomama.py
    diffmatte.py
    sdmatte.py
  qa/
    raft.py
    gmflow.py
    mmagic_metrics.py
  hitl/
    cvat.py
    label_studio.py
    fiftyone.py
```

## 重点参考仓库

| 优先级 | 仓库 | URL | 用途 | 注意事项 |
| ---: | --- | --- | --- | --- |
| 1 | SAM2 | https://github.com/facebookresearch/sam2 | video masklet 核心 | 权重与 license 单独核对 |
| 2 | Grounded-SAM-2 | https://github.com/IDEA-Research/Grounded-SAM-2 | grounding + SAM2 起点 | 连续 ID 仍需自做 reconciliation |
| 3 | SAMRefiner | https://github.com/linyq2117/SAMRefiner | coarse mask -> refined mask | 适合自动修边 |
| 4 | HQ-SAM | https://github.com/SysCV/sam-hq | 高质量 SAM 边界 | 关注 SAM2 方向 |
| 5 | MatAnyone | https://github.com/pq-yang/MatAnyone | target-assigned video matting | video human branch |
| 6 | MatAnyone2 | https://github.com/pq-yang/MatAnyone2 | MQE / VMReal /真实视频 matting 思路 | 部分训练/MQE资源需关注 release |
| 7 | MaGGIe | https://github.com/hmchuong/MaGGIe | multi-human mask-guided matting | image/video instance matting |
| 8 | SEMat | https://github.com/XiaRho/SEMat | semantic mask -> alpha | COCO-Matting 路线重点参考 |
| 9 | VideoMaMa | https://github.com/cvlab-kaist/VideoMaMa | video diffusion mask-to-matte | 非商用风险需核对 |
| 10 | DiffMatte | https://github.com/YihanHu-2022/DiffMatte | image diffusion matting refine | keyframe/ROI 分支 |
| 11 | RAFT / GMFlow | https://github.com/princeton-vl/RAFT / https://github.com/haofeixu/gmflow | optical-flow temporal QA | 可先做离线 QA |
| 12 | FiftyOne / CVAT | https://github.com/voxel51/fiftyone / https://github.com/cvat-ai/cvat | 数据审核与人工修正 | HITL 中控台 |

## License 与集成规则

- 商业化或大规模训练前，逐项核对 repo license、模型权重 license、数据集条款和 redistribution 条款。
- AGPL、CC BY-NC、research-only 权重不能默认进入商业训练链。
- 外部 repo 不 vendor 到主包；使用 `external_repos/`、环境变量或配置路径引用。
- 每个 adapter 必须支持 `--dry-run` 或 mock provider，单测不能依赖 GPU、权重或网络。
- 每个外部输出必须写回 `prompt_history`、`branch_source`、`quality_scores`、`license_meta`，保证 provenance 可追踪。

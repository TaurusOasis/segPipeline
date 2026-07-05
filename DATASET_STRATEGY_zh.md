# 人体 Matting 数据路线

这份文档把调研稿整理成 `segPipeline` 可执行的数据策略。核心原则很简单：

1. `mask` 不是 `matte`，COCONut、COCO-ReM、SA-V、SAM2 这类 hard mask 不能直接当 alpha label。
2. 训练人体 matting 要把 `alpha supervision`、`segmentation core supervision`、`boundary supervision`、`temporal supervision` 分开管理。
3. 二次重标注的关键不是样本量，而是质量评估。没有 quality evaluator 的自动 alpha 生成很容易把脏 label 放大。

对应的机器可读 registry 在 [configs/datasets.yaml](/home/genesis/Train/Code/segPipeline/configs/datasets.yaml)。**模型分层**（端侧 vs GPU teacher）见 [MODEL_TIERS_zh.md](/home/genesis/Train/Code/segPipeline/MODEL_TIERS_zh.md) 与 [configs/models.yaml](/home/genesis/Train/Code/segPipeline/configs/models.yaml)。

## 模型分层（端侧 vs GPU Teacher）

| 角色 | 模型 | 环境 | 用途 |
| --- | --- | --- | --- |
| **Edge 学生（部署）** | `yolo26s-seg` | RK3576 / 移动端 | 在线分割推理 |
| **检测（auto-label）** | `yolo26s-seg` | GPU | 仅出 person bbox，不直接当最终 mask |
| **Segment Teacher** | SAM2 | GPU | 默认 auto-label mask |
| **Boundary Teacher** | SamHQ | GPU | bad_boundary / review 重标 |
| **Distill Teacher** | `yolo26x-seg` | GPU 训练 | ultralytics/ 蒸馏 teacher，非部署 |
| **Ablation** | GrabCut | CPU | benchmark 对照，禁止生产 |

```text
GPU 标注链路：yolo26s-seg 检测框 → SAM2/SamHQ 生成 mask → QA 门控 → 清洗 mask JSONL
端侧部署：蒸馏后的 yolo26s-seg（RK3576 ONNX/RKNN）
蒸馏训练：yolo26x-seg teacher → yolo26s-seg student（见 ultralytics/distill_model.py）
```

## 优先路线

如果只能选一条主线，优先做：

```text
COCO-ReM / COCONut / SA-V
  -> yolo26s-seg person detector（edge 同架构，GPU 跑框）
  -> SAM2 / SamHQ GPU teacher mask or masklet
  -> SEMat / MatAnyone2-style alpha generation
  -> MQE-style quality scoring + human spot checks
  -> HIM-100K + HHM50K + COCO-Matting + VMReal joint training
  -> yolo26x-seg distill teacher → yolo26s-seg 端侧 student
```

这条路线比只堆 P3M-10K 或 VideoMatte240K 更接近真实视频产品场景：多人、遮挡、复杂背景、长时跟踪、头发和衣物边界同时存在。

## 第一梯队：直接训练 Alpha

| 数据集 | 价值 | 在本项目里的用途 |
| --- | --- | --- |
| COCO-Matting / SEMat | COCO person mask 到 human alpha 的二次重标注，38,251 human instance-level alpha mattes | 复杂自然图像 alpha teacher，验证 mask-to-matte 路线 |
| HIM-100K | 超过 100K human images，约 326K human instances，带 alpha/mask/bbox | 多人体 instance alpha、instance separation loss |
| HSM-200K | 群照中按 box 指定人体 matting，超过 200K human images | target-assigned / selected human matting |
| HHM50K / HHM2K | UHR human matting，50K/2K 图像，平均约 4K | 高分辨率边界 teacher、头发和衣物细节 |
| P3M-10K | 隐私保护 portrait alpha benchmark | 人像和头发边界补充，不作为唯一主数据 |
| VideoMatte240K | 经典绿幕视频 alpha，484 videos / 240,709 frames | video matting 基础预训练，注意 synthetic bias |
| VM800 / VMReal | target-assigned / real-world video human matting | 视频 alpha 和时序一致性训练 |
| MaGGIe 系列 | mask-guided multi-human instance matting | 多人合成、mask perturbation、instance matting |

## 第二梯队：二次重标注底座

| 数据集 | 价值 | 正确用法 |
| --- | --- | --- |
| COCONut | COCO + Objects365 高质量 universal segmentation，383K images / 5.18M masks | person core mask、semantic core loss、alpha 重标注输入 |
| COCO-ReM | COCO-2017 refined instance masks，修复边界、漏标和错标 | 替代原始 COCO person polygon，作为 trimap/mask prompt |
| Sama-COCO | COCO-2017 商业级重标注，person/crowd 更完整 | 多人、crowd、遮挡场景的 hard mask 输入 |
| SA-V / SAM2 | 51K videos / 643K masklets | 视频人体 track、masklet guidance、真实视频 alpha 重标注 |
| SA-1B | 11M images / 1.1B class-agnostic masks | mask prior 和采样池，需要 detector/grounding 过滤 person |
| HQSeg-44K / DIS5K | 极细边界 mask 数据 | boundary refinement、hard mask 边界 teacher |
| OpenImages / Objects365 | 大规模真实图像和检测/分割底座 | 复杂人体场景采样池，先做人筛选再 alpha 重标注 |

## 第三梯队：评测和泛化

| 数据集 | 用途 |
| --- | --- |
| VIM50 | 多人体 video instance matting benchmark |
| MOSE / MOSEv2 | 遮挡、消失重现、复杂 VOS 时序 |
| BURST | 多目标长期 tracking 和 per-frame masks |
| YouTube-VIS | 经典 video instance segmentation baseline |
| LVOS | 长时目标保持和 memory 漂移评估 |
| SynHairMan | 合成高质量头发视频边界补充 |

## 训练组织

不要把所有数据简单混起来。建议按 loss 和训练阶段分开：

```text
Alpha matte loss:
  HIM-100K / HSM-200K / HHM50K / P3M / VideoMatte240K / VM800 / VMReal / COCO-Matting

Segmentation core loss:
  COCONut / COCO-ReM / Sama-COCO / OpenImages / Objects365

Boundary loss:
  HHM50K / HQSeg-44K / DIS5K / P3M / SynHairMan

Temporal consistency loss:
  VideoMatte240K / VM800 / VMReal / SA-V-relabeled / MOSE-relabeled / BURST-relabeled

Instance separation loss:
  HIM-100K / HSM-200K / MaGGIe / VIM50
```

推荐训练节奏：

1. 用 HHM50K + P3M + HIM-100K 训练 image human matting teacher。
2. 加入 COCO-Matting，提升复杂自然图像中的人体泛化。
3. 用 COCONut / COCO-ReM 训练 segmentation core branch。
4. 用 VideoMatte240K + VM800 / VMReal 训练 video matting branch。
5. 用 SA-V / MOSE / BURST 做自动 alpha 重标注，扩充真实视频。
6. 用 teacher-student 蒸馏给轻量端侧模型。

## 二次重标注 Pipeline

完整 12 阶段设计见 [PIPELINE_v2_zh.md](/home/genesis/Train/Code/segPipeline/PIPELINE_v2_zh.md)。旧版 8 步简化流程仍可作为 image-only 快速路径，但视频 alpha 重标注应走完整 12 阶段。

```text
0. 数据源采样
   SA-V / YouTube / OpenImages视频 / 内部真实视频
   + COCO-ReM / COCONut / COCO-Matting / HIM / HHM

1. 视频预处理与采样分层
   shot detection / dedup / resolution / motion / 分桶

2. 人体发现与目标指定
   yolo26s-seg detector / GroundingDINO / pose / face / person classifier

3. RL Prompt Agent
   keyframe / box / positive point / negative point / mask / scribble gate

4. GPU segment teacher（SAM2 / SamHQ）
   bbox prompt → mask；review/bad_boundary 可切换 SamHQ

5. masklet 修正
   SAM2 correction / SAMRefiner / SamHQ / temporal & identity check

6. Matting-critical region
   adaptive trimap + ROI + edge/hair/motion unknown band

7. 多 teacher alpha
   Bv / Bi / Bd(diffusion refine) / Bs

8. MQE + rule-based QA
   reliable map + clip quality score

9. RL / bandit fusion + repair
   core→Bv, boundary→Bi/Bd, failed→rerun diffusion or HITL

10. Human-in-the-loop
   只修失败帧/区域

11. 最终 label 输出
    alpha.png / alpha.exr / mask / trimap_or_roi / eval_map / meta
```

## 和当前 `segPipeline` 的连接点

当前项目已有 CPU-only skeleton，并扩展到 12 阶段 pipeline 合约。COCONut 上已落地 **自动标注 vs GT 精度/效率 benchmark**，用于验证 step 2–4（检测 + prompt + SAM2）是否达到二次重标注门槛。

### 已实现 CLI

```text
hmp config models                        # 查看 edge / teacher 分层
hmp dataset ingest / stratify
hmp label dummy
hmp label yolo-sam2 --segment-mode sam2|samhq|grabcut
hmp label yolo-sam2 --teacher samhq       # SamHQ boundary 重标
hmp refine masks
hmp matting make-adaptive-trimap
hmp relabel queue / hitl-queue / export-labels
hmp eval mqe
hmp eval coconut-benchmark
hmp eval coconut-compare
hmp eval coconut-iterate          # 对比 + 迭代计划 + next_config_patch.yaml
hmp eval coconut-resummarize
hmp eval coconut-export-review
hmp eval coconut-import-annotations   # benchmark → annotations_raw.jsonl
hmp eval coconut-export-hitl          # review/reject → HITL queue
hmp eval coconut-apply-patch          # merge next_config_patch.yaml
hmp dataset coconut-sample            # COCONut val → manifest (step 0)
hmp pipeline bootstrap-from-benchmark # benchmark → manifest/annotations/HITL
bash scripts/run_coconut_relabel_e2e.sh mock|yolo_sam2
hmp pipeline run-relabel --provider mock|yolo_grabcut|yolo_sam2|yolo_samhq
hmp pipeline stages
```

完整 12 阶段设计见 [PIPELINE_v2_zh.md](/home/genesis/Train/Code/segPipeline/PIPELINE_v2_zh.md)；模块依赖与数据流见 [CODEMAP_zh.md](/home/genesis/Train/Code/segPipeline/CODEMAP_zh.md)；模型分层见 [MODEL_TIERS_zh.md](/home/genesis/Train/Code/segPipeline/MODEL_TIERS_zh.md)。

### COCONut 自动标注 Benchmark（2026-07-05）

数据：`relabeled_coco_val`（350 person instances / 128 images sampled）

| detector | SAM | mask IoU | boundary F1 | 吞吐 | accept | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| gt_bbox | GrabCut | 0.383 | 0.699 | 2.7 inst/s | **3.1%** | CPU mock，不可用于生产 |
| gt_bbox | SAM2 | 0.788 | 0.949 | 2.4 inst/s | **37.1%** | segmentation-core 上界 |
| yolo_person | GrabCut | 0.386 | 0.698 | 1.0 inst/s | **3.7%** | 与 gt+GrabCut 同级，仅 ablation |
| yolo_person | SAM2 | 0.771 | 0.937 | 0.83 inst/s | **34.6%** | **推荐端到端默认组合** |

质量门控（`configs/coconut_benchmark.yaml` → `quality_gates`）：

- `accept`: IoU≥0.85 且 boundary F1≥0.85，FP/FN ratio≤0.25
- `review`: 介于 accept 与 reject 之间 → 导出 `review_queue.jsonl` 供 prompt 修正
- `reject`: IoU<0.50 或 boundary F1<0.65 → 不进 alpha 训练

主要 error bucket：`background_leak`、`missed_foreground`、`detector_miss`、`bad_boundary`、`needs_scribble`。

运行：

```bash
bash scripts/run_coconut_compare.sh
# 或
PYTHONPATH=src hmp eval coconut-iterate --config configs/coconut_benchmark.yaml
```

输出：

- `runs/coconut_compare/compare_summary.md` — 四模式排名
- `runs/coconut_compare/iteration_plan.json` — 下一步动作
- `runs/coconut_compare/next_config_patch.yaml` — 可直接合并到 pipeline 配置
- `runs/coconut_compare/*/__mode__/review_queue.jsonl` — 待修正样本

**策略结论**：COCONut hard mask 适合作为 segmentation-core 监督；GrabCut 不能替代 SAM2。端到端：**端侧部署 yolo26s-seg**；auto-label / 清洗 / 蒸馏 teacher 用 **GPU SAM2/SamHQ**（见 [MODEL_TIERS_zh.md](/home/genesis/Train/Code/segPipeline/MODEL_TIERS_zh.md)）。review/reject 样本走 HITL + prompt-agent 闭环，bad_boundary 可切换 SamHQ 重标，再进入 Bv/Bi/Bd/Bs alpha 分支。

### 下一步接真实数据

1. ~~增加 dataset registry loader~~（已有 `configs/datasets.yaml` + `hmp dataset ingest`）
2. ~~COCONut benchmark~~（已完成；持续扩大 limit 与 hard bucket）
3. ~~将 benchmark 选中的 `yolo_sam2` 模式接入 pipeline~~（`hmp pipeline bootstrap-from-benchmark` + `run_coconut_relabel_e2e.sh`）
4. 用 `bootstrap-from-benchmark` / `coconut-export-review` 导出 review 队列；bad_boundary 用 `hmp eval coconut-export-bad-boundary` + `hmp eval coconut-relabel-boundary --teacher samhq`
5. 用 `hmp eval mqe` + `hmp matting fuse-alpha` 在 accept 样本上验证 mask→matte 质量
6. ~~benchmark 增加 `yolo_person × samhq` 模式~~（已加入 `coconut_compare.modes` 默认网格）
7. 从 SA-V / COCO-ReM 扩展 ingest，按 stratify bucket 追加 relabel queue
8. accept 清洗 mask → ultralytics COCONut 蒸馏（yolo26x-seg → yolo26s-seg）

## 风险清单

| 风险 | 处理方式 |
| --- | --- |
| license 不统一 | 每个 dataset ingest 前必须记录 license、用途限制、commercial/research-only |
| alpha 数据不可直接下载 | registry 里标记 `availability`，先接公开可得和需申请数据 |
| hard mask 被误用作 alpha | config 里区分 `direct_alpha` 和 `relabel_source` |
| 自动重标注 label 脏 | 必须有 quality score 和 review queue |
| 视频单帧好但 flicker | 增加 temporal metrics 和 video-level sampling |
| 多人 instance 混淆 | 保留 `instance_id`、`track_id`、`prompt_source`，不要只存单张 alpha |

## 已核验来源

- COCO-Matting / SEMat: https://arxiv.org/html/2410.06593v1
- MatAnyone2 / VMReal: https://pq-yang.github.io/projects/MatAnyone2/
- HIM-100K / E2E-HIM: https://arxiv.org/html/2403.01510v1
- HSM-200K: https://dl.acm.org/doi/10.1145/3640017
- COCONut: https://openaccess.thecvf.com/content/CVPR2024/html/Deng_COCONut_Modernizing_COCO_Segmentation_CVPR_2024_paper.html
- COCO-ReM: https://github.com/kdexd/coco-rem
- Sama-COCO: https://www.sama.com/sama-coco-dataset
- SA-V: https://ai.meta.com/datasets/segment-anything-video/
- HHM50K / HHM2K: https://github.com/nowsyn/SparseMat
- P3M-10K: https://github.com/JizhiziLi/P3M
- VideoMatte240K / PhotoMatte13K: https://grail.cs.washington.edu/projects/background-matting-v2/
- MaGGIe: https://maggie-matt.github.io/

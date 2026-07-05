# Pipeline v2 — RL Agents + COCONut Benchmark

Canonical stage registry: `src/hmp/pipeline/stages.py` (steps **0-11**).

## Flow

```text
0. 数据采样 — SA-V / real videos / OpenImages-video / YouTube / COCO-ReM / COCONut
1. 视频预处理与分桶 — shot / blur / motion / multi-person / hair / occlusion
2. 人体发现 — detector / GroundingDINO / pose / person classifier
3. RL Prompt Agent — keyframe + box/point/mask prompt + scribble gate
4. SAM2 / VOS masklet — temporally consistent person masklet
5. masklet refinement — SAMRefiner / HQ-SAM / temporal + identity check
6. matting-critical ROI — adaptive unknown / hair / hand / motion / occlusion / semi-transparent
7. 多分支 alpha — Bv / Bi / Bd / Bs
8. MQE / quality evaluator — reliable map + semantic/boundary/temporal/identity scores
9. RL Fusion & Repair Agent — per-region branch select / human repair / clip reject
10. Human-in-the-loop — 只修低质量区域
11. 输出 label — alpha / mask / eval_map / branch_source / prompt_history / quality_score
```

## COCONut 自动标注 vs GT 验证

数据路径（本机）：

- JSON: `/home/genesis/Train/Dataset/coconut/relabeled_coco_val.json`
- Masks: `/home/genesis/Train/Dataset/coconut/relabeled_coco_val/*.png`
- Images: `/home/genesis/Train/Dataset/coco2017/val2017/*.jpg`

运行：

```bash
bash scripts/run_coconut_benchmark.sh
bash scripts/run_coconut_compare.sh   # 需在 yolo26-cu133 环境（含 ultralytics）
# 或
hmp eval coconut-benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-compare --config configs/coconut_benchmark.yaml
```

配置项（`configs/coconut_benchmark.yaml`）：

| 字段 | 含义 |
| --- | --- |
| `detector_mode` | `gt_bbox` / `jitter_bbox` / `center_prior` / `yolo_person` |
| `sam_mode` | `grabcut` / `sam2` / `oracle` / `noisy_oracle` |
| `yolo_weights` | YOLO 权重（`yolo_person` 模式，默认 `ultralytics/yolo26s-seg.pt`） |
| `sam2_weights` | SAM2 权重（默认 `sam2_b.pt`，Ultralytics 自动下载） |
| `limit` | 采样 image 数（每图可含多 person instance） |

**2026-07-05 实测**（128 images, grabcut + gt_bbox）：

| 指标 | 值 |
| --- | --- |
| instances | 350 |
| mean mask IoU | 0.3826 |
| mean boundary F1 | 0.6994 |
| throughput | 2.82 inst/s |

**2026-07-05 SAM2 smoke**（8 images, `yolo26-cu133` 环境）：

| detector | sam | mask IoU | boundary F1 | inst/s |
| --- | --- | ---: | ---: | ---: |
| gt_bbox | sam2 | 0.8933 | 0.9703 | 1.12 |
| yolo_person | sam2 | 0.8003 | 0.9127 | 0.44 |

输出：

- `runs/coconut_benchmark/benchmark_records.jsonl` — 每 instance 对比记录
- `runs/coconut_benchmark/benchmark_summary.json` / `.md`
- `runs/coconut_benchmark/manifest.jsonl` / `annotations_pred.jsonl`

## CLI 速查

```bash
hmp pipeline stages
hmp eval coconut-benchmark --config configs/coconut_benchmark.yaml
hmp eval coconut-compare --config configs/coconut_benchmark.yaml
hmp agents prompt --config configs/pipeline.yaml --dry-run
hmp agents fusion --config configs/pipeline.yaml
hmp pipeline run-relabel --config configs/demo_relabel.yaml --provider mock
hmp pipeline run-relabel --config configs/demo_relabel.yaml --provider yolo_sam2
hmp label yolo-sam2 --config configs/pipeline.yaml --segment-mode sam2
```

## 模块

| 路径 | 作用 |
| --- | --- |
| `data/coconut_io.py` | COCONut panoptic 采样 + person GT mask |
| `agents/prompt_agent.py` | Step 3 RL Prompt Agent（heuristic v1） |
| `labeling/mock_sam2.py` | Step 4 GrabCut mock SAM2 |
| `labeling/yolo_person_detector.py` | Step 2 YOLO person 检测（lazy ultralytics） |
| `labeling/sam2_adapter.py` | Step 4 Ultralytics SAM2 适配（bbox prompt） |
| `labeling/yolo_sam2_labeler.py` | Step 2-4 YOLO + Prompt + SAM2/GrabCut 联合标注 |
| `labeling/labeler_factory.py` | pipeline provider 选择 labeler |
| `agents/fusion_agent.py` | Step 9 RL Fusion Agent |
| `eval/coconut_benchmark.py` | 自动标注 vs GT 精度/效率评测 |
| `eval/benchmark_compare.py` | 多 detector×SAM 模式对比表 |

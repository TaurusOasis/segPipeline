# segPipeline 代码地图 (HMP v2)

人体 matting 二次重标注管线：数据采样 → 自动标注 → 质量评估 → 融合 → HITL → 最终 label。

## 入口

| 入口 | 路径 | 说明 |
|------|------|------|
| CLI | `src/hmp/cli.py` | 所有 `hmp` 命令；回调内 lazy-import |
| 编排 | `src/hmp/pipeline/run_relabel.py` | 12 阶段 `--provider mock\|yolo_grabcut\|yolo_sam2` |
| 阶段注册 | `src/hmp/pipeline/stages.py` | `PIPELINE_STAGES` + `build_step_plan` |

## 12 阶段实现状态

| Step | 名称 | 模块 | 状态 |
|------|------|------|------|
| 0 | 数据采样 | `data/ingest.py`, `data/coconut_sample.py` | COCONut manifest CLI 已注册 |
| 1 | 分桶/去重 | `data/stratify.py` | 启发式 tags |
| 2 | 人体发现 | `labeling/yolo_person_detector.py` | YOLO person |
| 3 | Prompt Agent | `agents/prompt_agent.py` | box/point 规划 |
| 4 | SAM2/VOS | `labeling/sam2_adapter.py`, `mock_sam2.py` | SAM2 bbox-only / GrabCut mock |
| 5 | mask 精修 | `refine/refine_pipeline.py` | 本地后处理 |
| 6 | matting ROI | `matting/adaptive_trimap.py` | trimap/ROI |
| 7 | alpha teachers | `matting/alpha_teacher*.py` | mock 为主 |
| 8 | MQE | `eval/mqe.py` | 规则打分 |
| 9 | 融合 | `agents/fusion_agent.py` | 启发式 |
| 10 | HITL | `hitl/queue.py` | review 队列 |
| 11 | 导出 | `export/export_labels.py` | label 输出 |

## 共享标注内核 (Step 2–4 + Benchmark)

```
labeling/auto_label_core.py
  labeling_runtime_from_config(cfg)
  label_instance_from_bbox(...)  → InstanceLabelResult
       ├─ plan_prompts (prompt_agent)
       ├─ segment_from_prompt (sam2 / grabcut / oracle)
       ├─ postprocess_from_config
       └─ decision_and_tags (eval/label_quality.py)
```

生产与 benchmark **共用** `label_instance_from_bbox`，避免 QA 逻辑分叉。

## 质量门控

`eval/label_quality.py`

- `quality_gates_from_config` — 统一 `labeling.quality_gates` / `coconut_benchmark.quality_gates`
- `decision_and_tags` — accept / review / reject + error_tags + hint
- `parse_decision_from_prompt_history` — relabel queue 读取标注决策

`matting/relabel_queue.py` 从 `prompt_history.decision` 设置 `review_required` 与 `status`。

## COCONut Benchmark 链路

```
configs/coconut_benchmark.yaml
  → hmp eval coconut-benchmark          # 单模式
  → hmp eval coconut-compare / iterate  # 四模式 + iteration_plan
  → hmp eval coconut-resummarize        # 重算 QA
  → hmp eval coconut-export-review      # review_queue.jsonl
  → hmp eval coconut-import-annotations # → annotations_raw.jsonl
  → hmp eval coconut-export-hitl        # → hitl queue
  → hmp eval coconut-apply-patch        # merge next_config_patch.yaml
```

核心模块：

| 文件 | 职责 |
|------|------|
| `eval/coconut_benchmark.py` | GT 对比 benchmark；主循环调用 `label_instance_from_bbox` |
| `eval/benchmark_compare.py` | detector × SAM 四模式对比 |
| `eval/benchmark_bridge.py` | benchmark → annotation / HITL / config patch |

## Relabel 数据流

```
dataset coconut-sample → manifest.jsonl
label yolo-sam2        → annotations_raw.jsonl (+ prompt_history.decision)
refine masks           → annotations_refined.jsonl
matting trimap         → trimaps/
relabel queue          → relabel_queue.jsonl (QA-aware review_required)
pipeline run-relabel   → 全流程 orchestrator
```

## 关键 CLI

```bash
hmp dataset coconut-sample -c configs/coconut_benchmark.yaml
hmp label yolo-sam2 -c configs/relabel.yaml
hmp pipeline run-relabel -c configs/relabel.yaml --provider yolo_sam2
hmp eval coconut-iterate -c configs/coconut_benchmark.yaml
```

## 测试

`tests/` 覆盖 manifest、relabel queue、benchmark resummarize、yolo_sam2、bridge、coconut_sample 等。在 `segPipeline` 根目录：

```bash
PYTHONPATH=src pytest -q
```

真实 YOLO/SAM2 benchmark 需 `yolo26-cu133` 环境：

```bash
PYTHONPATH=src yolo26-cu133/bin/python -m hmp.cli eval coconut-iterate -c configs/coconut_benchmark.yaml
```

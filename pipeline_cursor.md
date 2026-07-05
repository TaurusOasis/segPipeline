# Pipeline 跟踪笔记（Cursor 专用）

> **说明**：本文件仅用于 Cursor 会话内跟踪进度与决策，**不替代** canonical 文档。  
> 设计文档仍以 [PIPELINE_zh.md](PIPELINE_zh.md)、[OPEN_SOURCE_INTEGRATION_TARGETS_zh.md](OPEN_SOURCE_INTEGRATION_TARGETS_zh.md)、[CODE_TARGETS_MEM_zh.md](CODE_TARGETS_MEM_zh.md) 为准。

最后更新：2026-07-05  
分支：`iter/prompt-agent-v2` · 最近 commit：`1eaf1bf`

---

## 1. 文档索引（只读引用）

| 文档 | 用途 |
| --- | --- |
| [PIPELINE_zh.md](PIPELINE_zh.md) | 12 阶段 pipeline 设计与 hmp 命令 |
| [OPEN_SOURCE_INTEGRATION_TARGETS_zh.md](OPEN_SOURCE_INTEGRATION_TARGETS_zh.md) | 开源 adapter 分阶段落地目标 |
| [CODE_TARGETS_MEM_zh.md](CODE_TARGETS_MEM_zh.md) | 三层架构 + V0–V4 版本路线 |
| [configs/code_targets.yaml](configs/code_targets.yaml) | 上述目标机器可读版 |
| [configs/reference_integrations.yaml](configs/reference_integrations.yaml) | 开源 repo registry |
| [DATASET_STRATEGY_zh.md](DATASET_STRATEGY_zh.md) | 数据集角色与 COCONut 策略 |
| [MODEL_TIERS_zh.md](MODEL_TIERS_zh.md) | 端侧 yolo26s-seg vs GPU teacher |
| [CODEMAP_zh.md](CODEMAP_zh.md) | 代码地图与模块职责 |

---

## 2. 固定架构决策（不改除非显式讨论）

1. **端侧部署**：只用 `yolo26s-seg`（hard instance segmentation）。
2. **GPU teacher**：SAM2 / SamHQ / yolo26x-seg 等仅用于 auto-label、清洗、蒸馏，不进端侧 runtime。
3. **hard mask ≠ alpha matte**：COCONut / SA-V masklet 不能直接当 alpha 监督。
4. **集成方式**：外部 repo → adapter / subprocess / command template；`src/hmp/adapters/` 尚未创建，当前逻辑在 `labeling/`、`eval/`、`yolo/`。
5. **版本顺序**：V0 图像分割 → V1 YOLO student → V2 视频 masklet → V3 video matting → V4 端侧 runtime。

---

## 3. 版本路线 × 当前状态

| 版本 | 目标 | Pipeline 阶段 | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| **V0** | 图像 instance mask + YOLO seg label | 0–2, 4–5, benchmark QA | 🟡 进行中 | COCONut 128 图 benchmark 已跑；accept overlay 桥接已建 |
| **V1** | yolo26s-seg baseline / 蒸馏 | 导出 + ultralytics 训练 | 🟡 桥接已建 | 待 GPU smoke test 验证 accept overlay 收益 |
| **V2** | 视频 masklet + temporal QA | 0–1, 4–5, 8 | ⚪ 未开始 | SA-V ingest 属此层；本地尚无 SA-V 数据 |
| **V3** | alpha / eval_map / fusion | 6–11 | 🟢 mock 已通 | MatAnyone/SEMat/VideoMaMa 真实 adapter 待接 |
| **V4** | 端侧 tracker + 轻量 matting | 部署 | ⚪ 后续 | — |

**当前聚焦**：V0 + V1。SA-V（V2）在 accept→蒸馏 smoke test 有信号后再开 adapter 骨架。

---

## 4. 12 阶段实现跟踪

| Step | 名称 | hmp 入口 | 集成目标（开源） | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 数据采样 | `hmp dataset ingest` / `coconut-sample` | SA-V, COCO-ReM, COCONut | 🟡 registry + COCONut sample 已有；SA-V ingest 未做 |
| 1 | 预处理分桶 | `hmp dataset stratify` | PySceneDetect, Decord | 🟡 heuristic tags |
| 2 | 人体发现 | `hmp label yolo-sam2` | YOLO, GroundingDINO | 🟡 YOLO person 已接 |
| 3 | Prompt Agent | `hmp agents prompt` | AlignSAM idea | 🟡 heuristic_v2 |
| 4 | masklet | `hmp label yolo-sam2` | SAM2, Cutie, XMem | 🟡 SAM2 bbox-only；视频 VOS 未接 |
| 5 | mask 精修 | `hmp refine masks` | SAMRefiner, HQ-SAM | 🟡 本地后处理；SAMRefiner 未接 |
| 6 | matting ROI | `hmp matting make-adaptive-trimap` | OpenCV, SCHP | 🟢 adaptive trimap 已实现 |
| 7 | 多 teacher alpha | `hmp matting process-queue` | MatAnyone, SEMat, VideoMaMa | 🟢 mock Bv/Bi/Bd/Bs |
| 8 | MQE | `hmp eval mqe` | RAFT, GMFlow, MMagic | 🟡 rule-based；learned MQE 未做 |
| 9 | Fusion Agent | `hmp agents fusion` / process-queue | RL fusion idea | 🟡 heuristic |
| 10 | HITL | `hmp relabel hitl-queue` | CVAT, FiftyOne | 🟡 JSONL 导出；平台未接 |
| 11 | 最终 label | `hmp relabel export-labels` | — | 🟢 AlphaLabelRecord manifest |
| 编排 | 全流程 | `hmp pipeline run-relabel` | — | 🟢 CPU mock E2E；GPU yolo_sam2 可选 |

图例：🟢 可用 · 🟡 部分/mock · ⚪ 未开始

---

## 5. 开源集成跟踪（对照 reference_integrations.yaml）

### 5.1 已接线（非 mock）

| ID | 用途 | 接入模块 | 限制 |
| --- | --- | --- | --- |
| `ultralytics_yolo` | person detector | `labeling/yolo_person_detector.py` | 端侧 yolo26s-seg 权重 |
| `sam2` | segment teacher | `labeling/sam2_adapter.py` + `sam_teacher.py` | 仅 bbox prompt |
| `samhq`（via models.yaml） | boundary 重标 | `sam_teacher.py`, `yolo_samhq` provider | 同 SAM2 API |
| `grabcut` | CPU ablation | `mock_sam2.py` | 非生产 |

### 5.2 计划 adapter（src/hmp/adapters/ 尚未创建）

| 优先级 | ID | Pipeline 阶段 | adapter 目标路径 | 状态 |
| ---: | --- | --- | --- | --- |
| P0 | `samrefiner` | 5 | `adapters/mask_refine/samrefiner.py` | ⚪ |
| P0 | `grounded_sam2` | 2 | `adapters/detection/grounded_sam2.py` | ⚪ |
| P1 | `cutie` / `xmem` | 4 | `adapters/vos/` | ⚪ |
| P1 | `raft` / `gmflow` | 8 | `adapters/qa/` | ⚪ |
| P2 | `matanyone` / `semat` | 7 | `adapters/matting/` | ⚪ |
| P2 | `videomama` / `diffmatte` | 7/9 | `adapters/diffusion/` | ⚪ |
| P3 | `fiftyone` / `cvat` | 10 | `adapters/hitl/` | ⚪ |
| P3 | `gymnasium` / `sb3` | 3/9 | `adapters/active_labeling/` | ⚪ |

### 5.3 V1 蒸馏（ultralytics 侧，非 hmp adapter）

| 链路 | 命令/脚本 | 状态 |
| --- | --- | --- |
| accept mask → YOLO val overlay | `hmp yolo export-accept-coconut` | 🟢 已实现 |
| 蒸馏启动计划 | `hmp yolo distill-plan` | 🟢 已实现 |
| 一键 | `bash scripts/export_accept_coconut_distill.sh` | 🟢 已实现 |
| 实际训练 | `ultralytics/scripts/train_yolo26s_seg_coconut_distill.py` | 🟡 待 smoke test |

---

## 6. COCONut benchmark 实测快照（128 图 / 350 inst）

| 模式 | mask IoU | accept |
| --- | ---: | ---: |
| gt_bbox + GrabCut | 0.383 | 3.1% |
| gt_bbox + SAM2 | 0.788 | 37.1% |
| yolo_person + GrabCut | 0.386 | 3.7% |
| **yolo_person + SAM2** | **0.771** | **34.6%** |
| yolo_person + SamHQ | 待 GPU 跑 | — |

生产迭代默认优先：`yolo_person + sam2`（`production_score_margin: 0.03`）。

产物目录：`runs/coconut_compare/` · 桥接配置：`configs/coconut_relabel.yaml`

---

## 7. 任务清单（Cursor 跟踪）

### 已完成

- [x] 12 阶段注册 + `run_relabel` 编排
- [x] COCONut benchmark + 五模式 compare（含 samhq 配置）
- [x] `benchmark_bridge`：bootstrap-from-benchmark、bad_boundary SamHQ 重标
- [x] model tiers（`configs/models.yaml` + `sam_teacher.py`）
- [x] accept → YOLO overlay + distill plan（`coconut_distill_bridge.py`）
- [x] E2E 脚本：`run_coconut_relabel_e2e.sh`（含可选 RELABEL_BOUNDARY + MQE）

### 进行中 / 下一步

- [ ] GPU 跑 `yolo_person×samhq` compare，量化边界 teacher 收益
- [ ] accept overlay + 短 epoch 蒸馏 smoke test（ultralytics）
- [ ] Stage 7 真实 alpha teacher adapter（Bv/Bi 优先 MatAnyone/SEMat dry-run）
- [ ] accept 样本上 `hmp eval mqe` + fuse-alpha 质量验证
- [ ] SA-V ingest adapter 骨架（dry-run，不依赖本地下载）
- [ ] `LABEL_SPEC_zh.md` + `configs/class_map.yaml`（CODE_TARGETS Phase 0）
- [ ] `src/hmp/adapters/` 基类 contract（command template + dry-run + output validation）

### 暂缓

- [ ] learned MQE（Phase 3）
- [ ] diffusion refine Bd 真实接入（Phase 2）
- [ ] RL prompt/fusion 训练（Phase 4）
- [ ] 端侧 video wrapper / matting student（V4）

---

## 8. 常用命令速查

```bash
# Benchmark
hmp eval coconut-compare -c configs/coconut_benchmark.yaml
hmp eval coconut-iterate -c configs/coconut_benchmark.yaml

# bad_boundary → SamHQ
hmp eval coconut-export-bad-boundary --benchmark-dir runs/coconut_compare/yolo_person__sam2
hmp eval coconut-relabel-boundary -c configs/coconut_relabel.yaml \
  --benchmark-dir runs/coconut_compare/yolo_person__sam2 --teacher samhq

# benchmark → pipeline
hmp pipeline bootstrap-from-benchmark -c configs/coconut_relabel.yaml
bash scripts/run_coconut_relabel_e2e.sh mock

# accept → 蒸馏
bash scripts/export_accept_coconut_distill.sh

# 测试
cd segPipeline && PYTHONPATH=src pytest -q --ignore=tests/test_alpha_metrics.py
```

环境：`hmp-py310`（pytest/CPU）· `yolo26-cu133`（YOLO/SAM2 GPU）

---

## 9. 会话变更 log

| 日期 | 内容 |
| --- | --- |
| 2026-07-05 | 创建本文件；梳理 V0–V4 × 12 阶段 × 开源集成状态；不修改 canonical 文档 |
| 2026-07-05 | 前序：SamHQ benchmark、bad_boundary 重标、accept→蒸馏桥接已合入 `iter/prompt-agent-v2` |

---

## 10. 待讨论项

1. **SA-V vs 蒸馏**：V0 accept→蒸馏 smoke test 通过后再开 SA-V adapter，是否同意？
2. **SamHQ 权重**：`sam_hq_vit_b.pt` 走 Ultralytics SAM API 还是 external subprocess？
3. **accept 阈值**：当前 IoU≥0.85 & BF1≥0.85 偏严（~35% accept），蒸馏 overlay 是否同时纳入 `review` 银标？
4. **canonical 文档**：重大里程碑时再手动同步 PIPELINE_zh / OPEN_SOURCE_INTEGRATION_TARGETS，本文件作中间态。

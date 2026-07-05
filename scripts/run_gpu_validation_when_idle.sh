#!/usr/bin/env bash
# Run GPU validation jobs only when at least one GPU has enough free memory.
# Safe to launch while DDP training occupies other GPUs — skips if all GPUs busy.
#
# Usage:
#   bash scripts/run_gpu_validation_when_idle.sh
#   MIN_FREE_MIB=18000 bash scripts/run_gpu_validation_when_idle.sh
#   WAIT=1 bash scripts/run_gpu_validation_when_idle.sh    # poll every 5min until idle
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
POLL_SEC="${POLL_SEC:-300}"
PYTHON="${PYTHON:-/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
LOG="${LOG:-runs/validation/gpu_validation.log}"
mkdir -p "$(dirname "$LOG")"
DATA_YAML="${DATA_YAML:-$ROOT/data/coconut/yolo_accept_overlay/data.yaml}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=',' read -r idx free; do
    idx="${idx// /}"
    free="${free// /}"
    if [ "$free" -ge "$MIN_FREE_MIB" ]; then
      echo "$idx"
      return 0
    fi
  done
  return 1
}

wait_for_gpu() {
  while true; do
    GPU="$(pick_gpu || true)"
    if [ -n "${GPU:-}" ]; then
      echo "$GPU"
      return 0
    fi
    echo "[wait] all GPUs < ${MIN_FREE_MIB} MiB free; sleep ${POLL_SEC}s ($(date -Is))" | tee -a "$LOG"
    sleep "$POLL_SEC"
  done
}

if [ "${WAIT:-0}" = "1" ]; then
  GPU="$(wait_for_gpu)"
else
  GPU="$(pick_gpu || true)"
fi

if [ -z "${GPU:-}" ]; then
  echo "[skip] all GPUs have < ${MIN_FREE_MIB} MiB free — DDP training likely active" | tee -a "$LOG"
  echo "  nvidia-smi snapshot:" | tee -a "$LOG"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$LOG"
  exit 0
fi

echo "[run] using GPU ${GPU} (min_free=${MIN_FREE_MIB} MiB)" | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "==> yolo_person × samhq benchmark (128 images)" | tee -a "$LOG"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "$LOG"
from pathlib import Path
from hmp.config import load_config, Config
from hmp.eval.benchmark_compare import run_coconut_compare

base = load_config("configs/coconut_benchmark.yaml").to_dict()
base["coconut_compare"] = {
    **base.get("coconut_compare", {}),
    "skip_existing": True,
    "force_rerun": False,
    "modes": [["yolo_person", "samhq"]],
}
base["coconut_benchmark"] = {
    **base.get("coconut_benchmark", {}),
    "device": 0,
}
cfg = Config(base)
run_coconut_compare(cfg, project_root=Path.cwd())
print("done: runs/coconut_compare/yolo_person__samhq/")
PY

echo "==> resummarize compare (all modes incl. samhq)" | tee -a "$LOG"
"${PYTHON}" -m hmp.cli eval coconut-iterate --config configs/coconut_benchmark.yaml 2>&1 | tee -a "$LOG"

if [ -f "$DATA_YAML" ]; then
  echo "==> distill smoke (2 epochs, batch=8, single GPU)" | tee -a "$LOG"
  STUDENT="${STUDENT:-/home/genesis/Train/Code/ultralytics/runs/segment/yolo26s-seg-lvis-coco80-distill-x-teacher-b80-2gpu/weights/best.pt}"
  TEACHER="${TEACHER:-/home/genesis/Train/Code/ultralytics/yolo26x-seg.pt}"
  "${PYTHON}" /home/genesis/Train/Code/ultralytics/scripts/train_yolo26s_seg_coconut_distill.py \
    --data "$DATA_YAML" \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --name smoke-coconut-accept-overlay-distill \
    --epochs 2 \
    --batch 8 \
    --device 0 \
    --workers 4 \
    --patience 100 \
    --no-swanlab \
    --exist-ok \
    2>&1 | tee -a "$LOG"
fi

echo "[done] see $LOG and runs/coconut_compare/compare_summary.md" | tee -a "$LOG"

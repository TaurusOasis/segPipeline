#!/usr/bin/env bash
# COCONut benchmark -> relabel pipeline bridge (steps 0-11).
#
# Prerequisites:
#   1. runs/coconut_compare/yolo_person__sam2/  (from scripts/run_coconut_compare.sh)
#   2. yolo26-cu133 for --provider yolo_sam2 live labeling; mock works on CPU for stages 5-11
#
# Usage:
#   bash scripts/run_coconut_relabel_e2e.sh [mock|yolo_sam2]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PROVIDER="${1:-mock}"
CONFIG="${CONFIG:-$ROOT/configs/coconut_relabel.yaml}"

if [ -x "/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp" ]; then
  HMP="/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp"
elif command -v hmp >/dev/null 2>&1; then
  HMP="hmp"
else
  export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
  HMP="python -m hmp.cli"
fi

if [ "$PROVIDER" = "yolo_sam2" ]; then
  PYTHON="${PYTHON:-/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python}"
  export PATH="$(dirname "$PYTHON"):$PATH"
  export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
  HMP="${PYTHON} -m hmp.cli"
fi

echo "==> Bootstrap manifest/annotations/HITL from benchmark (yolo_person__sam2)"
$HMP pipeline bootstrap-from-benchmark --config "$CONFIG"

echo "==> Run relabel pipeline stages 5-11 ($PROVIDER alpha teachers on accept/review masks)"
$HMP pipeline run-relabel --config "$CONFIG" --provider "$PROVIDER" --from-stage 5 --to-stage 11

echo ""
echo "COCONut relabel e2e complete."
echo "  manifest:     data/coconut/manifests/val128.jsonl"
echo "  annotations:  data/coconut/annotations/annotations_raw.jsonl"
echo "  relabel q:    data/coconut/alpha/relabel_queue.jsonl"
echo "  HITL q:       data/coconut/alpha/hitl_queue.jsonl"
echo "  alpha labels: data/coconut/alpha/alpha_labels.jsonl"

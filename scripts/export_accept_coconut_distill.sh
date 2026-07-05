#!/usr/bin/env bash
# Export benchmark accept masks into COCONut YOLO val overlay + print distill command.
#
# Prerequisites:
#   runs/coconut_compare/yolo_person__sam2/  (benchmark with annotations_pred.jsonl)
#   /home/genesis/Train/Dataset/COCONut_b_yolo_seg_v2
#
# Usage:
#   bash scripts/export_accept_coconut_distill.sh
#   bash scripts/export_accept_coconut_distill.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-$ROOT/configs/coconut_distill_bridge.yaml}"
DRY=()
if [ "${1:-}" = "--dry-run" ]; then
  DRY=(--dry-run)
fi

if [ -x "/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp" ]; then
  HMP="/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp"
elif command -v hmp >/dev/null 2>&1; then
  HMP="hmp"
else
  export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
  HMP="python -m hmp.cli"
fi

echo "==> Overlay accept masks onto COCONut YOLO val labels"
$HMP yolo export-accept-coconut --config "$CONFIG" "${DRY[@]}"

echo "==> Distillation launch plan (yolo26x-seg -> yolo26s-seg)"
$HMP yolo distill-plan --config "$CONFIG"

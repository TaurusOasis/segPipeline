#!/usr/bin/env bash
# Compare auto-label pipeline masks against COCONut GT on relabeled_coco_val.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/coconut_benchmark.yaml}"
hmp eval coconut-benchmark --config "$CONFIG"
echo "Summary: runs/coconut_benchmark/benchmark_summary.md"
echo "Masks: runs/coconut_benchmark/{pred_masks,gt_masks,diff_masks}"
echo "Contact sheet: runs/coconut_benchmark/contact_sheet_worst.png"

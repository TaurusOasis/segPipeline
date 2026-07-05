#!/usr/bin/env bash
# Run multi-mode COCONut benchmark comparison (gt/yolo x grabcut/sam2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/coconut_benchmark.yaml}"
PYTHON="${PYTHON:-/home/genesis/Tools/Anaconda/envs/yolo26-cu133/bin/python}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
"${PYTHON}" -m hmp.cli eval coconut-iterate --config "$CONFIG"
echo "Summary: runs/coconut_compare/compare_summary.md"
echo "Plan: runs/coconut_compare/iteration_plan.json"
echo "Patch: runs/coconut_compare/next_config_patch.yaml"

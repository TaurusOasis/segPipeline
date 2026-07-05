#!/usr/bin/env bash
# CPU-only demo pipeline for human-matting-pipeline (hmp).
# Runs end-to-end WITHOUT any GPU, model weights, SAM, YOLO, or RKNN:
#   fixtures -> manifest -> dummy labels -> local refine -> YOLO seg export
#            -> trimaps -> evaluation report
#
# Usage: bash scripts/run_demo_pipeline.sh [PROJECT_ROOT]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

# Pick the hmp env if available, else fall back to the active python.
if [ -x "/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp" ]; then
  HMP="/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp"
  PY="/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/python"
elif command -v hmp >/dev/null 2>&1; then
  HMP="hmp"; PY="python"
else
  echo "hmp CLI not found. Activate the hmp env or run 'pip install -e .' first." >&2
  exit 1
fi

CONFIG="$ROOT/configs/demo.yaml"

echo "==> [1/7] Generate tiny fixture images"
$PY - <<PYEOF
from pathlib import Path
from PIL import Image, ImageDraw
root = Path("$ROOT/data/demo_raw")
root.mkdir(parents=True, exist_ok=True)
for i in range(8):
    img = Image.new("RGB", (96, 128), (40 + i * 5, 60, 80))
    d = ImageDraw.Draw(img)
    # draw a vague "person" silhouette so the dummy labeler has something
    d.ellipse([36, 20, 60, 44], fill=(200, 180, 160))     # head
    d.rectangle([34, 44, 62, 110], fill=(60, 90, 160))      # body
    img.save(root / f"demo_{i:02d}.jpg")
print("wrote", len(list(root.glob("*.jpg"))), "fixture images")
PYEOF

echo "==> [2/7] Build manifest"
$HMP manifest build --config "$CONFIG" --overwrite

echo "==> [3/7] Run dummy labeler (centered-rectangle person masks)"
$HMP label dummy --config "$CONFIG"

echo "==> [4/7] Refine masks (local postprocess only)"
$HMP refine masks --config "$CONFIG"

echo "==> [5/7] Export YOLO segmentation dataset"
$HMP dataset export-yolo --config "$CONFIG"

echo "==> [6/7] Generate trimaps"
$HMP matting make-trimap --config "$CONFIG"

echo "==> [7/7] Build evaluation report"
$HMP eval report --config "$CONFIG"

echo ""
echo "Demo pipeline complete. Artifacts under $ROOT/data/demo_* and runs/eval_report.md"
echo "  - manifest:        data/demo_manifests/manifest.jsonl"
echo "  - raw masks:       data/demo_masks_raw/"
echo "  - refined masks:   data/demo_masks_refined/"
echo "  - YOLO seg:        data/demo_yolo_seg/ (images/ labels/ data.yaml)"
echo "  - trimaps:         data/demo_alpha/trimaps/"
echo "  - report:          runs/eval_report.md"
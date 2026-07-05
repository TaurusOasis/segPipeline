#!/usr/bin/env bash
# Full CPU-only 11-stage relabel demo for human-matting-pipeline (hmp).
# Runs end-to-end WITHOUT GPU / external model weights:
#   fixtures -> manifest -> ingest/stratify -> label -> refine -> adaptive trimap
#   -> relabel queue -> mock Bv/Bi/Bg/Bs -> MQE/fusion -> HITL queue -> alpha labels
#
# Usage: bash scripts/run_demo_relabel_pipeline.sh [PROJECT_ROOT]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

if [ -x "/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp" ]; then
  HMP="/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/hmp"
  PY="/home/genesis/Tools/Anaconda/envs/hmp-py310/bin/python"
elif command -v hmp >/dev/null 2>&1; then
  HMP="hmp"; PY="python"
else
  echo "hmp CLI not found. Run 'pip install -e .' first." >&2
  exit 1
fi

CONFIG="$ROOT/configs/demo_relabel.yaml"

echo "==> Generate tiny fixture images"
$PY - <<PYEOF
from pathlib import Path
from PIL import Image, ImageDraw
root = Path("$ROOT/data/demo_raw")
root.mkdir(parents=True, exist_ok=True)
for i in range(8):
    img = Image.new("RGB", (96, 128), (40 + i * 5, 60, 80))
    d = ImageDraw.Draw(img)
    d.ellipse([36, 20, 60, 44], fill=(200, 180, 160))
    d.rectangle([34, 44, 62, 110], fill=(60, 90, 160))
    img.save(root / f"demo_{i:02d}.jpg")
print("wrote", len(list(root.glob("*.jpg"))), "fixture images")
PYEOF

echo "==> Run full 11-stage relabel pipeline (mock alpha teachers)"
$HMP pipeline run-relabel --config "$CONFIG" --provider mock

echo ""
echo "Relabel demo complete. Key artifacts:"
echo "  - manifest:       data/demo_manifests/manifest.jsonl"
echo "  - refined masks:  data/demo_masks_refined/"
echo "  - adaptive ROI:   data/demo_alpha/roi/"
echo "  - relabel queue:  data/demo_alpha/relabel_queue.jsonl"
echo "  - branch alphas:  data/demo_alpha/branches/"
echo "  - fused alphas:   data/demo_alpha/fused/"
echo "  - eval maps:      data/demo_alpha/eval_maps/"
echo "  - HITL queue:     data/demo_alpha/hitl_queue.jsonl"
echo "  - alpha labels:   data/demo_alpha/alpha_labels.jsonl"

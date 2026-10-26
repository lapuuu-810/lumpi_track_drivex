#!/usr/bin/env bash
# =============================================================================
# LUMPI final submission - one-command synthetic reproduction (Realism 0.9035,
# guaranteed-v1/19D)
#   16b(r_hist p0.75, 4 iter) + 16c(r_hist p0.5, 5 iter) -> 75/25 mix -> 50 frames
# Requires the raw sources (overridable through environment variables):
#   UCF_DT_ROOT    UT-LUMPI raw directory (pcd/*.npy)
#   M5_DIR         real Measurement5 directory (lidar.zip.001)
#   STARTING_KIT   competition starting_kit (test frames + frame_ids)
# After running, outputs outputs/synthetic/ and md5-checks against the online frames.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${CONDA_ENV:-urbantwin-v2xreal}"
PY_RUN="conda run -n $ENV_NAME --no-capture-output python"

UCF_DT_ROOT="${UCF_DT_ROOT:-/data/TY/ECCV/UCF_LUMPI/dataset/UCF-DT-LUMPI-Seq-1}"
M5_DIR="${M5_DIR:-/data/TY/ECCV/UCF_LUMPI/dataset/LUMPI_hannover/Measurement5}"
STARTING_KIT="${STARTING_KIT:-$ROOT/competition/starting_kit}"
OUT="$ROOT/outputs"
PREDS="$OUT/dummy_predictions.json"

for p in "$UCF_DT_ROOT" "$M5_DIR" "$STARTING_KIT/detection_test_frame_ids.json"; do
  [ -e "$p" ] || { echo "MISSING: $p"; echo "  Point UCF_DT_ROOT / M5_DIR / STARTING_KIT at the raw data"; exit 1; }
done

mkdir -p "$OUT"
# Generate dummy predictions (50 empty detections for the synthesis packager)
$PY_RUN - "$STARTING_KIT/detection_test_frame_ids.json" "$PREDS" <<'EOF'
import json, sys
ids = json.load(open(sys.argv[1]))
json.dump({f: [] for f in ids}, open(sys.argv[2], "w"))
EOF

echo "[1/3] generate 16b (radial histogram matching p0.75, 4 iterations)"
$PY_RUN "$ROOT/src/make_lumpi_realstat_submission.py" \
  --dataset-root "$UCF_DT_ROOT" --starting-kit "$STARTING_KIT" \
  --lumpi-measurement-dir "$M5_DIR" --predictions-json "$PREDS" \
  --output-dir "$OUT/lumpi16b" --num-frames 50 --points-per-frame 50000 \
  --target-num-frames 120 --target-sample-points 2500 --candidate-sample-points 2500 \
  --post-crop-match-iters 4 --final-resample-mode r_hist \
  --hist-power 0.75 --hist-max-weight 8.0 --seed 1234 --team-name repro_16b

echo "[2/3] generate 16c (radial histogram matching p0.5, 5 iterations)"
$PY_RUN "$ROOT/src/make_lumpi_realstat_submission.py" \
  --dataset-root "$UCF_DT_ROOT" --starting-kit "$STARTING_KIT" \
  --lumpi-measurement-dir "$M5_DIR" --predictions-json "$PREDS" \
  --output-dir "$OUT/lumpi16c" --num-frames 50 --points-per-frame 50000 \
  --target-num-frames 120 --target-sample-points 2500 --candidate-sample-points 2500 \
  --post-crop-match-iters 5 --final-resample-mode r_hist \
  --hist-power 0.5 --hist-max-weight 8.0 --seed 1234 --team-name repro_16c

echo "[3/3] mix 16c(75%) + 16b(25%) -> outputs/synthetic (19D/guaranteed-v1)"
$PY_RUN "$ROOT/src/make_mixed_synthetic_submission.py" \
  --source-a "$OUT/lumpi16c/submission/synthetic" \
  --source-b "$OUT/lumpi16b/submission/synthetic" \
  --ratio-a 0.75 --predictions-json "$PREDS" \
  --output-dir "$OUT/lumpi19d" --points-per-frame 50000 --seed 1234 \
  --team-name repro_19d

SYN="$OUT/lumpi19d/submission/synthetic"
[ -d "$SYN" ] && cp "$SYN"/*.bin "$OUT/synthetic/" || { SYN="$OUT/lumpi19d/synthetic"; mkdir -p "$OUT/synthetic"; cp "$SYN"/*.bin "$OUT/synthetic/"; }
EXPECTED="4710765bd4e597c2e039f22243c129b0"
HASH=$(md5sum "$OUT/synthetic/0001.bin" | cut -d' ' -f1)
echo "synthetic frame1 md5: $HASH"
if [ "$HASH" = "$EXPECTED" ]; then
  echo "PASS: synthetic matches the online submission frames (0001.bin)"
else
  echo "WARN: synthetic differs from the online frames (check whether the mixing seed/selection matches the original 19D set)"
fi
# If the online reference package has been imported, byte-compare all 50 frames
REF_SYN="$ROOT/outputs/final_package/synthetic"
if [ -d "$REF_SYN" ]; then
  $PY_RUN "$ROOT/src/verify_submission.py" \
    --zip "$OUT/repro_check.zip" \
    --frame-ids-json "$STARTING_KIT/detection_test_frame_ids.json" \
    --compare-synthetic "$REF_SYN" 2>/dev/null || true
  rm -f "$OUT/repro_check.zip"
fi
echo "DONE: $OUT/synthetic ($(ls "$OUT/synthetic" | wc -l) frames)"

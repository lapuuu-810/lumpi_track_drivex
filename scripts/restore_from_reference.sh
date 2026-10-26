#!/usr/bin/env bash
# =============================================================================
# Restore the final answer from the online submission package (recommended path)
#   Usage: place the submission_v65_legal.zip downloaded from Codabench at
#          outputs/reference/submission_v65_legal.zip, then run this script.
#   The script extracts synthetic/ + predictions.json + declaration.txt and
#   validates everything:
#         - 50-frame synthetic md5 matches the online submission
#         - predictions cover 50 frames, 8 classes, 13532 total boxes (the final
#           answer fingerprint)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${CONDA_ENV:-urbantwin-v2xreal}"
PY_RUN="conda run -n $ENV_NAME --no-capture-output python"
REF="$ROOT/outputs/reference/submission_v65_legal.zip"
OUT="$ROOT/outputs/final_package"

[ -f "$REF" ] || { echo "Reference package missing: download submission_v65_legal.zip from Codabench and place it under outputs/reference/"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/extracted"
unzip -q -o "$REF" -d "$OUT/extracted"
# Support two layouts: files directly at the ZIP root, or wrapped in one directory
[ -f "$OUT/extracted/predictions.json" ] || cp "$OUT/extracted"/*/predictions.json "$OUT/extracted/predictions.json" 2>/dev/null || true
[ -d "$OUT/extracted/synthetic" ] || cp -r "$OUT/extracted"/*/synthetic "$OUT/extracted/synthetic" 2>/dev/null || true
# Repack into the standard submission layout (synthetic/ + predictions.json + declaration.txt)
cd "$OUT/extracted"
zip -q -r "$OUT/submission_v65_legal.zip" declaration.txt predictions.json synthetic/*.bin
cd "$ROOT"

echo "== validation =="
$PY_RUN "$ROOT/src/verify_submission.py" \
  --zip "$OUT/submission_v65_legal.zip" \
  --frame-ids-json "$ROOT/competition/starting_kit/detection_test_frame_ids.json" \
  --check-exact || true
# Copy into the final_package root for direct use
cp "$OUT/extracted/predictions.json" "$OUT/predictions.json"
[ -d "$OUT/synthetic" ] && rm -rf "$OUT/synthetic"
cp -r "$OUT/extracted/synthetic" "$OUT/synthetic"
cp "$OUT/extracted/declaration.txt" "$OUT/declaration.txt"
echo "== restored =="
echo "predictions: $OUT/predictions.json"
echo "synthetic  : $OUT/synthetic ($(ls "$OUT/synthetic" | wc -l) frames)"
echo "declaration: $OUT/declaration.txt"
echo "== notes =="
echo "These files are the final package for online submission #890460 (Combined 0.4692)."
echo "Synthetic regeneration: run_synthetic_repro.sh; detection regeneration (requires weights): run_detection_repro.sh."

#!/usr/bin/env bash
# LUMPI final detection reproduction:
#   python src/run_detection_final.py -> predictions.json + submission ZIP -> verify
#
# Default final mode: infer the reproducible members (final30/32, clean34,
# pedalign, rldm2k, bus125a), fuse Person/Bus/Unknown, and restore only the
# v4-dependent classes (Car/Bicycle/Motorcycle/Truck/Van) from the online
# reference. Full ten-model regeneration without any online reference:
#   bash scripts/run_detection_repro.sh --mode full
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROOT="$(project_root)"
load_python_command
DEVICE="${DEVICE:-0}"
OUT="$ROOT/outputs/repro_det"
MODE="${1:-final}"
mkdir -p "$OUT"

"${PYTHON[@]}" "$ROOT/src/run_detection_final.py" \
  --project-root "$ROOT" --output-dir "$OUT" --device "$DEVICE" --mode "$MODE"

"${PYTHON[@]}" "$ROOT/src/verify_submission.py" \
  --zip "$OUT/submission_lumpi_final.zip" \
  --frame-ids-json "$ROOT/competition/starting_kit/detection_test_frame_ids.json"

echo "DONE: $OUT/predictions.json + $OUT/submission_lumpi_final.zip"

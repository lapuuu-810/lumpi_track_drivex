#!/usr/bin/env bash
# Run 10 model exports, v6.5 class-wise fusion, and final-package validation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROOT="$(project_root)"
load_python_command
DEVICE="${DEVICE:-0}"
DATA_ROOT="${LUMPI_DATA_ROOT:-$ROOT/data}"
FORCE_PARTIAL="${1:-}" == "--force-partial" && FORCE_PARTIAL=1 || FORCE_PARTIAL=0
TEST_FRAMES="$ROOT/competition/starting_kit/detection_test_frames"
FRAME_IDS="$ROOT/competition/starting_kit/detection_test_frame_ids.json"
CHECKPOINTS="$ROOT/third_party/ckpts"
OUTPUT="$ROOT/outputs/repro_det"
RUNTIME_CONFIGS="$OUTPUT/runtime_configs"
OPENCDET="$ROOT/third_party/OpenPCDet"
mkdir -p "$OUTPUT" "$RUNTIME_CONFIGS"

declare -A checkpoint=(
  [final30]="$CHECKPOINTS/dsvt_final_epoch30.pth" [final32]="$CHECKPOINTS/dsvt_final_epoch32.pth"
  [clean34]="$CHECKPOINTS/dsvt_clean_epoch34.pth" [pedalign]="$CHECKPOINTS/dsvt_pedalign_epoch40.pth"
  [v4e36]="$CHECKPOINTS/dsvt_v4_epoch36.pth" [v4e38]="$CHECKPOINTS/dsvt_v4_epoch38.pth"
  [v4e40]="$CHECKPOINTS/dsvt_v4_epoch40.pth" [rldm2k]="$CHECKPOINTS/dsvt_rldm2k_epoch40.pth"
  [car65a]="$CHECKPOINTS/ucf_car65a_e8.pth" [bus125a]="$CHECKPOINTS/ucf_bus125a_e8.pth"
)
declare -A config=(
  [final30]=dsvt_vfield_g0s_lumpi_final.yaml [final32]=dsvt_vfield_g0s_lumpi_final.yaml
  [clean34]=dsvt_vfield_g0s_lumpi_clean.yaml [pedalign]=dsvt_vfield_g0s_lumpi_pedalign.yaml
  [v4e36]=dsvt_vfield_g0s_lumpi_v4.yaml [v4e38]=dsvt_vfield_g0s_lumpi_v4.yaml
  [v4e40]=dsvt_vfield_g0s_lumpi_v4.yaml [rldm2k]=dsvt_vfield_g0s_lumpi_d50k_rldm2k.yaml
  [car65a]=ucf_models/pointpillar_lumpi_car_only.yaml [bus125a]=ucf_models/pointpillar_lumpi_bus_only_realsize.yaml
)
models=(final30 final32 clean34 pedalign v4e36 v4e38 v4e40 rldm2k car65a bus125a)

[[ -d "$TEST_FRAMES" ]] || { echo "Missing test frames: $TEST_FRAMES" >&2; exit 1; }
missing=0
for model in "${models[@]}"; do
  [[ -f "${checkpoint[$model]}" ]] || { echo "Missing checkpoint: ${checkpoint[$model]}" >&2; missing=1; }
done
if (( missing )) && (( ! FORCE_PARTIAL )); then
  echo "Use --force-partial to export only the checkpoints that are available." >&2
  exit 1
fi

for model in "${models[@]}"; do
  [[ -f "${checkpoint[$model]}" ]] || continue
  runtime_config="$RUNTIME_CONFIGS/$model.yaml"
  "${PYTHON[@]}" "$ROOT/src/prepare_runtime_config.py" --config "$ROOT/configs/${config[$model]}" --project-root "$ROOT" --data-root "$DATA_ROOT" --output "$runtime_config"
  "${PYTHON[@]}" "$ROOT/src/export_opencd_predictions.py" --opencd-root "$OPENCDET" --cfg-file "$runtime_config" --ckpt "${checkpoint[$model]}" --test-frames-dir "$TEST_FRAMES" --frame-ids-json "$FRAME_IDS" --output-json "$OUTPUT/preds_$model.json" --score-thresh 0.05 --device "$DEVICE"
done

if (( ! FORCE_PARTIAL )); then
  "${PYTHON[@]}" "$ROOT/src/build_lumpi_final.py" --final32 "$OUTPUT/preds_final32.json" --clean34 "$OUTPUT/preds_clean34.json" --final30 "$OUTPUT/preds_final30.json" --v4e40 "$OUTPUT/preds_v4e40.json" --v4e38 "$OUTPUT/preds_v4e38.json" --v4e36 "$OUTPUT/preds_v4e36.json" --pedalign "$OUTPUT/preds_pedalign.json" --car65a "$OUTPUT/preds_car65a.json" --bus125a "$OUTPUT/preds_bus125a.json" --rldm2k "$OUTPUT/preds_rldm2k.json" --test-frames-dir "$TEST_FRAMES" --frame-ids-json "$FRAME_IDS" --synthetic-dir "$ROOT/outputs/synthetic" --declaration "$ROOT/declaration.txt" --output-dir "$OUTPUT"
  "${PYTHON[@]}" "$ROOT/src/verify_submission.py" --zip "$OUTPUT/submission_lumpi_final.zip" --frame-ids-json "$FRAME_IDS" --check-exact
fi

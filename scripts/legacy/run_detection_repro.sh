#!/usr/bin/env bash
# =============================================================================
# LUMPI final submission - one-command detection reproduction (Detection 0.1797,
# mAP 0.1258, #890460)
#   10 model exports -> v6.5 class-wise fusion + calibration ->
#   predictions.json + submission ZIP -> validation
# Requires 10 checkpoints (third_party/ckpts/; 5 DSVT + rldm2k must be restored
# from the 140 backup, see docs/REPRO_GUIDE.md; the 3 UCF checkpoints ship with
# the package).
# Usage: bash scripts/run_detection_repro.sh [--force-partial]
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${CONDA_ENV:-urbantwin-v2xreal}"
DEVICE="${DEVICE:-0}"
if command -v conda >/dev/null 2>&1; then
  PY_RUN="${PY_RUN:-conda run -n $ENV_NAME --no-capture-output python}"
else
  PY_RUN="${PY_RUN:-/opt/conda/envs/$ENV_NAME/bin/python}"
fi
FORCE_PARTIAL=0
[ "${1:-}" = "--force-partial" ] && FORCE_PARTIAL=1

TEST_FRAMES="$ROOT/competition/starting_kit/detection_test_frames"
FRAME_IDS="$ROOT/competition/starting_kit/detection_test_frame_ids.json"
OUT="$ROOT/outputs/repro_det"
CK="$ROOT/third_party/ckpts"
# ---- portable path setup ----------------------------------------------------
sed -i "s|__ROOT__|$ROOT|g" "$ROOT"/configs/*.yaml 2>/dev/null || true
sed -i "s|/data2/RS3D/LPP/urbantwin_lumpi/configs/dsvt_vfield_g0s_lumpi.yaml|$ROOT/configs/dsvt_vfield_g0s_lumpi.yaml|g" "$ROOT"/configs/*.yaml 2>/dev/null || true
ln -sfn "$ROOT/configs/ucf_models" "$ROOT/cfgs" 2>/dev/null || true
mkdir -p "$ROOT/cfgs/dataset_configs"
cp -f "$ROOT/configs/ucf_models/lumpi_custom_dataset.yaml" "$ROOT/cfgs/dataset_configs/" 2>/dev/null || true
OPENC="$ROOT/third_party/OpenPCDet"
mkdir -p "$OUT"

# ---- checkpoint manifest -----------------------------------------------------
declare -A CKS=(
  [final30]="$CK/dsvt_final_epoch30.pth"
  [final32]="$CK/dsvt_final_epoch32.pth"
  [clean34]="$CK/dsvt_clean_epoch34.pth"
  [pedalign]="$CK/dsvt_pedalign_epoch40.pth"
  [v4e36]="$CK/dsvt_v4_epoch36.pth"
  [v4e38]="$CK/dsvt_v4_epoch38.pth"
  [v4e40]="$CK/dsvt_v4_epoch40.pth"
  [rldm2k]="$CK/dsvt_rldm2k_epoch40.pth"
  [car65a]="$CK/ucf_car65a_e8.pth"
  [bus125a]="$CK/ucf_bus125a_e8.pth"
)
declare -A CFGS=(
  [final30]="dsvt_vfield_g0s_lumpi_final.yaml"
  [final32]="dsvt_vfield_g0s_lumpi_final.yaml"
  [clean34]="dsvt_vfield_g0s_lumpi_clean.yaml"
  [pedalign]="dsvt_vfield_g0s_lumpi_pedalign.yaml"
  [v4e36]="dsvt_vfield_g0s_lumpi_v4.yaml"
  [v4e38]="dsvt_vfield_g0s_lumpi_v4.yaml"
  [v4e40]="dsvt_vfield_g0s_lumpi_v4.yaml"
  [rldm2k]="dsvt_vfield_g0s_lumpi_d50k_rldm2k.yaml"
  [car65a]="ucf_models/pointpillar_lumpi_car_only.yaml"
  [bus125a]="ucf_models/pointpillar_lumpi_bus_only_realsize.yaml"
)
# u55a is optional (archived, not part of the final ensemble)
CKS[ucf55a]="$CK/ucf_7cls55a_e2.pth"
CFGS[ucf55a]="ucf_models/pointpillar_lumpi_7cls.yaml"

echo "=============================================================="
echo " LUMPI detection reproduction | root=$ROOT | device=$DEVICE"
echo "=============================================================="
missing=0
for n in final30 final32 clean34 pedalign v4e36 v4e38 v4e40 rldm2k car65a bus125a; do
  [ -f "${CKS[$n]}" ] || { echo "MISSING checkpoint: ${CKS[$n]}"; missing=1; }
done
if [ "$missing" = "1" ]; then
  echo "Some checkpoints are missing (5 DSVT + rldm2k must be restored from the 140 backup; see docs/REPRO_GUIDE.md)."
  if [ "$FORCE_PARTIAL" != "1" ]; then
    echo "Aborted. Restore the weights and rerun, or add --force-partial to use only the available models (result will differ from the final answer)."
    exit 1
  fi
  echo "--force-partial: continuing with the available models (the result will differ from the final answer)."
fi

[ -d "$TEST_FRAMES" ] || { echo "MISSING test frames: $TEST_FRAMES"; exit 1; }

infer_one() {
  local name="$1"
  local cfg="$ROOT/configs/${CFGS[$name]}"
  [ -f "${CKS[$name]}" ] || { echo "skip $name (ckpt missing)"; return; }
  echo "[infer] $name"
  $PY_RUN "$ROOT/src/export_opencd_predictions.py" \
    --opencd-root "$OPENC" --cfg-file "$cfg" --ckpt "${CKS[$name]}" \
    --test-frames-dir "$TEST_FRAMES" --frame-ids-json "$FRAME_IDS" \
    --output-json "$OUT/preds_${name}.json" --score-thresh 0.05 --device "$DEVICE"
}

for n in final30 final32 clean34 pedalign v4e36 v4e38 v4e40 rldm2k car65a bus125a; do
  infer_one "$n"
done

# ---- v6.5 fusion + calibration + packaging -----------------------------------
echo "[fuse] v6.5 class-wise fusion + calibration + packaging"
$PY_RUN "$ROOT/src/build_lumpi_final.py" \
  --final32 "$OUT/preds_final32.json" --clean34 "$OUT/preds_clean34.json" \
  --final30 "$OUT/preds_final30.json" --v4e40 "$OUT/preds_v4e40.json" \
  --v4e38 "$OUT/preds_v4e38.json" --v4e36 "$OUT/preds_v4e36.json" \
  --pedalign "$OUT/preds_pedalign.json" --car65a "$OUT/preds_car65a.json" \
  --bus125a "$OUT/preds_bus125a.json" --rldm2k "$OUT/preds_rldm2k.json" \
  --test-frames-dir "$TEST_FRAMES" --frame-ids-json "$FRAME_IDS" \
  --synthetic-dir "$ROOT/outputs/synthetic" \
  --declaration "$ROOT/declaration.txt" --output-dir "$OUT"

echo "[verify]"
$PY_RUN "$ROOT/src/verify_submission.py" --zip "$OUT/submission_lumpi_final.zip" \
  --frame-ids-json "$FRAME_IDS" --check-exact || echo "NOTE: counts differ from the final answer (possibly missing weights)"
echo "DONE: $OUT/predictions.json + $OUT/submission_lumpi_final.zip"

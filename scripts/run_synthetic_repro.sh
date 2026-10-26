#!/usr/bin/env bash
# LUMPI synthetic-frame reproduction:
#   python src/run_synthetic_final.py -> outputs/synthetic (50 frames) -> verify
#
# Raw sources are required for generation:
#   UCF_DT_ROOT=/path/to/UCF-DT-LUMPI-Seq-1 M5_DIR=/path/to/Measurement5
#     bash scripts/run_synthetic_repro.sh
# Internal dry-run (uses the bundled canonical frames):
#   bash scripts/run_synthetic_repro.sh --use-bundled
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROOT="$(project_root)"
load_python_command
OUT="$ROOT/outputs"
mkdir -p "$OUT"

SYN_ARGS=(--project-root "$ROOT" --output-dir "$OUT")
if [[ -n "${UCF_DT_ROOT:-}" ]]; then SYN_ARGS+=(--ucf-dt-root "$UCF_DT_ROOT"); fi
if [[ -n "${M5_DIR:-}" ]]; then SYN_ARGS+=(--m5-dir "$M5_DIR"); fi
if [[ "${1:-}" == "--use-bundled" ]]; then SYN_ARGS+=(--use-bundled); fi

"${PYTHON[@]}" "$ROOT/src/run_synthetic_final.py" "${SYN_ARGS[@]}"
echo "DONE: $OUT/synthetic ($(ls "$OUT/synthetic" | wc -l) frames)"

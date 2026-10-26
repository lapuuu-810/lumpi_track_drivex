#!/usr/bin/env bash
# Shared, side-effect-free shell helpers for reproduction entry points.
set -euo pipefail

project_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

python_command() {
  local environment="${CONDA_ENV:-urbantwin-v2xreal}"
  if command -v conda >/dev/null 2>&1; then
    printf '%s\0' conda run -n "$environment" --no-capture-output python
  else
    printf '%s\0' "/opt/conda/envs/$environment/bin/python"
  fi
}

load_python_command() {
  mapfile -d '' -t PYTHON < <(python_command)
}

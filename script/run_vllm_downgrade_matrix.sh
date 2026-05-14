#!/bin/bash
# ============================================================================
# Run one entry of the vLLM downgrade matrix
# ============================================================================
# This script intentionally does NOT create all environments automatically.
# Default behavior:
#   - operate on one VLLM_VERSION only
#   - require the target conda env to already be active
#
# Example:
#   VLLM_VERSION=0.8.5 bash script/create_vllm_torch26_env.sh
#   conda activate echo_vllm_torch26
#   VLLM_VERSION=0.8.5 bash script/run_vllm_downgrade_matrix.sh
#
# Manual loop example:
#   for V in 0.8.5 0.8.4 0.8.3 0.7.3 0.7.2 0.6.6 0.6.4.post1 0.6.4; do
#     ENV_NAME="echo_vllm_torch26_${V//./_}"
#     ENV_NAME="${ENV_NAME//-/_}"
#     ENV_NAME="$ENV_NAME" VLLM_VERSION="$V" bash script/create_vllm_torch26_env.sh
#     conda activate "$ENV_NAME"
#     VLLM_VERSION="$V" bash script/run_vllm_downgrade_matrix.sh
#     conda deactivate
#   done
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
EXPECTED_ENV="${EXPECTED_ENV:-echo_vllm_torch26}"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Selected VLLM_VERSION: $VLLM_VERSION"
echo "Active conda env: ${CONDA_DEFAULT_ENV:-'(not set)'}"

if [ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]; then
    echo "WARN: active env is not the default expected env: $EXPECTED_ENV"
    echo "If this is intentional for per-version env names, ignore this warning."
fi

echo "Running single-version downgrade smoke..."
VLLM_VERSION="$VLLM_VERSION" bash script/test6_vllm_downgrade_smoke.sh

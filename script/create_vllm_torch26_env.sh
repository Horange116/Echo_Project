#!/bin/bash
# ============================================================================
# Create isolated torch 2.6.0 + vLLM downgrade test env
# ============================================================================
# Goal:
#   Create a fresh conda env for testing one downgraded vLLM version against
#   torch 2.6.0 without touching the known-good VERL training env.
#
# Notes:
#   - Does NOT overwrite an existing env.
#   - Installs vLLM with --no-deps first to avoid surprise torch upgrades.
#   - If pip check reports missing deps, only installs a curated set manually.
#   - This script is for environment creation only; it does not run the smoke.
#
# Usage:
#   VLLM_VERSION=0.8.5 bash script/create_vllm_torch26_env.sh
#   ENV_NAME=echo_vllm_torch26_085 VLLM_VERSION=0.8.4 bash script/create_vllm_torch26_env.sh
# ============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-echo_vllm_torch26}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "ERROR: conda env already exists: $ENV_NAME"
    echo "Refusing to overwrite an existing env."
    exit 1
fi

echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
conda activate "$ENV_NAME"

echo "Installing torch ${TORCH_VERSION} (CUDA 12.4 wheel)..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    --index-url "${TORCH_CUDA_INDEX_URL}"

echo "Installing base dependencies..."
python -m pip install \
    transformers \
    accelerate \
    peft \
    librosa \
    soundfile \
    decord \
    fastapi \
    uvicorn

echo "Installing Qwen multimodal helper dependencies..."
if ! python -m pip install qwen-omni-utils; then
    echo "WARN: qwen-omni-utils install failed; trying qwen-vl-utils."
    python -m pip install qwen-vl-utils || true
fi

echo "Installing vLLM ${VLLM_VERSION} with --no-deps to avoid torch upgrades..."
python -m pip install "vllm==${VLLM_VERSION}" --no-deps

echo "Checking current torch/vLLM versions after install..."
python -c "import torch; print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"

echo "Running pip check (expected to reveal any unresolved dependencies)..."
set +e
PIP_CHECK_OUTPUT="$(python -m pip check 2>&1)"
PIP_CHECK_RC=$?
set -e
printf '%s\n' "$PIP_CHECK_OUTPUT"

if [ $PIP_CHECK_RC -ne 0 ]; then
    echo "pip check reported unresolved dependencies."
    echo "Installing a curated fallback dependency set without allowing torch upgrades..."

    # Curated package set for older vLLM releases. Keep this explicit.
    python -m pip install --no-deps \
        numpy \
        scipy \
        sentencepiece \
        tokenizers \
        psutil \
        protobuf \
        pyzmq \
        msgspec \
        cloudpickle \
        packaging \
        pydantic \
        prometheus-client \
        tiktoken \
        xformers || true

    echo "Re-running pip check after curated dependency install..."
    python -m pip check || true
fi

echo "Final environment summary:"
python -c "import sys; print(sys.version)"
python -c "import torch; print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
python -m pip check || true

echo ""
echo "Environment created: $ENV_NAME"
echo "Next step:"
echo "  conda activate $ENV_NAME"
echo "  VLLM_VERSION=${VLLM_VERSION} bash script/test6_vllm_downgrade_smoke.sh"

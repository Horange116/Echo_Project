#!/bin/bash
# Test28: Confidence-weighted reward alignment.
# CPU-only test — no GPU needed.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"

if command -v singularity >/dev/null 2>&1; then
    CONTAINER_CMD="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: neither singularity nor apptainer"
    exit 1
fi

for p in "$PROJECT_ROOT" "$SIF_PATH" "$CONTAINER_ROOT" "$MODEL_PATH"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing: $p"
        exit 1
    fi
done

"$CONTAINER_CMD" exec \
    --bind /hpai:/hpai \
    --bind /home:/home \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$MODEL_PATH:$MODEL_PATH" \
    --bind "$CONTAINER_ROOT:$CONTAINER_ROOT" \
    "$SIF_PATH" \
    bash -lc "
        export PATH='$CONTAINER_ROOT/bin':\"\$PATH\"
        export PYTHONNOUSERSITE=1
        export HF_HOME='${PROJECT_ROOT}/output/singularity/hf_cache'
        export TRANSFORMERS_CACHE='${PROJECT_ROOT}/output/singularity/hf_cache'
        cd '$PROJECT_ROOT'
        python script/test28_reward_confidence_align.py --model-path '$MODEL_PATH'
    "

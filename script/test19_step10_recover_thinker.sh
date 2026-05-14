#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_ROOT}/output/grpo_smoke/checkpoints/step_10}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
CONTAINER_CMD="${CONTAINER_CMD:-}"
GPU_ID="${GPU_ID:-0}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/output/rl_rollout/test19_step10_recover_thinker}"

STDOUT_LOG="${OUT_DIR}/test19_stdout.log"
STDERR_LOG="${OUT_DIR}/test19_stderr.log"

if [ -z "$CONTAINER_CMD" ]; then
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_CMD="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_CMD="apptainer"
    else
        echo "ERROR: neither singularity nor apptainer is installed."
        exit 1
    fi
fi

for p in "$PROJECT_ROOT" "$MODEL_PATH" "$SIF_PATH" "$ADAPTER_PATH"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: required path missing: $p"
        exit 1
    fi
done

if [ ! -x "${CONTAINER_ROOT}/bin/python" ]; then
    echo "ERROR: CONTAINER_ROOT does not look like a prepared Python root: $CONTAINER_ROOT"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "===== test19 step_10 thinker recovery diagnosis ====="
echo "MODEL_PATH: $MODEL_PATH"
echo "ADAPTER_PATH: $ADAPTER_PATH"
echo "GPU_ID: $GPU_ID"
echo "OUT_DIR: $OUT_DIR"
echo "===================================================="

set +e
"$CONTAINER_CMD" exec --nv \
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
        python scripts/rl/thinker_forward_diag.py \
            --model_path '$MODEL_PATH' \
            --adapter_path '$ADAPTER_PATH' \
            --gpu_id '$GPU_ID'
    " >"$STDOUT_LOG" 2>"$STDERR_LOG"
RUN_EXIT_CODE=$?
set -e

echo "Exit code: $RUN_EXIT_CODE"
echo "stdout: $STDOUT_LOG"
echo "stderr: $STDERR_LOG"

if [ "$RUN_EXIT_CODE" -ne 0 ]; then
    echo "test19 failed with exit code $RUN_EXIT_CODE"
    exit "$RUN_EXIT_CODE"
fi

echo "===== test19 done ====="

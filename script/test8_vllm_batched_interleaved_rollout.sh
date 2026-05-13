#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3}"
CONTAINER_CMD="${CONTAINER_CMD:-}"
MANIFEST_PATH="${MANIFEST_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eval_manifest_500.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
NUM_ROLLOUTS_PER_SAMPLE="${NUM_ROLLOUTS_PER_SAMPLE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/output/rl_rollout}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/test8_batched_interleaved_controller.json}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/output/interleaved_tmp/batched_controller}"

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

for p in "$PROJECT_ROOT" "$MODEL_PATH" "$SIF_PATH" "$MANIFEST_PATH"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: required path missing: $p"
        exit 1
    fi
done

if [ ! -x "${CONTAINER_ROOT}/bin/python" ]; then
    echo "ERROR: CONTAINER_ROOT does not look like a prepared Python root: $CONTAINER_ROOT"
    exit 1
fi

mkdir -p "$OUT_DIR" "$WORK_DIR"
cd "$PROJECT_ROOT"

echo "===== test8 batched interleaved rollout ====="
echo "CONTAINER_CMD: $CONTAINER_CMD"
echo "SIF_PATH: $SIF_PATH"
echo "MODEL_PATH: $MODEL_PATH"
echo "MANIFEST_PATH: $MANIFEST_PATH"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "NUM_ROLLOUTS_PER_SAMPLE: $NUM_ROLLOUTS_PER_SAMPLE"
echo "OUT_JSON: $OUT_JSON"
echo "============================================="

exec "$CONTAINER_CMD" exec --nv \
    --bind /hpai:/hpai \
    --bind /home:/home \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$MODEL_PATH:$MODEL_PATH" \
    --bind "$MANIFEST_PATH:$MANIFEST_PATH" \
    --bind "$CONTAINER_ROOT:$CONTAINER_ROOT" \
    "$SIF_PATH" \
    bash -lc "
        export PATH='$CONTAINER_ROOT/bin':\"\$PATH\"
        export PYTHONNOUSERSITE=1
        export HF_HOME='${PROJECT_ROOT}/output/singularity/hf_cache'
        export TRANSFORMERS_CACHE='${PROJECT_ROOT}/output/singularity/hf_cache'
        cd '$PROJECT_ROOT'
        python scripts/rl_rollout/echo_interleaved_rollout_controller.py \
            --model_path '$MODEL_PATH' \
            --manifest_path '$MANIFEST_PATH' \
            --num_samples '$NUM_SAMPLES' \
            --num_rollouts_per_sample '$NUM_ROLLOUTS_PER_SAMPLE' \
            --temperature '$TEMPERATURE' \
            --max_rounds '$MAX_ROUNDS' \
            --max_tokens '$MAX_TOKENS' \
            --gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' \
            --max_model_len '$MAX_MODEL_LEN' \
            --work_dir '$WORK_DIR' \
            --output_json '$OUT_JSON'
    "

#!/bin/bash
# Test29: Rollout n=8 stability smoke.
# Verify custom strict/rollout/GRPO chain at paper-default rollout.n=8.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
CONTAINER_CMD="${CONTAINER_CMD:-}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_ROOT}/output/grpo_smoke/checkpoints/step_10}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/output/GeneratedData/eval_manifest_500.jsonl}"

# n=8 is the key variable
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_STEPS="${MAX_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
MAX_TOKENS="${MAX_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-1.0}"
FINALIZE_MAX_TOKENS="${FINALIZE_MAX_TOKENS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_ID="${GPU_ID:-0}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-900}"

ROLLOUT_OUT_DIR="${ROLLOUT_OUT_DIR:-${PROJECT_ROOT}/output/rl_rollout/test29_n8_smoke}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${PROJECT_ROOT}/output/grpo_vllm_batched_n8_smoke29}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/output/interleaved_tmp/test29_n8_worker}"

STDOUT_LOG="${ROLLOUT_OUT_DIR}/test29_stdout.log"
STDERR_LOG="${ROLLOUT_OUT_DIR}/test29_stderr.log"

if [ -z "$CONTAINER_CMD" ]; then
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_CMD="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_CMD="apptainer"
    else
        echo "ERROR: neither singularity nor apptainer"
        exit 1
    fi
fi

for p in "$PROJECT_ROOT" "$MODEL_PATH" "$SIF_PATH" "$DATA_PATH" "$ADAPTER_PATH"; do
    if [ ! -e "$p" ]; then echo "ERROR: missing: $p"; exit 1; fi
done
if [ ! -x "${CONTAINER_ROOT}/bin/python" ]; then
    echo "ERROR: CONTAINER_ROOT not valid: $CONTAINER_ROOT"; exit 1
fi

mkdir -p "$ROLLOUT_OUT_DIR" "$TRAIN_OUT_DIR" "$WORK_DIR"
mkdir -p "${TRAIN_OUT_DIR}/logs" "${TRAIN_OUT_DIR}/checkpoints"

START_TS="$(date +%s)"

echo "===== test29 rollout n=8 stability smoke ====="
echo "KEY VARIABLE: NUM_ROLLOUTS=$NUM_ROLLOUTS"
echo "MAX_SAMPLES=$MAX_SAMPLES MAX_STEPS=$MAX_STEPS BATCH_SIZE=$BATCH_SIZE"
echo "ADAPTER_PATH=$ADAPTER_PATH (step_10, clean)"
echo "OPTIMIZER: AdamW(eps=1e-4)"
echo "ROLLOUT_MODE: per_task (single GPU)"
echo "FORWARD_MODE: strict_interleaved"
echo "=============================================="

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
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        cd '$PROJECT_ROOT'
        python scripts/rl/rollout_smoke_test.py \
            --model_path '$MODEL_PATH' \
            --adapter_path '$ADAPTER_PATH' \
            --data_path '$DATA_PATH' \
            --output_dir '$TRAIN_OUT_DIR' \
            --max_samples '$MAX_SAMPLES' \
            --num_rollouts '$NUM_ROLLOUTS' \
            --batch_size '$BATCH_SIZE' \
            --max_steps '$MAX_STEPS' \
            --max_rounds '$MAX_ROUNDS' \
            --max_new_tokens '$MAX_TOKENS' \
            --temperature '$TEMPERATURE' \
            --finalize_max_new_tokens '$FINALIZE_MAX_TOKENS' \
            --worker_timeout '$WORKER_TIMEOUT' \
            --gpu_id '$GPU_ID' \
            --rollout_backend vllm_batched \
            --rollout_worker_mode per_task \
            --grpo_forward_mode strict_interleaved \
            --policy_forward_micro_batch_size 1 \
            --worker_gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' \
            --worker_max_model_len '$MAX_MODEL_LEN' \
            --worker_work_dir '$WORK_DIR'
    " >"$STDOUT_LOG" 2>"$STDERR_LOG"
RUN_EXIT_CODE=$?
set -e

END_TS="$(date +%s)"
ELAPSED_SECONDS="$((END_TS - START_TS))"

echo ""
echo "Exit code: $RUN_EXIT_CODE"
echo "Elapsed: ${ELAPSED_SECONDS}s"

echo ""
echo "===== step lines ====="
grep "step " "$STDOUT_LOG" || echo "(no step lines)"

echo ""
echo "===== n=8 key metrics ====="
grep -E "num_rollouts|rollout_success_count|rollout_failed_count|strict_forward_success|strict_forward_failed|worker_restart_count|peak_memory|weights_healthy|grad_healthy|loss|correct|R " "$STDOUT_LOG" || echo "(no matches)"

echo ""
echo "===== nan/inf protection ====="
grep -n "\[nan-inf\]\|skipping backward\|SKIPPING\|WARNING\|weights_healthy" "$STDOUT_LOG" | tail -10 || echo "(no diag lines)"

echo ""
echo "===== checkpoint saves ====="
grep -i "checkpoint\|save_pretrained" "$STDOUT_LOG" | grep -v "config\|args" || echo "(no checkpoints saved)"

echo ""
echo "===== checkpoint dir ====="
ls -la "${TRAIN_OUT_DIR}/checkpoints/" 2>/dev/null || echo "(empty or missing)"

echo ""
echo "===== rollout logs ====="
ls -la "${TRAIN_OUT_DIR}/logs/" 2>/dev/null || echo "(empty or missing)"

echo ""
echo "stdout: $STDOUT_LOG"
echo "stderr: $STDERR_LOG"
echo "===== test29 done ====="

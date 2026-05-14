#!/bin/bash
# Test32: A3 stability regression — multi-GPU resource split, eps=1e-4.
# Training (actor/ref) on GPU 0, rollout workers on GPU 1.
# Verifies no nan/inf under the A3 device-separated regime.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_ROOT}/output/grpo_smoke/checkpoints/step_10}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/output/GeneratedData/eval_manifest_500.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/test32_a3_stability}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"

# Container runtime detection
CONTAINER_CMD="${CONTAINER_CMD:-}"
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

# Device role config (A3 resource split)
TRAIN_DEVICE="${TRAIN_DEVICE:-0}"
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-1}"
ROLLOUT_WORKER_MODE="${ROLLOUT_WORKER_MODE:-pool}"
NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-1}"

# Pipeline config (small, for fast stability verification)
MAX_SAMPLES="${MAX_SAMPLES:-4}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
MAX_STEPS="${MAX_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-96}"
FINALIZE_MAX_TOKENS="${FINALIZE_MAX_TOKENS:-64}"
GRPO_FORWARD_MODE="${GRPO_FORWARD_MODE:-text_only}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/checkpoints"
STDOUT_LOG="${OUTPUT_DIR}/test32_stdout.log"
STDERR_LOG="${OUTPUT_DIR}/test32_stderr.log"

for p in "$PROJECT_ROOT" "$MODEL_PATH" "$ADAPTER_PATH" "$DATA_PATH" "$SIF_PATH"; do
    if [ ! -e "$p" ]; then echo "ERROR: missing $p"; exit 1; fi
done
if [ ! -x "${CONTAINER_ROOT}/bin/python" ]; then
    echo "ERROR: CONTAINER_ROOT does not look like a prepared Python root: $CONTAINER_ROOT"
    exit 1
fi

START_TS="$(date +%s)"

echo "===== Test32: A3 Stability Regression (eps=1e-4) ====="
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  MODEL_PATH   = $MODEL_PATH"
echo "  ADAPTER_PATH = $ADAPTER_PATH"
echo "  DATA_PATH    = $DATA_PATH"
echo "  OUTPUT_DIR   = $OUTPUT_DIR"
echo ""
echo "  --- Device Roles (A3 resource split) ---"
echo "  TRAIN_DEVICE           = $TRAIN_DEVICE"
echo "  ROLLOUT_DEVICES        = $ROLLOUT_DEVICES"
echo "  ROLLOUT_WORKER_MODE    = $ROLLOUT_WORKER_MODE"
echo "  NUM_ROLLOUT_WORKERS    = $NUM_ROLLOUT_WORKERS"
echo ""
echo "  --- Pipeline Config ---"
echo "  MAX_SAMPLES  = $MAX_SAMPLES"
echo "  NUM_ROLLOUTS = $NUM_ROLLOUTS"
echo "  MAX_STEPS    = $MAX_STEPS"
echo "  BATCH_SIZE   = $BATCH_SIZE"
echo "  MAX_ROUNDS   = $MAX_ROUNDS"
echo "  TEMPERATURE  = $TEMPERATURE"
echo "  FORWARD_MODE = $GRPO_FORWARD_MODE"
echo "  OPTIMIZER    = AdamW(eps=1e-4)"
echo ""

# ── Snapshot 1: idle GPUs before start ──
echo "--- [snapshot 1] GPU state BEFORE pipeline ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# ── Phase 1: Run the pipeline ──
echo "--- Starting pipeline ---"
echo "  stdout → $STDOUT_LOG"
echo "  stderr → $STDERR_LOG"

export HF_HOME="${PROJECT_ROOT}/output/singularity/hf_cache"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/output/singularity/hf_cache"
export PYTHONNOUSERSITE=1
cd "$PROJECT_ROOT"

# shellcheck disable=SC2086
${CONTAINER_ROOT}/bin/python -u ${PROJECT_ROOT}/scripts/rl/rollout_smoke_test.py \
    --model_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_device "$TRAIN_DEVICE" \
    --rollout_worker_devices "$ROLLOUT_DEVICES" \
    --rollout_worker_mode "$ROLLOUT_WORKER_MODE" \
    --num_rollout_workers "$NUM_ROLLOUT_WORKERS" \
    --worker_use_singularity \
    --worker_sif_path "$SIF_PATH" \
    --worker_container_root "$CONTAINER_ROOT" \
    --max_samples "$MAX_SAMPLES" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --max_rounds "$MAX_ROUNDS" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_TOKENS" \
    --finalize_max_new_tokens "$FINALIZE_MAX_TOKENS" \
    --grpo_forward_mode "$GRPO_FORWARD_MODE" \
    --num_epochs 1 \
    --seed 42 \
    > "$STDOUT_LOG" 2> "$STDERR_LOG"

PIPE_EXIT=$?
echo "  Pipeline exit code: $PIPE_EXIT"

# ── Snapshot 2: GPU state after pipeline ──
echo ""
echo "--- [snapshot 2] GPU state AFTER pipeline ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

echo ""
END_TS="$(date +%s)"
ELAPSED=$(( END_TS - START_TS ))
echo "Duration: ${ELAPSED}s"

echo ""
echo "===== Test32: Summary ====="
if grep -q "Device Roles:" "$STDOUT_LOG" 2>/dev/null; then
    echo "  ✅ Device role header printed"
    grep "Training\|Rollout worker\|Worker mode" "$STDOUT_LOG"
else
    echo "  ⚠️  No device role header found"
fi

echo ""
# Check for nan/inf
NAN_INF=$(grep -c "nan-inf\|nan_inf\|SKIPPING" "$STDOUT_LOG" 2>/dev/null || echo 0)
if [ "$NAN_INF" -gt 0 ]; then
    echo "  ⚠️  nan/inf detected ($NAN_INF occurrences)"
    grep "nan-inf\|nan_inf\|SKIPPING\|weights_healthy" "$STDOUT_LOG"
else
    echo "  ✅ No nan/inf detected"
fi

echo ""
if grep -q " \[PASS\]\|step.*loss.*R\|Reached --max_steps\|total_wall_time" "$STDOUT_LOG" 2>/dev/null; then
    echo "  ✅ Pipeline completed"
    grep -E "step.*loss.*R\|rollout_success\|rollout_failed\|worker_restart\|total_wall_time\|Reached --max_steps" "$STDOUT_LOG"
elif [ $PIPE_EXIT -eq 0 ]; then
    echo "  ✅ Pipeline exit code 0"
else
    echo "  ❌ Pipeline failed (exit $PIPE_EXIT)"
    echo "  === Last 30 lines of stderr ==="
    tail -30 "$STDERR_LOG"
fi

exit $PIPE_EXIT

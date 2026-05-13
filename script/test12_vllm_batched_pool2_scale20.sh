#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
CONTAINER_CMD="${CONTAINER_CMD:-}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_ROOT}/output/grpo_smoke/checkpoints/final}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/output/GeneratedData/eval_manifest_500.jsonl}"

MAX_SAMPLES="${MAX_SAMPLES:-20}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
MAX_STEPS="${MAX_STEPS:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
MAX_TOKENS="${MAX_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-1.0}"
FINALIZE_MAX_TOKENS="${FINALIZE_MAX_TOKENS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-2}"
WORKER_DEVICES="${WORKER_DEVICES:-0,1}"
GPU_ID="${GPU_ID:-0}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-900}"

ROLLOUT_OUT_DIR="${ROLLOUT_OUT_DIR:-${PROJECT_ROOT}/output/rl_rollout/test12_pool2_scale20}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${PROJECT_ROOT}/output/grpo_vllm_batched_pool2_scale20}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/output/interleaved_tmp/test12_pool2_worker}"

STDOUT_LOG="${ROLLOUT_OUT_DIR}/test12_stdout.log"
STDERR_LOG="${ROLLOUT_OUT_DIR}/test12_stderr.log"
SUMMARY_JSON="${ROLLOUT_OUT_DIR}/test12_summary.json"

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

for p in "$PROJECT_ROOT" "$MODEL_PATH" "$SIF_PATH" "$DATA_PATH" "$ADAPTER_PATH"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: required path missing: $p"
        exit 1
    fi
done

if [ ! -x "${CONTAINER_ROOT}/bin/python" ]; then
    echo "ERROR: CONTAINER_ROOT does not look like a prepared Python root: $CONTAINER_ROOT"
    exit 1
fi

mkdir -p "$ROLLOUT_OUT_DIR" "$TRAIN_OUT_DIR" "$WORK_DIR"
mkdir -p "${TRAIN_OUT_DIR}/logs" "${TRAIN_OUT_DIR}/checkpoints"

START_TS="$(date +%s)"

echo "===== test12 vllm_batched pool2 custom GRPO scale20 ====="
echo "CONTAINER_CMD: $CONTAINER_CMD"
echo "MODEL_PATH: $MODEL_PATH"
echo "ADAPTER_PATH: $ADAPTER_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "MAX_SAMPLES: $MAX_SAMPLES"
echo "NUM_ROLLOUTS: $NUM_ROLLOUTS"
echo "MAX_STEPS: $MAX_STEPS"
echo "MAX_ROUNDS: $MAX_ROUNDS"
echo "MAX_TOKENS: $MAX_TOKENS"
echo "TEMPERATURE: $TEMPERATURE"
echo "FINALIZE_MAX_TOKENS: $FINALIZE_MAX_TOKENS"
echo "GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_ROLLOUT_WORKERS: $NUM_ROLLOUT_WORKERS"
echo "WORKER_DEVICES: $WORKER_DEVICES"
echo "ROLLOUT_OUT_DIR: $ROLLOUT_OUT_DIR"
echo "TRAIN_OUT_DIR: $TRAIN_OUT_DIR"
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
            --rollout_worker_mode pool \
            --num_rollout_workers '$NUM_ROLLOUT_WORKERS' \
            --worker_devices '$WORKER_DEVICES' \
            --grpo_forward_mode text_only \
            --policy_forward_micro_batch_size 1 \
            --worker_gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' \
            --worker_max_model_len '$MAX_MODEL_LEN' \
            --worker_work_dir '$WORK_DIR'
    " >"$STDOUT_LOG" 2>"$STDERR_LOG"
RUN_EXIT_CODE=$?
set -e

END_TS="$(date +%s)"
ELAPSED_SECONDS="$((END_TS - START_TS))"

export TRAIN_OUT_DIR STDOUT_LOG STDERR_LOG SUMMARY_JSON RUN_EXIT_CODE ELAPSED_SECONDS NUM_ROLLOUTS
python3 - <<'PY'
import json
import os
import re
from pathlib import Path

train_out = Path(os.environ["TRAIN_OUT_DIR"])
stdout_log = Path(os.environ["STDOUT_LOG"])
stderr_log = Path(os.environ["STDERR_LOG"])
summary_json = Path(os.environ["SUMMARY_JSON"])
run_exit_code = int(os.environ["RUN_EXIT_CODE"])
elapsed = int(os.environ["ELAPSED_SECONDS"])

summary = {
    "run_exit_code": run_exit_code,
    "elapsed_seconds": elapsed,
    "rollout_success_count": None,
    "rollout_failed_count": None,
    "reward_mean": None,
    "reward_min": None,
    "reward_max": None,
    "step_count": None,
    "pool_batch_times_s": [],
    "worker_pool_line": None,
    "trainer_args_json": str(train_out / "args.json") if (train_out / "args.json").exists() else None,
    "rollouts_jsonl": str(train_out / "logs" / "rollouts.jsonl") if (train_out / "logs" / "rollouts.jsonl").exists() else None,
    "stdout_log": str(stdout_log),
    "stderr_log": str(stderr_log),
}

log_text = ""
if stdout_log.exists():
    log_text += stdout_log.read_text(errors="ignore")
if stderr_log.exists():
    log_text += "\n" + stderr_log.read_text(errors="ignore")

succ = re.findall(r"rollout_success_count:\s+(\d+)", log_text)
fail = re.findall(r"rollout_failed_count:\s+(\d+)", log_text)
if succ:
    summary["rollout_success_count"] = int(succ[-1])
if fail:
    summary["rollout_failed_count"] = int(fail[-1])

step_matches = re.findall(r"\bstep\s+(\d+)\s+\|", log_text)
if step_matches:
    summary["step_count"] = max(int(x) for x in step_matches) + 1

pool_line = re.findall(r"\[pool\]\s+\d+\s+workers on devices\s+\[[^\]]+\]", log_text)
if pool_line:
    summary["worker_pool_line"] = pool_line[-1]

pool_times = re.findall(r"\[pool\]\s+\d+\s+tasks in\s+([0-9.]+)s", log_text)
if pool_times:
    summary["pool_batch_times_s"] = [float(x) for x in pool_times]

rollouts_jsonl = train_out / "logs" / "rollouts.jsonl"
if rollouts_jsonl.exists():
    rewards = []
    with rollouts_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "rollout_total" in row:
                rewards.append(float(row["rollout_total"]))
    if rewards:
        summary["reward_mean"] = sum(rewards) / len(rewards)
        summary["reward_min"] = min(rewards)
        summary["reward_max"] = max(rewards)

summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print("===== test12 summary =====")
print(f"run_exit_code: {summary['run_exit_code']}")
print(f"elapsed_seconds: {summary['elapsed_seconds']}")
print(f"rollout_success_count: {summary['rollout_success_count']}")
print(f"rollout_failed_count: {summary['rollout_failed_count']}")
print(f"reward_mean: {summary['reward_mean']}")
print(f"reward_min: {summary['reward_min']}")
print(f"reward_max: {summary['reward_max']}")
print(f"step_count: {summary['step_count']}")
print(f"worker_pool_line: {summary['worker_pool_line']}")
print(f"pool_batch_times_s: {summary['pool_batch_times_s']}")
print(f"summary_json: {summary_json}")
print(f"stdout_log: {stdout_log}")
print(f"stderr_log: {stderr_log}")
print("==========================")
PY

if [ "$RUN_EXIT_CODE" -ne 0 ]; then
    echo "test12 failed with exit code $RUN_EXIT_CODE"
    exit "$RUN_EXIT_CODE"
fi

echo "===== test12 done ====="

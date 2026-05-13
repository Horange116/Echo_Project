#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
CONTAINER_CMD="${CONTAINER_CMD:-}"
MANIFEST_PATH="${MANIFEST_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eval_manifest_500.jsonl}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
MAX_TOKENS="${MAX_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.9}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/output/rl_rollout}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/test9_worker_vllm_batched.json}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/output/interleaved_tmp/worker_smoke}"

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

# Extract sample and base64-encode to avoid shell escaping issues
SAMPLE_B64=$(python3 -c "
import json, sys, base64
with open('$MANIFEST_PATH') as f:
    for i, line in enumerate(f):
        if i == $SAMPLE_INDEX:
            s = json.loads(line.strip())
            if 'multi_choice' in s and 'choices' not in s:
                s['choices'] = s['multi_choice']
            print(base64.b64encode(json.dumps(s, ensure_ascii=False).encode()).decode())
            break
")

echo "===== test9 worker vllm_batched smoke ====="
echo "CONTAINER_CMD: $CONTAINER_CMD"
echo "OUT_JSON: $OUT_JSON"
echo "SAMPLE_B64 len: ${#SAMPLE_B64}"
echo "============================================"

exec "$CONTAINER_CMD" exec --nv \
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
        SAMPLE_JSON=\$(echo '$SAMPLE_B64' | base64 -d)
        python scripts/rl/isolated_rollout_worker.py \
            --sample_json \"\$SAMPLE_JSON\" \
            --model_path '$MODEL_PATH' \
            --rollout_backend vllm_batched \
            --max_rounds '$MAX_ROUNDS' \
            --max_new_tokens '$MAX_TOKENS' \
            --num_generations '$NUM_GENERATIONS' \
            --temperature '$TEMPERATURE' \
            --gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' \
            --max_model_len '$MAX_MODEL_LEN' \
            --work_dir '$WORK_DIR' \
            --timeout 600
    " > "$OUT_JSON" 2> "${OUT_JSON%.json}.err"

echo "===== test9 done ====="

#!/bin/bash
# Test30: VERL data format alignment.
# Convert custom JSONL → VERL Parquet → dry-load with RLHFDataset.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/output/GeneratedData/eval_manifest_500.jsonl}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${PROJECT_ROOT}/output/verl_data_align_test30/test30_eval_manifest_4.parquet}"

if command -v singularity >/dev/null 2>&1; then
    CONTAINER_CMD="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CMD="apptainer"
else
    echo "ERROR: neither singularity nor apptainer"
    exit 1
fi

for p in "$PROJECT_ROOT" "$SIF_PATH" "$CONTAINER_ROOT" "$MODEL_PATH" "$INPUT_JSONL"; do
    if [ ! -e "$p" ]; then echo "ERROR: missing: $p"; exit 1; fi
done

echo "=== Test30: VERL Data Format Alignment ==="
echo "MODEL_PATH=$MODEL_PATH"
echo

# Step 1: Convert JSONL → Parquet (host python has pyarrow)
echo "--- Step 1: JSONL → Parquet ---"
mkdir -p "$(dirname "$OUTPUT_PARQUET")"
python3 -c "
import json, pyarrow as pa, pyarrow.parquet as pq

prompt_col, audios_col, answer_col = [], [], []
with open('$INPUT_JSONL') as f:
    for i, line in enumerate(f):
        if i >= 4: break
        s = json.loads(line.strip())
        choices_str = ', '.join(s['choices'])
        prompt = s['question'] + ' Choose the answer from ' + choices_str + '. Think step-by-step. Refer to the specific audio segments while thinking, and indicate the corresponding timestamps with <seg>start, end</seg>. Answer in the format of <think>...</think><answer>...</answer>.'
        messages = json.dumps([{'role': 'user', 'content': '<audio>' + prompt}])
        prompt_col.append(messages)
        audios_col.append(json.dumps([s['audio_path']]))
        answer_col.append(s['answer'])

table = pa.table({'prompt': pa.array(prompt_col, type=pa.string()), 'audios': pa.array(audios_col, type=pa.string()), 'answer': pa.array(answer_col, type=pa.string())})
pq.write_table(table, '$OUTPUT_PARQUET')
print(f'  Wrote {len(table)} samples to $OUTPUT_PARQUET')
" 2>&1

# Step 2: Dry-load with VERL's RLHFDataset (inside container)
echo ""
echo "--- Step 2: Dry-load with VERL RLHFDataset ---"
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
        export MODEL_PATH='$MODEL_PATH'
        export PARQUET_PATH='$OUTPUT_PARQUET'
        export PROJECT_ROOT='$PROJECT_ROOT'
        cd '$PROJECT_ROOT'
        python script/test30_verl_data_step2.py
    "

echo ""
echo "=== Test30: Done ==="

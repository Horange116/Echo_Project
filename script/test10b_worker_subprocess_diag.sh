#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/output/singularity/nvidia_cuda_12.4.sif}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${PROJECT_ROOT}/output/singularity/miniconda3/envs/vllm085}"
CONTAINER_CMD="${CONTAINER_CMD:-singularity}"

# Write sample to temp file for safe passing
SAMPLE_TMP="${PROJECT_ROOT}/output/interleaved_tmp/_diag_sample.json"
python3 -c "
import json
with open('${PROJECT_ROOT}/output/GeneratedData/eval_manifest_500.jsonl') as f:
    for i, line in enumerate(f):
        if i == 0:
            s = json.loads(line.strip())
            if 'multi_choice' in s and 'choices' not in s:
                s['choices'] = s['multi_choice']
            with open('${SAMPLE_TMP}', 'w') as out:
                json.dump(s, out)
            break
"

echo "===== test10b diag: worker as subprocess ====="

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
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        cd '$PROJECT_ROOT'
        python -c \"
import subprocess, sys, json, os
with open('${SAMPLE_TMP}') as f:
    sample = json.load(f)
sample_json = json.dumps(sample, ensure_ascii=False)
cmd = [
    sys.executable, '-u',
    'scripts/rl/isolated_rollout_worker.py',
    '--sample_json', sample_json,
    '--model_path', '$MODEL_PATH',
    '--rollout_backend', 'vllm_batched',
    '--max_rounds', '2',
    '--max_new_tokens', '96',
    '--num_generations', '1',
    '--temperature', '0.9',
    '--gpu_memory_utilization', '0.85',
    '--max_model_len', '32768',
    '--work_dir', '${PROJECT_ROOT}/output/interleaved_tmp/worker_smoke_diag',
    '--timeout', '600',
]
print('DIAG: spawning worker...', file=sys.stderr)
sys.stderr.flush()
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=700)
print(f'DIAG: rc={proc.returncode}', file=sys.stderr)
print(f'DIAG: stdout_len={len(proc.stdout)}', file=sys.stderr)
print(f'DIAG: stderr_len={len(proc.stderr)}', file=sys.stderr)
if proc.stdout:
    print('STDOUT:', proc.stdout[:1000])
else:
    print('STDOUT: (empty)')
print('STDERR_TAIL:')
print(proc.stderr[-3000:])
\"
    " 2>&1

echo "===== test10b done ====="

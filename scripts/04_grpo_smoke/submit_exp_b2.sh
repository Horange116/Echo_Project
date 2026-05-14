#!/bin/bash
#SBATCH -J exp_b2_fwd
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/04_grpo_smoke/slurm-exp-b2-%j.out

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

echo "[$(date)] Exp B2: forward-only from saved rollouts"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Host:    $(hostname)"
echo "  GPU:     $CUDA_VISIBLE_DEVICES"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Wait for rollout file to exist
ROLLOUT_FILE="output/grpo_smoke/exp_b/rollout_outputs.jsonl"
MAX_WAIT=300
WAITED=0
while [ ! -f "$ROLLOUT_FILE" ]; do
  if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: timeout waiting for $ROLLOUT_FILE"
    exit 1
  fi
  echo "[$(date)] Waiting for $ROLLOUT_FILE ... (${WAITED}s)"
  sleep 10
  WAITED=$((WAITED + 10))
done

echo "[$(date)] Rollout file found: $(wc -l < $ROLLOUT_FILE) lines"

python -u scripts/04_grpo_smoke/exp_b2_forward.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --rollout_file "$ROLLOUT_FILE" \
  --output_dir "output/grpo_smoke/exp_b"

echo "[$(date)] Exp B2 complete"

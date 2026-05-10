#!/bin/bash
#SBATCH -J v9b_batch_eval
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/04_grpo_smoke/slurm-batch-eval-%j.out

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

echo "[$(date)] Starting batch interleaved eval (v9b-2epoch, 20 samples)"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Host:    $(hostname)"
echo "  GPU:     $CUDA_VISIBLE_DEVICES"

python -u scripts/04_grpo_smoke/batch_interleaved_eval.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --data_path "output/judge/split_rl.jsonl" \
  --output_dir "output/interleaved_eval/v9b_2epoch_batch20" \
  --num_samples 20 \
  --max_rounds 5 \
  --max_new_tokens 128 \
  --temperature 0.7

echo "[$(date)] Batch eval complete"

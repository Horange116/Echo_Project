#!/bin/bash
# Test 2: Persistent worker + text_only
#SBATCH -J rl_test2_persist
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/rl/slurm-test2-%j.out

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

echo "[$(date)] Test 2: persistent + text_only"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Host:    $(hostname)"
echo "  GPU:     $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python -u scripts/rl/rollout_smoke_test.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --data_path "dataJson/NAQA/EAQA_RL.jsonl" \
  --output_dir "output/grpo_test2_persistent" \
  --max_samples 20 \
  --num_rollouts 4 \
  --batch_size 4 \
  --learning_rate 1e-6 \
  --kl_coef 0.04 \
  --num_epochs 1 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --max_rounds 2 \
  --temperature 0.9 \
  --max_new_tokens 96 \
  --finalize_max_new_tokens 64 \
  --policy_forward_micro_batch_size 4 \
  --worker_timeout 600 \
  --max_steps 3 \
  --rollout_worker_mode persistent \
  --grpo_forward_mode text_only

echo "[$(date)] Test 2 complete"

#!/bin/bash
#SBATCH -J interleaved_eval20
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/03_interleaved/slurm-interleaved-eval20-%j.out

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1

# Auto-detect GPU with most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

echo "[$(date)] Starting interleaved eval (20 samples)"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python -u scripts/03_interleaved/batch_interleaved_eval.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --manifest_path "output/GeneratedData/eval_manifest_500.jsonl" \
  --output_dir "output/interleaved_eval" \
  --run_name "v9b_2epoch" \
  --num_samples 20 \
  --max_rounds 5 \
  --max_new_tokens_per_round 128 \
  --temperature 0.7 \
  --tmp_dir "output/interleaved_tmp"

echo "[$(date)] All done"

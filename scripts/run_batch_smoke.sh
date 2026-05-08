#!/bin/bash
#SBATCH -J batch_smoke
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/slurm-batch-%j.out
#SBATCH -e scripts/slurm-batch-%j.err

set -e

# ── 工作目录 ──
cd /home/s2025244189/s2025244265/Projects/Echo_Project

# ── 指定空闲 GPU ──
export CUDA_VISIBLE_DEVICES=4

# ── conda ──
source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

# ── paths ──
BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/home/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v7-20260505-145145/checkpoint-749"
SKELETON="output/GeneratedData/qa_skeleton.jsonl"
OUTPUT_DIR="output/interleaved"
NUM_SAMPLES=20

mkdir -p "$OUTPUT_DIR" output/interleaved_tmp

echo "[$(date)] Starting batch interleaved smoke test"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Samples: $NUM_SAMPLES"
echo "  Output:  $OUTPUT_DIR"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python -u scripts/batch_interleaved_smoke.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --skeleton_path "$SKELETON" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --seed 42 \
  --max_rounds 5 \
  --max_new_tokens_per_round 128 \
  --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Batch interleaved inference completed"

#!/bin/bash
#SBATCH -J targeted_v2
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/slurm-targeted-v2-%j.out
#SBATCH -e scripts/slurm-targeted-v2-%j.err

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export CUDA_VISIBLE_DEVICES=4

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/home/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v7-20260505-145145/checkpoint-749"
OUTPUT_DIR="output/interleaved/targeted_v2_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR" output/interleaved_tmp

echo "[$(date)] Starting targeted v2 (single model load)"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Output:  $OUTPUT_DIR"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Model loads once inside the Python script
python -u scripts/targeted_five.py \
  "$BASE_MODEL" "$ADAPTER" "$OUTPUT_DIR"

echo "[$(date)] All done"

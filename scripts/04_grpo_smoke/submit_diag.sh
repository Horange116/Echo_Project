#!/bin/bash
#SBATCH -J diag_audioset
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/04_grpo_smoke/diag-%j.out

set -e
cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)
source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo
python -u scripts/04_grpo_smoke/diag_audioset.py

#!/bin/bash
#SBATCH -J custom_loop_5
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -o scripts/04_grpo_smoke/custom_loop_5.out
#SBATCH -e scripts/04_grpo_smoke/custom_loop_5.err
#SBATCH --mem=80G
#SBATCH --qos=qmultiple9
#SBATCH -t 0:30:00

source /home/s2025244189/miniconda3/bin/activate qwen_echo
export QWEN_OMNI_SKIP_SPK=1
export HF_HUB_OFFLINE=1

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project

python3 -u scripts/04_grpo_smoke/test_custom_loop_5.py

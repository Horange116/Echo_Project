#!/bin/bash
#SBATCH -J eval_v9b_2epoch
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-eval-v9b-2epoch-%j.out

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export CUDA_VISIBLE_DEVICES=5

CHECKPOINT="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

/home/s2025244189/miniconda3/envs/qwen_echo/bin/python scripts/01_fixed_eval/eval_from_manifest.py \
    --model_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --adapter_path "$CHECKPOINT" \
    --eval_manifest "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eval_manifest_500.jsonl" \
    --output_dir "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/eval_results/v9b_diverse_cot_2epoch_eval_500" \
    --batch_size 8 \
    --max_new_tokens 256

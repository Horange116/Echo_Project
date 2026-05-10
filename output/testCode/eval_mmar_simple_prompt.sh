#!/bin/bash
#SBATCH -J mmar_simple
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-mmar-simple-%j.out

cd /home/s2025244189/s2025244265/Projects/Echo_Project

MMAR_BASE="/hpai/aios3.0/private/user/s2025244189/s2025244180/Dataset/MMAR"

/home/s2025244189/miniconda3/envs/qwen_echo/bin/python scripts/01_fixed_eval/eval_mmar_simple_prompt.py \
    --model_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --test_jsonl "${MMAR_BASE}/sft/mmar_all.jsonl" \
    --audio_dir "${MMAR_BASE}/mmar-audio" \
    --output_dir "/home/s2025244189/s2025244265/Projects/Echo_Project/output/MMAR_eval/base_simple_prompt" \
    --batch_size 8 \
    --max_new_tokens 128 \
    --temperature 0.7

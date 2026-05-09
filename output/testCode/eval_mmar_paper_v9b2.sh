#!/bin/bash
#SBATCH -J mmar_v9b2_paper
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-mmar-v9b2-paper-%j.out

cd /home/s2025244189/s2025244265/Projects/Echo_Project

MMAR_BASE="/hpai/aios3.0/private/user/s2025244189/s2025244180/Dataset/MMAR"
CKPT="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

/home/s2025244189/miniconda3/envs/qwen_echo/bin/python scripts/01_fixed_eval/eval_mmar_paper_prompt.py \
    --model_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --adapter_path "${CKPT}" \
    --test_jsonl "${MMAR_BASE}/sft/mmar_all.jsonl" \
    --audio_dir "${MMAR_BASE}/mmar-audio" \
    --output_dir "/home/s2025244189/s2025244265/Projects/Echo_Project/output/MMAR_eval/v9b_2epoch_paper_prompt" \
    --batch_size 8 \
    --max_new_tokens 256

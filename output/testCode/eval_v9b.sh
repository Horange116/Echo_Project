#!/bin/bash
#SBATCH -J eval_v9b
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-eval-v9b-%j.out

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/
export QWEN_OMNI_SKIP_SPK=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/s2025244189/miniconda3/envs/qwen_echo/bin/python scripts/01_fixed_eval/eval_from_manifest.py \
    --model_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --adapter_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-clean-diverse-cot-20260508-212134/v0-20260508-212211/checkpoint-1539" \
    --eval_manifest "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eval_manifest_500.jsonl" \
    --output_dir "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/eval_v9b_manifest500" \
    --batch_size 16 \
    --max_new_tokens 256

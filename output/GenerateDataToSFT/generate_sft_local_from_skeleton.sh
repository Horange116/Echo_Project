#!/bin/bash
#SBATCH -J local_13000
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=qmultiple9
#SBATCH -o output/GenerateDataToSFT/slurm-local-13000-%j.out

export CUDA_VISIBLE_DEVICES=5
python "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GenerateDataToSFT/generate_sft_local_from_skeleton.py"
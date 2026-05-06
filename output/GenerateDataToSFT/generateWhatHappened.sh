#!/bin/bash
#SBATCH -J WhatHappened
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-%j.out

export CUDA_VISIBLE_DEVICES=4
python "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GenerateDataToSFT/generateWhatHappened.py"

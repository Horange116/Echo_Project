#!/bin/bash
#SBATCH -J testBeforeRun
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --qos qmultiple9

export CUDA_VISIBLE_DEVICES=3
python "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo/output/test/testBeforeRun.py"
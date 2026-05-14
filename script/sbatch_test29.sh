#!/bin/bash
#SBATCH --job-name=test29_n8_smoke
#SBATCH --output=/home/s2025244189/s2025244265/Projects/Echo_Project/script/sbatch_test29_%j.out
#SBATCH --error=/home/s2025244189/s2025244265/Projects/Echo_Project/script/sbatch_test29_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=A800Z
#SBATCH --qos=qmultiple9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

bash /home/s2025244189/s2025244265/Projects/Echo_Project/script/test29_rollout_n8_smoke.sh

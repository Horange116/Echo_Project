#!/bin/bash
#SBATCH --job-name=test33_a4_scaleup
#SBATCH --output=/home/s2025244189/s2025244265/Projects/Echo_Project/script/sbatch_test33_%j.out
#SBATCH --error=/home/s2025244189/s2025244265/Projects/Echo_Project/script/sbatch_test33_%j.err
#SBATCH --gres=gpu:2
#SBATCH --partition=A800Z
#SBATCH --qos=qmultiple9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

bash /home/s2025244189/s2025244265/Projects/Echo_Project/script/test33_a4_scaleup_split.sh
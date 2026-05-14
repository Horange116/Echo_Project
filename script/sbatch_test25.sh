#!/bin/bash
#SBATCH -p A800Z
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=script/sbatch_test25_%%j.out
#SBATCH --error=script/sbatch_test25_%%j.err

GPU_ID=0 bash script/test25_grpo_scale_after_eps_fix.sh

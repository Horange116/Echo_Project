#!/bin/bash
# DeepSpeed ZeRO smoke test for Qwen2.5-Omni-7B + LoRA
#SBATCH -J ds_zero_smoke
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=qmultiple9
#SBATCH -o output/testCode/ds-zero-smoke-%j.out

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project

export QWEN_OMNI_SKIP_SPK=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="verl:$PYTHONPATH"
unset ROCR_VISIBLE_DEVICES

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER_PATH="output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"
AUDIO_PATH="/home/s2025244189/s2025244265/Projects/Echo_Project/mnt/bn/wdq-base1/data/ALMs/EAQA/audios/AudioSet/audio_21.wav"
REPORT_JSON="output/debug/ds_zero_smoke_report.json"

echo "[$(date)] DeepSpeed ZeRO Smoke Test"
echo "  Base:   $BASE_MODEL"
echo "  Adapter: $ADAPTER_PATH"
echo "  Audio:  $AUDIO_PATH"
echo "  Report: $REPORT_JSON"
echo "  Host:   $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

mkdir -p output/debug

# Run ZeRO-2 test
echo "[$(date)] === Stage 2 ==="
torchrun --nproc_per_node=2 scripts/debug/ds_zero_qwen_omni_smoke.py \
    --zero_stage 2 \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --audio_path "$AUDIO_PATH" \
    --report_json "$REPORT_JSON"

echo "[$(date)] ZeRO-2 exit code: $?"

# Run ZeRO-3 test
echo "[$(date)] === Stage 3 ==="
torchrun --nproc_per_node=2 scripts/debug/ds_zero_qwen_omni_smoke.py \
    --zero_stage 3 \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --audio_path "$AUDIO_PATH" \
    --report_json "$REPORT_JSON"

echo "[$(date)] ZeRO-3 exit code: $?"
echo "[$(date)] DeepSpeed smoke test complete"

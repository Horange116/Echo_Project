#!/bin/bash
# Submit interleaved strategy comparison A/B/C/D
#
# Usage: sbatch scripts/03_interleaved/submit_strategy_comparison.sh
#
# Each strategy runs independently on 1 GPU. All 4 can run in parallel.
#SBATCH -J strat_compare
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/03_interleaved/slurm-strat-compare-%j.out

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1

# Auto-detect GPU with most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

echo "[$(date)] Starting interleaved strategy comparison"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Host:    $(hostname)"
echo "  GPU:     $CUDA_VISIBLE_DEVICES"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# ── Strategy A: No guard + No finalize ──
#   on_duplicate_seg=ignore_continue, finalize_on_stop=False
#   ≈ original basic version behavior (full multi-round, no answer forced)
echo ""
echo "═══════════════════════════════════════════"
echo "  Strategy A: ignore_continue + no finalize"
echo "═══════════════════════════════════════════"
python -u scripts/03_interleaved/batch_interleaved_eval.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --manifest_path "output/GeneratedData/eval_manifest_500.jsonl" \
  --output_dir "output/interleaved_eval/strategy_compare" \
  --run_name "A_ignore_no_final" \
  --num_samples 20 \
  --max_rounds 5 \
  --temperature 0.7 \
  --tmp_dir "output/interleaved_tmp" \
  --on_duplicate_seg ignore_continue \
  --finalize_on_stop false

# ── Strategy B: Stop on dup + Finalize (current default) ──
echo ""
echo "═══════════════════════════════════════════"
echo "  Strategy B: stop + finalize (default)"
echo "═══════════════════════════════════════════"
python -u scripts/03_interleaved/batch_interleaved_eval.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --manifest_path "output/GeneratedData/eval_manifest_500.jsonl" \
  --output_dir "output/interleaved_eval/strategy_compare" \
  --run_name "B_stop_finalize" \
  --num_samples 20 \
  --max_rounds 5 \
  --temperature 0.7 \
  --tmp_dir "output/interleaved_tmp" \
  --on_duplicate_seg stop \
  --finalize_on_stop true

# ── Strategy C: Ignore dup + Finalize at max_rounds ──
echo ""
echo "═══════════════════════════════════════════"
echo "  Strategy C: ignore_continue + finalize"
echo "═══════════════════════════════════════════"
python -u scripts/03_interleaved/batch_interleaved_eval.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --manifest_path "output/GeneratedData/eval_manifest_500.jsonl" \
  --output_dir "output/interleaved_eval/strategy_compare" \
  --run_name "C_ignore_finalize" \
  --num_samples 20 \
  --max_rounds 5 \
  --temperature 0.7 \
  --tmp_dir "output/interleaved_tmp" \
  --on_duplicate_seg ignore_continue \
  --finalize_on_stop true

# ── Strategy D: Insert-once + Finalize at max_rounds ──
echo ""
echo "═══════════════════════════════════════════"
echo "  Strategy D: insert_once_continue + finalize"
echo "═══════════════════════════════════════════"
python -u scripts/03_interleaved/batch_interleaved_eval.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --manifest_path "output/GeneratedData/eval_manifest_500.jsonl" \
  --output_dir "output/interleaved_eval/strategy_compare" \
  --run_name "D_insert_once_finalize" \
  --num_samples 20 \
  --max_rounds 5 \
  --temperature 0.7 \
  --tmp_dir "output/interleaved_tmp" \
  --on_duplicate_seg insert_once_continue \
  --finalize_on_stop true

echo ""
echo "[$(date)] All strategies completed"

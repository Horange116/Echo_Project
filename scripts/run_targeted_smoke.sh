#!/bin/bash
#SBATCH -J targeted_smoke
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/slurm-targeted-%j.out
#SBATCH -e scripts/slurm-targeted-%j.err

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export CUDA_VISIBLE_DEVICES=4

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/home/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v7-20260505-145145/checkpoint-749"
OUTPUT_DIR="output/interleaved/targeted_smoke_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR" output/interleaved_tmp

echo "[$(date)] Starting targeted interleaved smoke test (5 samples)"
echo "  Model:   $BASE_MODEL"
echo "  Adapter: $ADAPTER"
echo "  Output:  $OUTPUT_DIR"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# ============ Sample 1: gap ============
python -u scripts/interleaved_infer.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --audio_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--7UmfOkRbM_30000.flac" \
  --question "How long after the first male speech ends does the first human voice begin?" \
  --choices '["0.4 seconds", "0.7 seconds", "0.1 seconds", "0 seconds"]' \
  --output_json "$OUTPUT_DIR/01_gap.json" \
  --max_rounds 5 --max_new_tokens_per_round 128 --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Sample 1/5 done"

# ============ Sample 2: count_before ============
python -u scripts/interleaved_infer.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --audio_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--BIwg9KRxI_130000.flac" \
  --question "How many timestamped sound events have finished before the third noise begins?" \
  --choices '["1", "3", "2", "4"]' \
  --output_json "$OUTPUT_DIR/02_count_before.json" \
  --max_rounds 5 --max_new_tokens_per_round 128 --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Sample 2/5 done"

# ============ Sample 3: repeated_event_gap ============
python -u scripts/interleaved_infer.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --audio_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--CHY2qO5zc_30000.flac" \
  --question "How much time passes between the end of the first alarm clock and the start of the second alarm clock?" \
  --choices '["0.1 seconds", "0.4 seconds", "1.0 second", "0.7 seconds"]' \
  --output_json "$OUTPUT_DIR/03_repeated_event_gap.json" \
  --max_rounds 5 --max_new_tokens_per_round 128 --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Sample 3/5 done"

# ============ Sample 4: duration_compare ============
python -u scripts/interleaved_infer.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --audio_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--EnKcYsPas_210000.flac" \
  --question "Which event lasts longer, the cough or the whispering?" \
  --choices '["the cough", "the whispering", "they last the same amount of time", "neither sound is audible"]' \
  --output_json "$OUTPUT_DIR/04_duration_compare.json" \
  --max_rounds 5 --max_new_tokens_per_round 128 --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Sample 4/5 done"

# ============ Sample 5: gap ============
python -u scripts/interleaved_infer.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --audio_path "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--N8Xc-3C3k_30000.flac" \
  --question "How long after the first male speech ends does the impact sounds begin?" \
  --choices '["1.0 second", "0.1 seconds", "0.7 seconds", "0.4 seconds"]' \
  --output_json "$OUTPUT_DIR/05_gap.json" \
  --max_rounds 5 --max_new_tokens_per_round 128 --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Sample 5/5 done"

# === Summary ===
echo ""
echo "=============================================="
echo "  Targeted Smoke Test Complete"
echo "  Output dir: $OUTPUT_DIR"
echo "=============================================="

for f in "$OUTPUT_DIR"/*.json; do
    echo ""
    echo "--- $(basename $f) ---"
    python3 -u -c "
import json
with open('$f') as fh:
    r = json.load(fh)
print(f'  final_answer: {r[\"final_answer\"]}')
print(f'  num_rounds:   {r[\"num_rounds\"]}')
for rd in r.get('round_debug', []):
    print(f'  Round {rd[\"round\"]}: stop={rd[\"round_stop_reason\"]} '
          f'insert={\"✓\" if rd[\"insert_success\"] else \"✗\"} '
          f'continue={\"✓\" if rd[\"continued_after_insert\"] else \"✗\"} '
          f'audios={rd[\"num_audios_before\"]}->{rd[\"num_audios_after\"]} '
          f'seg={rd[\"detected_seg_text\"] or \"—\"}')
print(f'  segments:     {len(r[\"used_segments\"])}')
print(f'  parse_errors: {r.get(\"parse_errors\")}')
    "
done

echo "[$(date)] All done"

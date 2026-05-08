#!/bin/bash
#SBATCH -J interleaved_smoke
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/slurm-interleaved-%j.out
#SBATCH -e scripts/slurm-interleaved-%j.err

set -e

# ── 工作目录 ──
cd /home/s2025244189/s2025244265/Projects/Echo_Project

# ── 指定空闲 GPU ──
export CUDA_VISIBLE_DEVICES=4

# ── conda ──
source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

# ── paths ──
BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/home/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v7-20260505-145145/checkpoint-749"
AUDIO="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00058-of-00216_37e3489e/audios/C1dXq04kxKc_30000.flac"
OUTPUT="/home/s2025244189/s2025244265/Projects/Echo_Project/output/interleaved/smoke_result_v7.json"

mkdir -p "$(dirname "$OUTPUT")" output/interleaved_tmp

echo "[$(date)] Starting interleaved inference smoke test"
echo "  Base model: $BASE_MODEL"
echo "  Adapter:    $ADAPTER"
echo "  Audio:      $AUDIO"
echo "  Output:     $OUTPUT"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python -u scripts/interleaved_infer.py \
  --model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER" \
  --audio_path "$AUDIO" \
  --question "For how long is the thunder audible while the /m/0bzvm2 is also present?" \
  --choices '["2.8 seconds", "1.9 seconds", "2.5 seconds", "2.2 seconds"]' \
  --output_json "$OUTPUT" \
  --max_rounds 5 \
  --max_new_tokens_per_round 128 \
  --temperature 0.7 \
  --tmp_dir output/interleaved_tmp

echo "[$(date)] Interleaved inference completed"

# Print result summary
python -u -c "
import json
with open('$OUTPUT') as f:
    r = json.load(f)
print('=== Result ===')
print(f'  Final answer: {r[\"final_answer\"]}')
print(f'  Rounds:       {r[\"num_rounds\"]}')
print(f'  Segments:     {len(r[\"used_segments\"])}')
print(f'  Parse errors: {r.get(\"parse_errors\")}')
for s in r.get('used_segments', []):
    print(f'  Seg: round={s[\"round\"]} [{s[\"start\"]:.2f}, {s[\"end\"]:.2f}]')
"

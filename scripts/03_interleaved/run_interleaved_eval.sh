#!/bin/bash
#SBATCH -J interleaved_eval
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o scripts/03_interleaved/slurm-interleaved-eval-%j.out

set -e

cd /home/s2025244189/s2025244265/Projects/Echo_Project
export QWEN_OMNI_SKIP_SPK=1

# Pick freest GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

BASE_MODEL="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
ADAPTER="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"
MANIFEST="output/GeneratedData/eval_manifest_500.jsonl"
N_SAMPLES=${1:-10}
OUTPUT_DIR="output/interleaved_eval/v9b_2epoch_smoke${N_SAMPLES}"

mkdir -p "$OUTPUT_DIR" output/interleaved_tmp

echo "[$(date)] Starting interleaved eval ($N_SAMPLES samples)"
echo "  Model:    $BASE_MODEL"
echo "  Adapter:  $ADAPTER"
echo "  Manifest: $MANIFEST"
echo "  Output:   $OUTPUT_DIR"

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Read manifest and run interleaved inference for N samples
python3 -u -c "
import json, os, sys, time, subprocess
from datetime import datetime

manifest_path = '$MANIFEST'
n = $N_SAMPLES
output_dir = '$OUTPUT_DIR'
model_path = '$BASE_MODEL'
adapter_path = '$ADAPTER'

samples = []
with open(manifest_path) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

valid = [s for s in samples if os.path.exists(s.get('audio_path', ''))]
selected = valid[:n]

print(f'Manifest: {len(samples)} total, {len(valid)} valid, selected {len(selected)}')

results = []
for i, s in enumerate(selected):
    sid = s.get('id', f'sample_{i}')
    print(f\"\\n[{datetime.now()}] [{i+1}/{len(selected)}] {sid}\")
    print(f'  Q: {s[\"question\"]}')
    print(f'  GT: {s[\"answer\"]}')
    print(f'  Audio: {os.path.basename(s[\"audio_path\"])}')

    # Call interleaved_infer.py for each sample (loads model each time)
    out_path = os.path.join(output_dir, f'sample_{i}.json')
    t0 = time.time()

    cmd = [
        sys.executable, 'scripts/interleaved_infer.py',
        '--model_path', model_path,
        '--adapter_path', adapter_path,
        '--audio_path', s['audio_path'],
        '--question', s['question'],
        '--choices', json.dumps(s['choices']),
        '--gold_answer', s.get('answer', ''),
        '--output_json', out_path,
        '--max_rounds', '5',
        '--max_new_tokens_per_round', '128',
        '--temperature', '0.7',
        '--tmp_dir', 'output/interleaved_tmp',
    ]
    subprocess.run(cmd, check=True)

    elapsed = time.time() - t0

    with open(out_path) as f:
        r = json.load(f)

    pred = r.get('pred_answer', r.get('final_answer', ''))
    gt = r.get('gold_answer', s.get('answer', ''))
    is_correct = (pred == gt) and bool(pred) and bool(gt)
    used_segs = len(r.get('used_segments', []))
    triggered = any(rd.get('insert_success', False) for rd in r.get('round_debug', []))

    results.append({
        'id': sid,
        'pred_answer': pred,
        'ground_truth': gt,
        'is_correct': is_correct,
        'num_rounds': r.get('num_rounds', 0),
        'used_segments': used_segs,
        'interleaved_triggered': triggered,
        'elapsed': round(elapsed, 1),
        'has_finalize': r.get('has_final_answer', False),
        'stop_reason': r.get('stop_reason', ''),
        'duplicate_segments': r.get('num_duplicate_segments', 0),
        'final_response_preview': r.get('final_response', '')[:200],
    })

    print(f'  Pred: \"{pred}\" | GT: \"{gt}\" | {\"✓\" if is_correct else \"✗\"} | {elapsed:.1f}s')
    print(f'  Rounds: {r.get(\"num_rounds\")} | Segs: {used_segs} | Finalize: {r.get(\"finalize_used\", False)}')

# Summary
correct = sum(1 for r in results if r['is_correct'])
triggered = sum(1 for r in results if r['interleaved_triggered'])
finalized = sum(1 for r in results if r['has_finalize'])
dup_detected = sum(1 for r in results if r['duplicate_segments'] > 0)
total_segs = sum(r['used_segments'] for r in results)
errors = sum(1 for r in results if r.get('error'))

summary = {
    'timestamp': str(datetime.now()),
    'config': {
        'model_path': model_path,
        'adapter_path': adapter_path,
        'manifest_path': manifest_path,
        'num_samples': len(selected),
    },
    'results': results,
    'totals': {
        'correct': correct,
        'accuracy': round(correct/len(results)*100, 1) if results else 0,
        'interleaved_triggered': triggered,
        'finalize_used': finalized,
        'duplicate_detected': dup_detected,
        'avg_segments': round(total_segs/len(results), 1) if results else 0,
        'total_elapsed': round(sum(r['elapsed'] for r in results), 1),
        'errors': errors,
    },
}

with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Also write JSONL
with open(os.path.join(output_dir, 'results.jsonl'), 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print()
print('='*60)
print(f'  Samples: {len(results)}')
print(f'  Correct: {correct} ({summary[\"totals\"][\"accuracy\"]}%)')
print(f'  Interleaved triggered: {triggered}/{len(results)}')
print(f'  Finalize used: {finalized}/{len(results)}')
print(f'  Duplicate detected: {dup_detected}/{len(results)}')
print(f'  Avg segments: {summary[\"totals\"][\"avg_segments\"]}')
print(f'  Total time: {summary[\"totals\"][\"total_elapsed\"]}s')
print(f'  Output: {output_dir}/')
print('='*60)
"

echo "[$(date)] All done"

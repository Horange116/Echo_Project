#!/usr/bin/env python3
"""Quick test of run_interleaved_custom on 5 samples."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from interleaved_infer_custom import load_model_and_processor, run_interleaved_custom

MODEL_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
DATA_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/judge/split_rl.jsonl"
OUT_DIR = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/interleaved_eval/custom_loop_test"
CONTINUE_MODE = "assistant_append"  # prompt | silent | assistant_append

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
with open(DATA_PATH) as f:
    all_data = [json.loads(line) for line in f]

# Filter valid
valid = [d for d in all_data if os.path.exists(d["audio_path"])]
batch = valid[:5]
print(f"Testing {len(batch)} samples with continue_mode='{CONTINUE_MODE}'")

# Load model
print("Loading model...")
t0 = time.time()
model, processor = load_model_and_processor(MODEL_PATH)
print(f"Model loaded ({time.time()-t0:.1f}s)")

# Run
results = []
for i, item in enumerate(batch):
    print(f"\n--- [{i+1}/{len(batch)}] {item['id']} ---")
    t_start = time.time()
    try:
        result = run_interleaved_custom(
            model, processor,
            audio_path=item["audio_path"],
            question=item["question"],
            choices=item["choices"],
            max_rounds=5,
            max_new_tokens_per_round=128,
            temperature=0.7,
            tmp_dir=os.path.join(OUT_DIR, "tmp"),
            gold_answer=item["answer"],
            continue_mode=CONTINUE_MODE,
        )
        elapsed = time.time() - t_start
        print(f"  rounds={result['total_rounds']} segs={result['num_inserted_segments']} "
              f"pred={result['pred_answer']} gold={item['answer']} "
              f"correct={result.get('answer_correct')} ({elapsed:.1f}s)")
        results.append(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append({"sample_id": item["id"], "error": str(e)})

# Summary
correct = sum(1 for r in results if r.get("answer_correct") == True)
has_ans = sum(1 for r in results if r.get("has_final_answer"))
print(f"\n{'='*50}")
print(f"RESULTS: {correct}/{len(batch)} correct, {has_ans}/{len(batch)} with answer")
for r in results:
    print(f"  {r.get('sample_id','?'):30s} rounds={r.get('total_rounds')} segs={r.get('num_inserted_segments')} "
          f"pred={r.get('pred_answer','?'):8s} correct={r.get('answer_correct')}")

# Save
out_path = os.path.join(OUT_DIR, "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")

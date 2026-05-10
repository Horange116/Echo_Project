"""
Batch interleaved inference for v9b-2epoch model evaluation.

Loads model once, runs N samples from split_rl.jsonl,
saves per-sample results to a single JSON file.
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.interleaved_infer import load_model_and_processor, run_interleaved


def main():
    parser = argparse.ArgumentParser(description="Batch interleaved inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_path", default="output/judge/split_rl.jsonl")
    parser.add_argument("--output_dir", default="output/interleaved_eval/v9b_2epoch_batch20")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue_mode", default="prompt",
                        choices=["prompt", "silent", "context"],
                        help="Continue mode for seg insertion rounds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.data_path) as f:
        all_data = [json.loads(line) for line in f]
    print(f"Loaded {len(all_data)} samples from {args.data_path}")

    # Filter valid audio files
    valid = [d for d in all_data if os.path.exists(d["audio_path"])]
    skipped = len(all_data) - len(valid)
    if skipped:
        print(f"Skipped {skipped} samples with missing audio")

    # Take first num_samples
    batch = valid[:args.num_samples]
    print(f"Running {len(batch)} samples")

    # Load model
    print(f"Loading model from {args.model_path}")
    t0 = time.time()
    model, processor = load_model_and_processor(args.model_path, args.adapter_path)
    print(f"Model loaded ({time.time() - t0:.1f}s), device: {model.device}")

    # Run inference
    results = []
    summary_rows = []
    n_correct = 0
    n_with_answer = 0

    for i, item in enumerate(batch):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(batch)}] {item['id']}")
        print(f"  Question: {item['question'][:80]}...")
        print(f"  Choices: {item['choices']}")
        print(f"  Answer: {item['answer']}")

        t_start = time.time()
        try:
            result = run_interleaved(
                model, processor,
                audio_path=item["audio_path"],
                question=item["question"],
                choices=item["choices"],
                max_rounds=args.max_rounds,
                max_new_tokens_per_round=args.max_new_tokens,
                temperature=args.temperature,
                sample_rate=args.sample_rate,
                tmp_dir=os.path.join(args.output_dir, "tmp"),
                gold_answer=item["answer"],
                continue_mode=args.continue_mode,
            )
            elapsed = time.time() - t_start
            result["sample_id"] = item["id"]
            result["elapsed"] = round(elapsed, 1)
            results.append(result)

            correct = result.get("answer_correct")
            has_ans = result.get("has_final_answer", False)
            if has_ans:
                n_with_answer += 1
            if correct:
                n_correct += 1

            row = {
                "id": item["id"],
                "correct": correct,
                "pred": result.get("pred_answer"),
                "gold": item["answer"],
                "rounds": result["total_rounds"],
                "unique_segs": result["num_inserted_segments"],
                "dup_segs": result["num_duplicate_segments"],
                "stop": result["stop_reason"],
                "elapsed": round(elapsed, 1),
            }
            summary_rows.append(row)
            print(f"  -> pred={result.get('pred_answer')}, gold={item['answer']}, "
                  f"correct={correct}, rounds={result['total_rounds']}, "
                  f"segs={result['num_inserted_segments']}, "
                  f"dup={result['num_duplicate_segments']}, "
                  f"stop={result['stop_reason']}, {elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "sample_id": item["id"],
                "error": str(e),
                "question": item["question"],
                "choices": item["choices"],
                "gold_answer": item["answer"],
            })

    # Summary
    total = len(batch)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({total} samples)")
    print(f"  Accuracy:      {n_correct}/{total} ({n_correct/total*100:.1f}%)" if total else "  Accuracy:      N/A")
    print(f"  With answer:   {n_with_answer}/{total}")
    print(f"  Avg rounds:    {sum(r.get('total_rounds',0) for r in results if 'total_rounds' in r)/max(sum(1 for r in results if 'total_rounds' in r),1):.1f}")
    print(f"  Avg unique segs: {sum(r.get('num_inserted_segments',0) for r in results if 'num_inserted_segments' in r)/max(sum(1 for r in results if 'num_inserted_segments' in r),1):.1f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    output = {
        "config": vars(args),
        "summary": {
            "total": total,
            "correct": n_correct,
            "accuracy": round(n_correct / total, 4) if total else 0,
            "with_answer": n_with_answer,
        },
        "results": results,
        "summary_rows": summary_rows,
    }
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_file}")

    # Also save a compact CSV-like summary
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

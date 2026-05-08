#!/usr/bin/env python3
"""
Batch interleaved inference smoke test.

Loads model once, runs N samples from qa_skeleton.jsonl,
and writes individual results + a summary.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from interleaved_infer import load_model_and_processor, run_interleaved


def pick_samples(skeleton_path, n, seed=None):
    """Pick n samples from qa_skeleton.jsonl that have existing audio files."""
    import random as _random
    if seed is not None:
        _random.seed(seed)

    samples = []
    with open(skeleton_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    # Filter: audio file must exist
    valid = [s for s in samples if os.path.exists(s["audio_path"])]
    skipped = len(samples) - len(valid)

    if len(valid) == 0:
        print("错误: 没有找到任何 audio_path 存在的样本")
        sys.exit(1)

    _random.shuffle(valid)
    selected = valid[:n]

    print(f"  QA 样本总数: {len(samples)}")
    print(f"  有效(音频存在): {len(valid)}")
    print(f"  跳过(音频缺失): {skipped}")
    print(f"  本次选取: {len(selected)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Batch interleaved inference smoke test")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--skeleton_path", default="output/GeneratedData/qa_skeleton.jsonl")
    parser.add_argument("--output_dir", default="output/interleaved")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--max_new_tokens_per_round", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tmp_dir", default="output/interleaved_tmp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    # Pick samples
    print(f"[{datetime.now()}] 选取 {args.num_samples} 个测试样本...")
    samples = pick_samples(args.skeleton_path, args.num_samples, seed=args.seed)

    # Load model once
    print(f"[{datetime.now()}] 加载模型: {args.model_path}")
    t0 = time.time()
    model, processor = load_model_and_processor(args.model_path, args.adapter_path)
    print(f"[{datetime.now()}] 模型加载完成 (耗时 {time.time() - t0:.1f}s), device: {model.device}")

    # Run batch
    all_results = []
    correct = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        question = sample["question"]
        choices = sample["choices"]
        answer_gt = sample["answer"]
        audio_path = sample["audio_path"]
        skeleton_id = sample.get("skeleton_id", f"sample_{i}")

        print(f"\n[{datetime.now()}] [{i+1}/{total}] {skeleton_id}")
        print(f"  Q: {question}")
        print(f"  GT: {answer_gt}")
        print(f"  Audio: {os.path.basename(audio_path)}")

        t1 = time.time()
        result = run_interleaved(
            model, processor,
            audio_path=audio_path,
            question=question,
            choices=choices,
            max_rounds=args.max_rounds,
            max_new_tokens_per_round=args.max_new_tokens_per_round,
            temperature=args.temperature,
            tmp_dir=args.tmp_dir,
        )
        elapsed = time.time() - t1

        pred = result["final_answer"]
        is_correct = (pred == answer_gt)
        if is_correct:
            correct += 1

        result["skeleton_id"] = skeleton_id
        result["ground_truth"] = answer_gt
        result["is_correct"] = is_correct
        result["elapsed_seconds"] = round(elapsed, 1)
        all_results.append(result)

        print(f"  Pred: {pred} | GT: {answer_gt} | {'✓' if is_correct else '✗'} | {elapsed:.1f}s")

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    summary = {
        "timestamp": str(datetime.now()),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "num_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 1),
        "results": all_results,
    }

    summary_path = os.path.join(args.output_dir, "batch_result.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  批量测试完成")
    print(f"  总数: {total}, 正确: {correct}, 准确率: {accuracy:.1f}%")
    print(f"  结果写入: {summary_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

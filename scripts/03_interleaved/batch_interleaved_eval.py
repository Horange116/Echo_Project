#!/usr/bin/env python3
"""
Batch interleaved inference evaluation from eval_manifest.

Loads model once, runs N ordered samples from eval_manifest.jsonl,
outputs per-sample JSONL + detailed summary with interleaved diagnostics.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from interleaved_infer import run_interleaved, THINK_ANSWER_PATTERN


def load_model_and_processor(model_path, adapter_path=None):
    """Load model and processor, forced to CUDA (avoids device_map='auto' CPU fallback)."""
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        offload_folder=None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.base_model.disable_talker()
    model.eval()
    return model, processor


def pick_from_manifest(manifest_path, n):
    """Pick first n samples from eval_manifest (ordered, not random)."""
    samples = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    valid = [s for s in samples if os.path.exists(s.get("audio_path", ""))]
    skipped = len(samples) - len(valid)

    if len(valid) == 0:
        print("错误: 没有 audio_path 存在的样本")
        sys.exit(1)

    selected = valid[:n]

    print(f"  Manifest 总数: {len(samples)}")
    print(f"  有效(音频存在): {len(valid)}")
    print(f"  跳过(音频缺失): {skipped}")
    print(f"  本次选取: {len(selected)}")
    return selected


def has_seg_in_response(response):
    return "<seg>" in response and "</seg>" in response


def is_fully_structured(response):
    return bool(THINK_ANSWER_PATTERN.search(response))


def answer_in_choices(answer, choices):
    return answer in choices if choices else False


def main():
    parser = argparse.ArgumentParser(description="Batch interleaved eval from manifest")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--manifest_path", default="output/GeneratedData/eval_manifest_500.jsonl")
    parser.add_argument("--output_dir", default="output/interleaved_eval")
    parser.add_argument("--run_name", default="v9b_2epoch")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--max_new_tokens_per_round", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tmp_dir", default="output/interleaved_tmp")
    # Strategy parameters
    parser.add_argument("--on_duplicate_seg", default="stop",
                        choices=["stop", "ignore_continue", "insert_once_continue"])
    parser.add_argument("--finalize_on_stop", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--duplicate_iou_threshold", type=float, default=0.8)
    parser.add_argument("--max_duplicate_segments", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    output_jsonl = os.path.join(args.output_dir, f"{args.run_name}_smoke{args.num_samples}.jsonl")
    output_summary = os.path.join(args.output_dir, f"{args.run_name}_smoke{args.num_samples}_summary.json")

    # Pick samples (ordered)
    print(f"[{datetime.now()}] 选取 {args.num_samples} 个样本，来源: {args.manifest_path}")
    samples = pick_from_manifest(args.manifest_path, args.num_samples)

    # Load model once
    print(f"[{datetime.now()}] 加载模型: {args.model_path}")
    print(f"  Adapter: {args.adapter_path}")
    t0 = time.time()
    model, processor = load_model_and_processor(args.model_path, args.adapter_path)
    print(f"[{datetime.now()}] 模型加载完成 ({time.time() - t0:.1f}s), device: {model.device}")

    # Run batch
    results = []
    total = len(samples)

    for i, sample in enumerate(samples):
        sid = sample.get("id", f"sample_{i}")
        question = sample["question"]
        choices = sample.get("choices", [])
        answer_gt = sample.get("answer", "")
        audio_path = sample["audio_path"]

        print(f"\n[{datetime.now()}] [{i+1}/{total}] {sid}")
        print(f"  Q: {question}")
        print(f"  GT: {answer_gt}")
        print(f"  Audio: {os.path.basename(audio_path)}")

        t1 = time.time()
        try:
            result = run_interleaved(
                model, processor,
                audio_path=audio_path,
                question=question,
                choices=choices,
                gold_answer=answer_gt,
                max_rounds=args.max_rounds,
                max_new_tokens_per_round=args.max_new_tokens_per_round,
                temperature=args.temperature,
                tmp_dir=args.tmp_dir,
                on_duplicate_seg=args.on_duplicate_seg,
                finalize_on_stop=args.finalize_on_stop,
                duplicate_iou_threshold=args.duplicate_iou_threshold,
                max_duplicate_segments=args.max_duplicate_segments,
            )
            elapsed = time.time() - t1
            error = None
        except Exception as e:
            elapsed = time.time() - t1
            error = str(e)
            result = {
                "final_response": "",
                "final_answer": "",
                "used_segments": [],
                "used_segment_paths": [],
                "round_outputs": [],
                "round_debug": [],
                "num_rounds": 0,
                "parse_errors": [str(e)],
            }

        # Compute metrics
        resp = result.get("final_response", "")
        pred = result.get("pred_answer", result.get("final_answer", ""))
        is_correct = (pred == answer_gt) and bool(pred) and bool(answer_gt)
        seg_count = len(result.get("used_segments", []))
        interleaved_triggered = any(
            rd.get("insert_success", False) for rd in result.get("round_outputs", [])
        )
        total_segment_files = len(result.get("used_segment_paths", []))
        existing_files = sum(
            1 for p in result.get("used_segment_paths", []) if os.path.exists(p)
        )

        # Segment durations (from round_outputs)
        seg_durations = [
            rd.get("clipped_segment_duration", 0)
            for rd in result.get("round_outputs", [])
            if rd.get("clipped_segment_duration") is not None
        ]

        entry = {
            "id": sid,
            "question": question,
            "choices": choices,
            "ground_truth": answer_gt,
            "pred_answer": pred,
            "is_correct": is_correct,
            "error": error,
            "elapsed_seconds": round(elapsed, 1),
            "has_seg": has_seg_in_response(resp),
            "fully_structured": is_fully_structured(resp),
            "answer_in_choices": answer_in_choices(pred, choices),
            "num_rounds": result.get("total_rounds", 0),
            "interleaved_triggered": interleaved_triggered,
            "used_segments_count": seg_count,
            "used_segment_files_exist": existing_files,
            "used_segment_files_total": total_segment_files,
            "segment_durations": seg_durations,
            "total_segment_duration": round(sum(seg_durations), 3),
            "final_response": resp,
            "used_segments": result.get("used_segments", []),
            "used_segment_paths": result.get("used_segment_paths", []),
            "round_debug": result.get("round_outputs", []),
            "parse_errors": result.get("parse_errors"),
        }
        results.append(entry)

        print(f"  Pred: {pred} | GT: {answer_gt} | {'✓' if is_correct else '✗'} | {elapsed:.1f}s")
        print(f"  Rounds: {entry['num_rounds']} | Segs: {seg_count} | Interleaved: {interleaved_triggered}")

    # Write JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    total_ok = sum(1 for r in results if r["is_correct"])
    triggered = sum(1 for r in results if r["interleaved_triggered"])
    has_seg_count = sum(1 for r in results if r["has_seg"])
    fully_struct_count = sum(1 for r in results if r["fully_structured"])
    in_choices_count = sum(1 for r in results if r["answer_in_choices"])
    all_segments = [r["used_segments_count"] for r in results]
    total_seg_files = sum(r["used_segment_files_total"] for r in results)
    total_seg_exist = sum(r["used_segment_files_exist"] for r in results)
    total_parse_errors = sum(len(r.get("parse_errors") or []) for r in results)
    errors = sum(1 for r in results if r["error"] is not None)

    summary = {
        "timestamp": str(datetime.now()),
        "config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "manifest_path": args.manifest_path,
            "num_samples": total,
            "max_rounds": args.max_rounds,
            "max_new_tokens_per_round": args.max_new_tokens_per_round,
            "temperature": args.temperature,
            "on_duplicate_seg": args.on_duplicate_seg,
            "finalize_on_stop": args.finalize_on_stop,
            "duplicate_iou_threshold": args.duplicate_iou_threshold,
            "max_duplicate_segments": args.max_duplicate_segments,
        },
        "summary": {
            "total_samples": total,
            "correct": total_ok,
            "accuracy": round(total_ok / total * 100, 1) if total else 0,
            "errors": errors,
            "interleaved_triggered_count": triggered,
            "interleaved_triggered_pct": round(triggered / total * 100, 1) if total else 0,
            "has_seg_count": has_seg_count,
            "fully_structured_count": fully_struct_count,
            "answer_in_choices_count": in_choices_count,
            "avg_used_segments": round(sum(all_segments) / total, 2) if total else 0,
            "max_used_segments": max(all_segments) if all_segments else 0,
            "total_segment_files_created": total_seg_files,
            "total_segment_files_exist": total_seg_exist,
            "segment_file_existence_pct": round(total_seg_exist / total_seg_files * 100, 1) if total_seg_files else 0,
            "total_parse_errors": total_parse_errors,
            "total_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results), 1),
            "avg_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results) / total, 1) if total else 0,
        },
        "per_sample_results": [
            {
                "id": r["id"],
                "pred_answer": r["pred_answer"],
                "ground_truth": r["ground_truth"],
                "is_correct": r["is_correct"],
                "num_rounds": r["num_rounds"],
                "interleaved_triggered": r["interleaved_triggered"],
                "used_segments_count": r["used_segments_count"],
                "elapsed_seconds": r["elapsed_seconds"],
                "has_seg": r["has_seg"],
                "fully_structured": r["fully_structured"],
                "answer_in_choices": r["answer_in_choices"],
                "total_segment_duration": r["total_segment_duration"],
            }
            for r in results
        ],
    }

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  批量 interleaved eval 完成")
    print(f"  总数: {total}")
    print(f"  正确: {total_ok} ({summary['summary']['accuracy']}%)")
    print(f"  Interleaved 触发: {triggered}/{total}")
    print(f"  平均 used_segments: {summary['summary']['avg_used_segments']}")
    print(f"  总 segment files: {total_seg_files} (存在: {total_seg_exist})")
    print(f"  解析错误: {total_parse_errors}")
    print(f"  结果写入: {output_jsonl}")
    print(f"  摘要写入: {output_summary}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

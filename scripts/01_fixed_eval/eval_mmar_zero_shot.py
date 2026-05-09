#!/usr/bin/env python3
"""
MMAR zero-shot evaluation with base Qwen2.5-Omni-7B (no LoRA).

Usage:
  python scripts/eval_mmar_zero_shot.py \
      --model_path /path/to/Qwen2.5-Omni-7B \
      --test_jsonl /path/to/MMAR/sft/test.jsonl \
      --audio_dir /path/to/MMAR/mmar-audio \
      --output_dir /path/to/output \
      --batch_size 8 \
      --max_new_tokens 64
"""

import argparse
import json
import os
import re
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import librosa
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

SAMPLE_RATE = 16000


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def parse_options_from_question(question):
    """Extract options list from question text like '...\nOptions:\nA. Yes\nB. No'"""
    lines = question.split("\n")
    options = []
    main_question = []
    in_options = False
    for line in lines:
        if line.strip().lower().startswith("options"):
            in_options = True
            continue
        if in_options:
            line = line.strip()
            if re.match(r'^[A-Z]\.\s', line):
                option_text = re.sub(r'^[A-Z]\.\s*', '', line).strip()
                options.append(option_text)
            elif line:
                options.append(line)
        else:
            main_question.append(line)

    return " ".join(main_question).strip(), options


def build_question(item):
    """Build prompt for model inference."""
    q_text, options = parse_options_from_question(item["question"])
    if options:
        choices_str = json.dumps(options, ensure_ascii=False)
        return (
            q_text
            + " Choose the answer from "
            + choices_str
            + ". Answer in the format of <answer>...</answer>."
        )
    return item["question"] + " Answer in the format of <answer>...</answer>."


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_response(response):
    """Extract answer from model response."""
    result = {
        "has_answer_tag": False,
        "pred_answer": "",
    }
    match = ANSWER_PATTERN.search(response or "")
    if match:
        result["has_answer_tag"] = True
        result["pred_answer"] = match.group(1).strip()
    else:
        # fallback: use whole response as answer
        result["pred_answer"] = (response or "").strip()
    return result


def load_model(model_path, adapter_path=None):
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.base_model.disable_talker()
    model.eval()
    return model, processor


def run_batch_inference(model, processor, batch_items, max_new_tokens):
    audio_list = []
    text_list = []
    for item in batch_items:
        audio_data, _ = librosa.load(item["audio_path"], sr=SAMPLE_RATE)
        audio_list.append(audio_data)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        if isinstance(text, list):
            text = text[0]
        text_list.append(text)

    inputs = processor(text=text_list, audio=audio_list, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_audio=False,
            speaker=None,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="MMAR zero-shot evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None,
                        help="LoRA adapter path (optional)")
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "eval_report.json"

    # Load test data
    test_items = list(read_jsonl(args.test_jsonl))
    print(f"MMAR 测试样本数: {len(test_items)}")
    print(f"模型: {args.model_path}")
    print(f"输出: {output_dir}")
    print(f"batch_size: {args.batch_size}")
    print()

    # Load model
    model, processor = load_model(args.model_path, args.adapter_path)

    # Prepare eval records
    eval_records = []
    for item in test_items:
        audio_path = os.path.join(args.audio_dir, item["audio_url"])
        if not os.path.exists(audio_path):
            print(f"  音频不存在: {audio_path}")
            continue
        question_text = build_question(item)
        eval_records.append({
            "id": item["id"],
            "question": question_text,
            "gold_answer": item["answer"].strip(),
            "audio_path": audio_path,
            "raw_question": item["question"],
            "category": item.get("metadata", {}).get("category", "unknown"),
            "modality": item.get("metadata", {}).get("modality", "unknown"),
        })

    print(f"有效样本数: {len(eval_records)}")
    print()

    # Run inference
    stats = Counter()
    cat_stats = {}
    mod_stats = {}

    done_count = 0
    for batch_start in range(0, len(eval_records), args.batch_size):
        batch = eval_records[batch_start: batch_start + args.batch_size]
        batch_payload = [
            {"audio_path": rec["audio_path"], "question": rec["question"]}
            for rec in batch
        ]

        try:
            responses = run_batch_inference(model, processor, batch_payload, args.max_new_tokens)
        except Exception as e:
            print(f"  batch 推理失败，回退单条: {repr(e)[:200]}")
            responses = []
            for payload in batch_payload:
                try:
                    resp = run_batch_inference(model, processor, [payload], args.max_new_tokens)
                    responses.append(resp[0])
                except Exception as e2:
                    responses.append(f"__error__:{e2}")

        for rec, response in zip(batch, responses):
            done_count += 1
            try:
                if isinstance(response, str) and response.startswith("__error__"):
                    raise Exception(response)

                response = str(response).strip()
                parsed = parse_response(response)
                pred_answer = parsed["pred_answer"]
                gold_answer = rec["gold_answer"]
                is_correct = pred_answer.lower() == gold_answer.lower()

                stats["processed"] += 1
                stats["correct"] += int(is_correct)
                stats["has_answer_tag"] += int(parsed["has_answer_tag"])

                cat = rec["category"]
                if cat not in cat_stats:
                    cat_stats[cat] = Counter()
                cat_stats[cat]["processed"] += 1
                cat_stats[cat]["correct"] += int(is_correct)

                mod = rec["modality"]
                if mod not in mod_stats:
                    mod_stats[mod] = Counter()
                mod_stats[mod]["processed"] += 1
                mod_stats[mod]["correct"] += int(is_correct)

                pred_record = {
                    "id": rec["id"],
                    "category": cat,
                    "modality": mod,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "has_answer_tag": parsed["has_answer_tag"],
                    "correct": is_correct,
                    "response": response,
                }
                append_jsonl(predictions_path, pred_record)

                if done_count % 10 == 0:
                    print(f"  [{done_count}/{len(eval_records)}]  acc={stats['correct']}/{stats['processed']} ({stats['correct']/max(1,stats['processed'])*100:.1f}%)")

            except Exception as e:
                stats["failed"] += 1
                pred_record = {
                    "id": rec["id"],
                    "category": rec["category"],
                    "modality": rec["modality"],
                    "gold_answer": rec["gold_answer"],
                    "pred_answer": "",
                    "has_answer_tag": False,
                    "correct": False,
                    "response": str(response) if isinstance(response, str) else "",
                    "error": repr(e),
                }
                append_jsonl(predictions_path, pred_record)

    # Build report
    processed = max(1, stats["processed"])
    correct = stats["correct"]
    report = {
        "model_path": args.model_path,
        "test_jsonl": args.test_jsonl,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "timestamp": timestamp,
        "num_samples": processed,
        "accuracy": correct / processed,
        "correct": correct,
        "total": processed,
        "has_answer_tag_rate": stats["has_answer_tag"] / processed,
        "failed": stats.get("failed", 0),
        "by_category": {k: {"accuracy": v["correct"] / max(1, v["processed"]),
                            "correct": v["correct"], "total": v["processed"]}
                        for k, v in sorted(cat_stats.items())},
        "by_modality": {k: {"accuracy": v["correct"] / max(1, v["processed"]),
                            "correct": v["correct"], "total": v["processed"]}
                        for k, v in sorted(mod_stats.items())},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"MMAR Zero-Shot 评估完成")
    print(f"  总样本: {processed}")
    print(f"  正确: {correct} ({correct/processed*100:.1f}%)")
    print(f"  has_answer_tag: {report['has_answer_tag_rate']:.2%}")
    print(f"  失败: {stats.get('failed', 0)}")
    print(f"  报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

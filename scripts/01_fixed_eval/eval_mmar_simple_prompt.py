#!/usr/bin/env python3
"""
MMAR evaluation with simple prompt — no CoT, no <answer> tag requirement.
Evaluates using MMAR official word-token matching.

Prompt:
    {question} Choose the answer from {choices}. Answer directly with the correct choice.
"""

import argparse
import json
import os
import re
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


def parse_options(question):
    """Extract main question text and choices list from MMAR format."""
    lines = question.split("\n")
    choices = []
    question_lines = []
    in_options = False
    for line in lines:
        if line.strip().lower().startswith("options"):
            in_options = True
            continue
        if in_options:
            line = line.strip()
            m = re.match(r"^[A-Z]\.\s*(.+)", line)
            if m:
                choices.append(m.group(1).strip())
            elif line:
                choices.append(line)
        else:
            question_lines.append(line)
    return " ".join(question_lines).strip(), choices


def build_simple_prompt(question_text, choices):
    """Simple prompt — no CoT, no format requirement."""
    if choices:
        choices_str = json.dumps(choices, ensure_ascii=False)
        prompt = f"{question_text} Choose the answer from {choices_str}. Answer directly with the correct choice."
    else:
        prompt = f"{question_text} Answer directly with the correct choice."
    return prompt


def word_tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def mmar_match(prediction, answer, choices):
    """
    MMAR official evaluation: word-token matching.

    Returns dict with conditions:
      - cond1: all answer tokens in prediction
      - cond2: no wrong-choice tokens in prediction
      - correct: both conditions met
    """
    pred_tokens = word_tokenize(prediction)
    ans_tokens = word_tokenize(answer)
    cond1 = ans_tokens.issubset(pred_tokens) if ans_tokens else False
    incorrect_tokens = set()
    for c in choices:
        if c.lower() != answer.lower():
            incorrect_tokens |= word_tokenize(c)
    cond2 = pred_tokens.isdisjoint(incorrect_tokens) if incorrect_tokens else True
    return {
        "correct": cond1 and cond2,
        "cond1": cond1,
        "cond2": cond2,
    }


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


def run_batch_inference(model, processor, batch_items, max_new_tokens, temperature):
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
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="MMAR evaluation with simple prompt"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
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
    print(f"adapter: {args.adapter_path or '(base model)'}")
    print(f"输出: {output_dir}")
    print(f"max_new_tokens: {args.max_new_tokens}, temperature: {args.temperature}")
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
        q_text, choices = parse_options(item["question"])
        prompt = build_simple_prompt(q_text, choices)
        eval_records.append({
            "id": item["id"],
            "prompt": prompt,
            "choices": choices,
            "gold_answer": item["answer"].strip(),
            "audio_path": audio_path,
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
            {"audio_path": rec["audio_path"], "question": rec["prompt"]}
            for rec in batch
        ]

        try:
            responses = run_batch_inference(
                model, processor, batch_payload, args.max_new_tokens, args.temperature
            )
        except Exception as e:
            print(f"  batch 推理失败，回退单条: {repr(e)[:200]}")
            responses = []
            for payload in batch_payload:
                try:
                    resp = run_batch_inference(
                        model, processor, [payload], args.max_new_tokens, args.temperature
                    )
                    responses.append(resp[0])
                except Exception as e2:
                    responses.append(f"__error__:{e2}")

        for rec, response in zip(batch, responses):
            done_count += 1
            try:
                if isinstance(response, str) and response.startswith("__error__"):
                    raise Exception(response)

                response = str(response).strip()
                result = mmar_match(response, rec["gold_answer"], rec["choices"])
                gold = rec["gold_answer"]

                stats["processed"] += 1
                stats["mmar_correct"] += int(result["correct"])
                stats["cond1"] += int(result["cond1"])
                stats["cond2"] += int(result["cond2"])

                # Category stats
                cat = rec["category"]
                if cat not in cat_stats:
                    cat_stats[cat] = Counter()
                cat_stats[cat]["processed"] += 1
                cat_stats[cat]["correct"] += int(result["correct"])

                # Modality stats
                mod = rec["modality"]
                if mod not in mod_stats:
                    mod_stats[mod] = Counter()
                mod_stats[mod]["processed"] += 1
                mod_stats[mod]["correct"] += int(result["correct"])

                # Write prediction
                pred_record = {
                    "id": rec["id"],
                    "category": cat,
                    "modality": mod,
                    "question": rec["prompt"],
                    "choices": rec["choices"],
                    "gold_answer": gold,
                    "raw_response": response,
                    "mmar_correct": result["correct"],
                    "cond1": result["cond1"],
                    "cond2": result["cond2"],
                }
                append_jsonl(predictions_path, pred_record)

                if done_count % 10 == 0:
                    s = stats
                    p = max(1, s["processed"])
                    print(f"  [{done_count}/{len(eval_records)}]  "
                          f"MMAR_acc={s['mmar_correct']}/{s['processed']} ({s['mmar_correct']/p*100:.1f}%)  "
                          f"cond1={s['cond1']}/{s['processed']} ({s['cond1']/p*100:.1f}%)")

            except Exception as e:
                stats["failed"] += 1
                pred_record = {
                    "id": rec["id"],
                    "category": rec["category"],
                    "modality": rec["modality"],
                    "question": rec["prompt"],
                    "choices": rec["choices"],
                    "gold_answer": rec["gold_answer"],
                    "raw_response": str(response) if isinstance(response, str) else "",
                    "mmar_correct": False,
                    "cond1": False,
                    "cond2": False,
                    "error": repr(e),
                }
                append_jsonl(predictions_path, pred_record)

    # Build report
    p = max(1, stats["processed"])
    report = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "test_jsonl": args.test_jsonl,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
        },
        "prompt_format": "simple_direct_no_cot",
        "evaluation_metric": "MMAR_official_word_token_match",
        "timestamp": timestamp,
        "num_samples": stats["processed"],
        "mmar_accuracy": stats["mmar_correct"] / p,
        "cond1_rate": stats["cond1"] / p,
        "cond2_rate": stats["cond2"] / p,
        "mmar_correct": stats["mmar_correct"],
        "total": stats["processed"],
        "failed": stats["failed"],
        "by_category": {
            k: {
                "total": v["processed"],
                "accuracy": v["correct"] / max(1, v["processed"]),
                "correct": v["correct"],
            }
            for k, v in sorted(cat_stats.items())
        },
        "by_modality": {
            k: {
                "total": v["processed"],
                "accuracy": v["correct"] / max(1, v["processed"]),
                "correct": v["correct"],
            }
            for k, v in sorted(mod_stats.items())
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"MMAR 简单 Prompt 评估完成")
    print(f"  总样本: {stats['processed']}")
    print(f"  MMAR_acc:  {stats['mmar_correct']}/{stats['processed']} ({stats['mmar_correct']/p*100:.1f}%)")
    print(f"  cond1:     {stats['cond1']}/{stats['processed']} ({stats['cond1']/p*100:.1f}%)")
    print(f"  cond2:     {stats['cond2']}/{stats['processed']} ({stats['cond2']/p*100:.1f}%)")
    print(f"  失败: {stats['failed']}")
    print(f"  报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

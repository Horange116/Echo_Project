#!/usr/bin/env python3
"""
MMAR evaluation with Echo paper Appendix E.2 standard prompt.

Paper prompt:
    [QUESTION] Choose the answer from [CHOICES].
    Think step-by-step. Refer to the specific audio segments while thinking,
    and indicate the corresponding timestamps.
    Answer in the format of <think>...</think><answer>...</answer>.

Output:
  - eval_report.json: strict_acc, fallback_acc, has_think_answer, etc.
  - predictions.jsonl: detailed per-sample results
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

# Regex patterns
THINK_ANSWER_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>", re.S
)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.S)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.S)
SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")


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


def build_paper_prompt(question_text, choices):
    """Build Echo paper Appendix E.2 standard prompt."""
    if choices:
        choices_str = json.dumps(choices, ensure_ascii=False)
        prompt = (
            f"{question_text} Choose the answer from {choices_str}. "
            "Think step-by-step. Refer to the specific audio segments while thinking, "
            "and indicate the corresponding timestamps. "
            "Answer in the format of <think>...</think><answer>...</answer>."
        )
    else:
        prompt = (
            f"{question_text} "
            "Think step-by-step. Refer to the specific audio segments while thinking, "
            "and indicate the corresponding timestamps. "
            "Answer in the format of <think>...</think><answer>...</answer>."
        )
    return prompt


def parse_response(response, choices):
    """
    Parse model response for think/answer/seg structure and fallback matching.

    Returns dict with:
      - raw_response
      - has_think_answer: bool (strict <think>...</think><answer>...</answer>)
      - has_answer_tag: bool (<answer> tag present)
      - has_seg: bool (<seg> tag present in think block)
      - strict_pred_answer: str (extracted from <answer> tag)
      - fallback_pred_answer: str (best match from choices in full response)
      - answer_in_choices: bool (strict_pred matches a valid choice)
    """
    result = {
        "has_think_answer": False,
        "has_answer_tag": False,
        "has_seg": False,
        "strict_pred_answer": "",
        "fallback_pred_answer": "",
        "answer_in_choices": False,
    }

    resp = (response or "").strip()

    # 1. Check strict <think>...</think><answer>...</answer>
    ta_match = THINK_ANSWER_PATTERN.search(resp)
    if ta_match:
        result["has_think_answer"] = True
        result["has_answer_tag"] = True
        result["strict_pred_answer"] = ta_match.group("answer").strip()
        # Check seg in think block
        think_text = ta_match.group("think")
        if SEG_PATTERN.search(think_text):
            result["has_seg"] = True
    else:
        # 2. Fallback: just <answer> tag
        a_match = ANSWER_PATTERN.search(resp)
        if a_match:
            result["has_answer_tag"] = True
            result["strict_pred_answer"] = a_match.group(1).strip()
        # Check think tag separately
        if THINK_PATTERN.search(resp):
            # seg might be in standalone think block
            t_match = THINK_PATTERN.search(resp)
            if t_match and SEG_PATTERN.search(t_match.group(1)):
                result["has_seg"] = True

    # strict_pred in choices?
    if result["strict_pred_answer"] and choices:
        result["answer_in_choices"] = any(
            result["strict_pred_answer"].lower() == c.lower() for c in choices
        )

    # 3. Fallback: match choices in full response
    resp_lower = resp.lower()
    best_match = ""
    for c in choices:
        if c.lower() in resp_lower:
            best_match = c
            break
    result["fallback_pred_answer"] = best_match

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
    parser = argparse.ArgumentParser(
        description="MMAR evaluation with Echo paper Appendix E.2 prompt"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
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
        prompt = build_paper_prompt(q_text, choices)
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
                parsed = parse_response(response, rec["choices"])
                gold = rec["gold_answer"]

                # Strict correct
                strict_correct = parsed["strict_pred_answer"].lower() == gold.lower()
                # Fallback correct
                fallback_correct = parsed["fallback_pred_answer"].lower() == gold.lower()

                stats["processed"] += 1
                stats["strict_correct"] += int(strict_correct)
                stats["fallback_correct"] += int(fallback_correct)
                stats["has_think_answer"] += int(parsed["has_think_answer"])
                stats["has_answer_tag"] += int(parsed["has_answer_tag"])
                stats["has_seg"] += int(parsed["has_seg"])
                stats["answer_in_choices"] += int(parsed["answer_in_choices"])
                if parsed["strict_pred_answer"] and not parsed["answer_in_choices"]:
                    stats["pred_not_in_choices"] += 1

                # Category stats
                cat = rec["category"]
                if cat not in cat_stats:
                    cat_stats[cat] = Counter()
                cat_stats[cat]["processed"] += 1
                cat_stats[cat]["strict_correct"] += int(strict_correct)
                cat_stats[cat]["fallback_correct"] += int(fallback_correct)

                # Modality stats
                mod = rec["modality"]
                if mod not in mod_stats:
                    mod_stats[mod] = Counter()
                mod_stats[mod]["processed"] += 1
                mod_stats[mod]["strict_correct"] += int(strict_correct)
                mod_stats[mod]["fallback_correct"] += int(fallback_correct)

                # Write prediction
                pred_record = {
                    "id": rec["id"],
                    "category": cat,
                    "modality": mod,
                    "question": rec["prompt"],
                    "choices": rec["choices"],
                    "gold_answer": gold,
                    "raw_response": response,
                    "strict_pred_answer": parsed["strict_pred_answer"],
                    "fallback_pred_answer": parsed["fallback_pred_answer"],
                    "strict_correct": strict_correct,
                    "fallback_correct": fallback_correct,
                    "has_think_answer": parsed["has_think_answer"],
                    "has_answer_tag": parsed["has_answer_tag"],
                    "has_seg": parsed["has_seg"],
                    "answer_in_choices": parsed["answer_in_choices"],
                }
                append_jsonl(predictions_path, pred_record)

                if done_count % 10 == 0:
                    s = stats
                    p = max(1, s["processed"])
                    print(f"  [{done_count}/{len(eval_records)}]  "
                          f"strict={s['strict_correct']}/{s['processed']} ({s['strict_correct']/p*100:.1f}%)  "
                          f"fallback={s['fallback_correct']}/{s['processed']} ({s['fallback_correct']/p*100:.1f}%)")

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
                    "strict_pred_answer": "",
                    "fallback_pred_answer": "",
                    "strict_correct": False,
                    "fallback_correct": False,
                    "has_think_answer": False,
                    "has_answer_tag": False,
                    "has_seg": False,
                    "answer_in_choices": False,
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
            "batch_size": args.batch_size,
        },
        "prompt_format": "paper_appendix_E2",
        "timestamp": timestamp,
        "num_samples": stats["processed"],
        "strict_acc": stats["strict_correct"] / p,
        "fallback_acc": stats["fallback_correct"] / p,
        "strict_correct": stats["strict_correct"],
        "fallback_correct": stats["fallback_correct"],
        "total": stats["processed"],
        "has_think_answer": stats["has_think_answer"] / p,
        "has_answer_tag": stats["has_answer_tag"] / p,
        "has_seg": stats["has_seg"] / p,
        "answer_in_choices": stats["answer_in_choices"] / p,
        "pred_not_in_choices": stats["pred_not_in_choices"],
        "failed": stats["failed"],
        "by_category": {
            k: {
                "total": v["processed"],
                "strict_acc": v["strict_correct"] / max(1, v["processed"]),
                "fallback_acc": v["fallback_correct"] / max(1, v["processed"]),
                "strict_correct": v["strict_correct"],
                "fallback_correct": v["fallback_correct"],
            }
            for k, v in sorted(cat_stats.items())
        },
        "by_modality": {
            k: {
                "total": v["processed"],
                "strict_acc": v["strict_correct"] / max(1, v["processed"]),
                "fallback_acc": v["fallback_correct"] / max(1, v["processed"]),
                "strict_correct": v["strict_correct"],
                "fallback_correct": v["fallback_correct"],
            }
            for k, v in sorted(mod_stats.items())
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"MMAR 论文 Prompt 评估完成")
    print(f"  总样本: {stats['processed']}")
    print(f"  strict_acc:  {stats['strict_correct']}/{stats['processed']} ({stats['strict_correct']/p*100:.1f}%)")
    print(f"  fallback_acc:{stats['fallback_correct']}/{stats['processed']} ({stats['fallback_correct']/p*100:.1f}%)")
    print(f"  has_think:   {report['has_think_answer']:.2%}")
    print(f"  has_answer:  {report['has_answer_tag']:.2%}")
    print(f"  has_seg:     {report['has_seg']:.2%}")
    print(f"  ans_in_choices:{report['answer_in_choices']:.2%}")
    print(f"  pred_not_in_choices: {stats['pred_not_in_choices']}")
    print(f"  失败: {stats['failed']}")
    print(f"  报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

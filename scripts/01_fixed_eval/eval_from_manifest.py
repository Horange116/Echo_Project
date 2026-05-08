#!/usr/bin/env python3
"""
基于固定评估集清单 (eval_manifest.jsonl) 的评估脚本。

用法:
  python scripts/eval_from_manifest.py \\
      --model_path /path/to/Qwen2.5-Omni-7B \\
      --adapter_path /path/to/lora/checkpoint \\
      --eval_manifest /path/to/eval_manifest.jsonl \\
      --output_dir /path/to/eval_output \\
      --batch_size 16 \\
      --max_new_tokens 256

输出:
  - eval_report.json:  总体统计 + 按类型统计 + 配置信息
  - predictions.jsonl: 每条样本的详细结果
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
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# 正则: 匹配 <think>...</think><answer>...</answer>
THINK_ANSWER_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.S,
)
# 正则: 匹配 <seg>start, end</seg>
SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")

SAMPLE_RATE = 16000


def read_jsonl(path):
    """逐行读取 JSONL，返回 (line_no, dict) 迭代器。"""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def append_jsonl(path, obj):
    """追加写入 JSONL。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def build_question(item):
    """将 raw question + choices 拼成模型的输入 prompt。"""
    choices_str = json.dumps(item["choices"], ensure_ascii=False)
    return (
        item["question"]
        + " Choose the answer from "
        + choices_str
        + ". Think step-by-step. Refer to the specific audio segments while thinking, "
        + "and indicate the corresponding timestamps with <seg>start, end</seg>. "
        + "Answer in the format of <think>...</think><answer>...</answer>."
    )


def parse_response(response):
    """解析模型输出，提取 think/answer/segment 结构。"""
    result = {
        "has_think_answer": False,
        "has_seg_in_think": False,
        "seg_format_valid": False,
        "answer_block_nonempty": False,
        "answer_text": "",
        "segments": [],
        "fully_structured": False,
    }
    match = THINK_ANSWER_PATTERN.match(response or "")
    if not match:
        return result

    think_text = match.group("think").strip()
    answer_text = match.group("answer").strip()
    seg_matches = SEG_PATTERN.findall(think_text)

    result["has_think_answer"] = True
    result["answer_text"] = answer_text
    result["answer_block_nonempty"] = bool(answer_text)
    if seg_matches:
        result["has_seg_in_think"] = True
        result["seg_format_valid"] = True
        result["segments"] = [[float(start), float(end)] for start, end in seg_matches]

    result["fully_structured"] = all([
        result["has_think_answer"],
        result["has_seg_in_think"],
        result["seg_format_valid"],
        result["answer_block_nonempty"],
    ])
    return result


def load_model_and_processor(model_path, adapter_path=None):
    """加载 Qwen2.5-Omni 基座模型 + 可选 LoRA 适配器。"""
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.base_model.disable_talker()
    model.eval()
    return model, processor


def run_inference(model, processor, audio_path, question, max_new_tokens):
    """单条推理。"""
    audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_data},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=text, audio=audio_data, return_tensors="pt", padding=True)
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
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def run_batch_inference(model, processor, batch_items, max_new_tokens):
    """批量推理。"""
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
        description="基于固定评估集清单的评估脚本"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Qwen2.5-Omni 基座模型路径"
    )
    parser.add_argument(
        "--adapter_path", default=None,
        help="LoRA adapter 路径（可选，不提供则直接用 base model）"
    )
    parser.add_argument(
        "--eval_manifest", required=True,
        help="eval_manifest.jsonl 路径"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="输出目录（内含 eval_report.json + predictions.jsonl）"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="生成最大 token 数 (默认 256)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="推理 batch size (默认 16)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 准备输出目录
    # ------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "eval_report.json"

    # ------------------------------------------------------------
    # 读取 manifest
    # ------------------------------------------------------------
    manifest_items = []
    for line_no, item in read_jsonl(args.eval_manifest):
        # 只保留 audio_path 存在的样本
        ap = item.get("audio_path", "")
        if not ap or not os.path.exists(ap):
            print(f"  跳过 (音频不存在): {item.get('id')} -> {ap}")
            continue
        manifest_items.append(item)

    print(f"评估样本数: {len(manifest_items)}")
    print(f"manifest: {args.eval_manifest}")
    print(f"model: {args.model_path}")
    print(f"adapter: {args.adapter_path or '(无, 直接使用 base model)'}")
    print(f"输出目录: {output_dir}")
    print()

    # ------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------
    model, processor = load_model_and_processor(args.model_path, args.adapter_path)

    # ------------------------------------------------------------
    # 准备 eval records
    # ------------------------------------------------------------
    eval_records = []
    for item in manifest_items:
        qa_type = item.get("type", "unknown")
        question = build_question(item)
        gold_answer = str(item.get("answer", "")).strip()
        eval_records.append({
            "manifest_item": item,
            "qa_type": qa_type,
            "question": question,
            "gold_answer": gold_answer,
        })

    # ------------------------------------------------------------
    # 逐批推理 + 解析
    # ------------------------------------------------------------
    stats = Counter()
    type_stats = {}

    done_count = 0
    for batch_start in range(0, len(eval_records), args.batch_size):
        batch = eval_records[batch_start: batch_start + args.batch_size]
        batch_payload = [
            {"audio_path": rec["manifest_item"]["audio_path"], "question": rec["question"]}
            for rec in batch
        ]

        # 批量推理（失败时回退到单条）
        try:
            responses = run_batch_inference(model, processor, batch_payload, args.max_new_tokens)
        except Exception as batch_error:
            print(f"  batch 推理失败，回退单条: {repr(batch_error)[:200]}")
            responses = []
            for payload in batch_payload:
                try:
                    resp = run_inference(model, processor, payload["audio_path"],
                                         payload["question"], args.max_new_tokens)
                    responses.append(resp)
                except Exception as single_error:
                    responses.append({"__error__": single_error})

        # 解析每条结果
        for rec, response in zip(batch, responses):
            done_count += 1
            item = rec["manifest_item"]
            qa_type = rec["qa_type"]
            gold_answer = rec["gold_answer"]

            try:
                if isinstance(response, dict) and "__error__" in response:
                    raise response["__error__"]

                response = str(response).strip()
                parsed = parse_response(response)
                pred_answer = parsed["answer_text"]
                choices = item.get("choices") or []
                answer_in_choices = pred_answer in choices
                answer_correct = pred_answer == gold_answer

                # 累加统计
                stats["processed"] += 1
                stats["has_think_answer"] += int(parsed["has_think_answer"])
                stats["has_seg"] += int(parsed["has_seg_in_think"])
                stats["fully_structured"] += int(parsed["fully_structured"])
                stats["answer_in_choices"] += int(answer_in_choices)
                stats["answer_correct"] += int(answer_correct)

                # 按类型累加
                if qa_type not in type_stats:
                    type_stats[qa_type] = Counter()
                type_stats[qa_type]["processed"] += 1
                type_stats[qa_type]["fully_structured"] += int(parsed["fully_structured"])
                type_stats[qa_type]["has_seg"] += int(parsed["has_seg_in_think"])
                type_stats[qa_type]["answer_correct"] += int(answer_correct)

                # 写 predictions.jsonl
                pred_record = {
                    "id": item.get("id"),
                    "type": qa_type,
                    "question": item.get("question"),
                    "choices": choices,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "has_think_answer": parsed["has_think_answer"],
                    "has_seg": parsed["has_seg_in_think"],
                    "fully_structured": parsed["fully_structured"],
                    "answer_in_choices": answer_in_choices,
                    "answer_correct": answer_correct,
                    "response": response,
                }
                append_jsonl(predictions_path, pred_record)

                print(f"  [{done_count}/{len(eval_records)}] {item.get('id','?')}  "
                      f"pred={pred_answer}  gold={gold_answer}  "
                      f"struct={parsed['fully_structured']}  correct={answer_correct}")

            except Exception as e:
                stats["failed"] += 1
                pred_record = {
                    "id": item.get("id"),
                    "type": qa_type,
                    "question": item.get("question"),
                    "choices": item.get("choices"),
                    "gold_answer": gold_answer,
                    "pred_answer": "",
                    "has_think_answer": False,
                    "has_seg": False,
                    "fully_structured": False,
                    "answer_in_choices": False,
                    "answer_correct": False,
                    "response": str(response) if isinstance(response, str) else "",
                    "error": repr(e),
                }
                append_jsonl(predictions_path, pred_record)
                print(f"  [{done_count}/{len(eval_records)}] {item.get('id','?')}  ERROR: {repr(e)[:120]}")

    # ------------------------------------------------------------
    # 构建报告
    # ------------------------------------------------------------
    processed = max(1, stats["processed"])
    report = {
        "checkpoint_path": args.model_path,
        "adapter_path": args.adapter_path,
        "eval_manifest_path": args.eval_manifest,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "sample_rate": SAMPLE_RATE,
        },
        "timestamp": timestamp,
        "num_samples_in_manifest": len(manifest_items),
        "stats": dict(stats),
        "rates": {
            "has_think_answer": stats["has_think_answer"] / processed,
            "has_seg": stats["has_seg"] / processed,
            "fully_structured": stats["fully_structured"] / processed,
            "answer_in_choices": stats["answer_in_choices"] / processed,
            "answer_acc": stats["answer_correct"] / processed,
        },
        "type_stats": {k: dict(v) for k, v in sorted(type_stats.items())},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"评估完成")
    print(f"  report: {report_path}")
    print(f"  predictions: {predictions_path}")
    print(f"  有效样本: {stats['processed']}")
    print(f"  失败: {stats.get('failed', 0)}")
    print(f"  has_think_answer: {report['rates']['has_think_answer']:.2%}")
    print(f"  has_seg:          {report['rates']['has_seg']:.2%}")
    print(f"  fully_structured: {report['rates']['fully_structured']:.2%}")
    print(f"  answer_in_choices:{report['rates']['answer_in_choices']:.2%}")
    print(f"  answer_acc:       {report['rates']['answer_acc']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

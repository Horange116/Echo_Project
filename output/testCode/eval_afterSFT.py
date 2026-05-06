# -*- coding: utf-8 -*-
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


BASE_MODEL_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/"

# Change this to your newly trained checkpoint.
ADAPTER_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/testResult/v8-20260505-175434/checkpoint-1498/"

EVAL_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/qa_skeleton.jsonl"
OUTPUT_DIR = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/testResult/eval_afterSFT"

# Training used skeleton indices 0-12999, so evaluation should start after that.
START_INDEX = 13000
MAX_SAMPLES = 500
EVAL_BATCH_SIZE = 16

MAX_NEW_TOKENS = 256
SAMPLE_RATE = 16000

THINK_ANSWER_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.S,
)
SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def build_question(item):
    return (
        item["question"]
        + " Choose the answer from "
        + repr(item["choices"])
        + ". Think step-by-step. Refer to the specific audio segments while thinking, and indicate the corresponding timestamps with <seg>start, end</seg>. "
        + "Answer in the format of <think>...</think><answer>...</answer>."
    )


def parse_response(response):
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


def load_model_and_processor():
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"

    processor = Qwen2_5OmniProcessor.from_pretrained(BASE_MODEL_PATH)
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.base_model.disable_talker()
    model.eval()
    return model, processor


def run_inference(model, processor, audio_path, question):
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

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio_data, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_audio=False,
            speaker=None,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def run_batch_inference(model, processor, batch_items):
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
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        text_list.append(text)

    inputs = processor(text=text_list, audio=audio_list, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_audio=False,
            speaker=None,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def collect_eval_items():
    items = []
    for line_no, item in read_jsonl(EVAL_JSONL):
        if line_no < START_INDEX:
            continue
        if len(items) >= MAX_SAMPLES:
            break
        audio_path = item.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue
        items.append((line_no, item))
    return items


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / ("eval_outputs_%s.jsonl" % timestamp)
    report_json = out_dir / ("eval_report_%s.json" % timestamp)

    items = collect_eval_items()
    print("eval items:", len(items))
    print("adapter:", ADAPTER_PATH)
    print("output:", output_jsonl)

    model, processor = load_model_and_processor()

    stats = Counter()
    type_stats = {}

    eval_records = []
    for line_no, item in items:
        qa_type = item.get("qa_type", "unknown")
        type_stats.setdefault(qa_type, Counter())
        question = build_question(item)
        gold_answer = str(item.get("answer", "")).strip()
        eval_records.append((item, {
            "line_no": line_no,
            "skeleton_id": item.get("skeleton_id"),
            "qa_type": qa_type,
            "audio_path": item.get("audio_path"),
            "question": question,
            "choices": item.get("choices"),
            "gold_answer": gold_answer,
        }))

    done_count = 0
    for batch_start in range(0, len(eval_records), EVAL_BATCH_SIZE):
        batch = eval_records[batch_start:batch_start + EVAL_BATCH_SIZE]
        batch_payload = [
            {"audio_path": item["audio_path"], "question": record["question"]}
            for item, record in batch
        ]

        try:
            responses = run_batch_inference(model, processor, batch_payload)
        except Exception as batch_error:
            print("batch failed, fallback to single inference:", repr(batch_error))
            responses = []
            for item, record in batch:
                try:
                    responses.append(run_inference(model, processor, item["audio_path"], record["question"]))
                except Exception as single_error:
                    responses.append({"__error__": single_error})

        for (item, record), response in zip(batch, responses):
            done_count += 1
            qa_type = item.get("qa_type", "unknown")
            gold_answer = str(item.get("answer", "")).strip()

            try:
                if isinstance(response, dict) and "__error__" in response:
                    raise response["__error__"]

                response = str(response).strip()
                parsed = parse_response(response)
                pred_answer = parsed["answer_text"]
                answer_in_choices = pred_answer in (item.get("choices") or [])
                answer_correct = pred_answer == gold_answer

                record.update({
                    "raw_response": response,
                    "structure_check": parsed,
                    "pred_answer": pred_answer,
                    "answer_in_choices": answer_in_choices,
                    "answer_correct": answer_correct,
                })

                stats["processed"] += 1
                stats["has_think_answer"] += int(parsed["has_think_answer"])
                stats["has_seg"] += int(parsed["has_seg_in_think"])
                stats["fully_structured"] += int(parsed["fully_structured"])
                stats["answer_in_choices"] += int(answer_in_choices)
                stats["answer_correct"] += int(answer_correct)

                type_stats[qa_type]["processed"] += 1
                type_stats[qa_type]["fully_structured"] += int(parsed["fully_structured"])
                type_stats[qa_type]["answer_correct"] += int(answer_correct)

            except Exception as e:
                record.update({
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
                stats["failed"] += 1
                type_stats[qa_type]["failed"] += 1

            append_jsonl(output_jsonl, record)
            print("[%d/%d] %s pred=%s gold=%s structured=%s correct=%s" % (
                done_count,
                len(eval_records),
                item.get("skeleton_id"),
                record.get("pred_answer"),
                gold_answer,
                record.get("structure_check", {}).get("fully_structured"),
                record.get("answer_correct"),
            ))

    processed = max(1, stats["processed"])
    report = {
        "base_model_path": BASE_MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "eval_jsonl": EVAL_JSONL,
        "start_index": START_INDEX,
        "max_samples": MAX_SAMPLES,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "output_jsonl": str(output_jsonl),
        "stats": dict(stats),
        "rates": {
            "fully_structured": stats["fully_structured"] / processed,
            "has_think_answer": stats["has_think_answer"] / processed,
            "has_seg": stats["has_seg"] / processed,
            "answer_in_choices": stats["answer_in_choices"] / processed,
            "answer_acc": stats["answer_correct"] / processed,
        },
        "type_stats": {k: dict(v) for k, v in type_stats.items()},
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

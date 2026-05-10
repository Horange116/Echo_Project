#!/usr/bin/env python3
"""
Audio-interleaved inference with duplicate-segment protection and diagnostics.

Implements Echo paper inference mechanism:
1. Input complete audio + question + choices
2. Model generates text; detect <seg>start,end</seg> to crop & insert
3. IoU-based duplicate detection prevents infinite loops
4. Finalization round forces an answer when interleaved loop stops
5. Per-round and overall diagnostics recorded
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import (
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
    StoppingCriteria, StoppingCriteriaList,
)

SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")
THINK_ANSWER_PATTERN = re.compile(
    r"(?:<think>(?P<think>.*?)</think>\s*)?<answer>(?P<answer>.*?)</answer>",
    re.S,
)
SAMPLE_RATE = 16000


# ── helpers ──

def segments_iou(a_start, a_end, b_start, b_end):
    """IoU of two time segments."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_duplicate_seg(start, end, used_segments, threshold):
    """Check if (start,end) duplicates any previously-used segment via IoU."""
    for seg in used_segments:
        iou = segments_iou(start, end, seg["start"], seg["end"])
        if iou >= threshold:
            return seg, iou
    return None, 0.0


def build_initial_prompt(question, choices):
    """Construct initial prompt asking for seg-timestamped reasoning."""
    choices_str = json.dumps(choices, ensure_ascii=False)
    return (
        question
        + " Choose the answer from "
        + choices_str
        + ". Think step-by-step. Refer to the specific audio segments while thinking, "
        + "and indicate the corresponding timestamps with <seg>start, end</seg>. "
        + "Answer in the format of <think>...</think><answer>...</answer>."
    )


def build_continue_prompt():
    """Prompt after inserting a cropped audio segment."""
    return (
        "I have listened to the audio segment you referenced. "
        "Continue your reasoning and provide the final answer. "
        "Use <seg>start, end</seg> if you need to reference more segments."
    )


def build_finalize_prompt():
    """Prompt for finalization round — no new audio, just answer."""
    return (
        "Now provide the final answer using the information already analyzed. "
        "Answer in <answer>...</answer>."
    )


def parse_segments(text):
    """Extract all <seg>start,end</seg> from text."""
    return [(float(s), float(e)) for s, e in SEG_PATTERN.findall(text)]


def has_answer(text):
    """Check if text contains a complete <answer> tag."""
    return bool(THINK_ANSWER_PATTERN.search(text))


def extract_answer(text):
    """Extract answer from <answer>...</answer>. Returns '' if not found."""
    m = THINK_ANSWER_PATTERN.search(text)
    return m.group("answer").strip() if m else ""


def extract_latest_segments(text, known_count):
    """Return new seg pairs beyond known_count."""
    all_segs = parse_segments(text)
    if len(all_segs) > known_count:
        return all_segs[known_count:]
    return []


def clamp_seg(start, end, duration):
    """Clamp segment to [0, duration]. Returns None if degenerate."""
    start = max(0.0, min(float(start), duration))
    end = max(0.0, min(float(end), duration))
    if start >= end:
        return None
    return (start, end)


def save_segment_audio(audio, sr, start, end, output_dir, round_idx, seg_idx):
    """Crop and save a segment to a WAV file."""
    os.makedirs(output_dir, exist_ok=True)
    start_s = int(start * sr)
    end_s = int(end * sr)
    segment = audio[start_s:end_s]
    path = os.path.join(output_dir, f"round{round_idx}_seg{seg_idx}.wav")
    import soundfile as sf
    sf.write(path, segment, sr)
    return path


# ── model loading ──

def load_model_and_processor(model_path, adapter_path=None):
    """Load Qwen2.5-Omni model and processor. Returns (model, processor)."""
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


# ── inference helpers ──

def build_conversation(prompt, all_generated_text, used_segments, audio_full,
                       sample_rate, audio_list, round_idx, is_finalize,
                       continue_mode="prompt"):
    """Build the conversation messages for the current round.

    continue_mode controls the follow-up user message after seg insertion:
      "prompt"  — audio + "I have listened..." instruction (default)
      "silent"  — pure audio, no text
      "context" — audio + assistant's previous text (no instruction)
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_full},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if all_generated_text.strip():
        conversation.append({
            "role": "assistant",
            "content": all_generated_text.strip(),
        })

    if not is_finalize:
        for idx, seg_info in enumerate(used_segments):
            seg_audio, _ = librosa.load(seg_info["segment_path"], sr=sample_rate)
            audio_list.append(seg_audio)

            if idx == len(used_segments) - 1 and round_idx > 0:
                if continue_mode == "silent":
                    content = [{"type": "audio", "audio": seg_audio}]
                elif continue_mode == "context":
                    # Assistant's previous text + new audio (text before audio)
                    content = [
                        {"type": "text", "text": all_generated_text.strip()},
                        {"type": "audio", "audio": seg_audio},
                    ]
                else:  # "prompt"
                    content = [
                        {"type": "audio", "audio": seg_audio},
                        {"type": "text", "text": build_continue_prompt()},
                    ]
                conversation.append({
                    "role": "user",
                    "content": content,
                })

    return conversation


class SegStoppingCriteria(StoppingCriteria):
    """Stop generation as soon as </seg> or </answer> appears.

    Priority is </seg> (checked first) so interleaved audio insertion
    fires before the model can complete a premature <answer>.
    """
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_length:]
        if len(generated) == 0:
            return False
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        if "</seg>" in text:
            return True
        if "</answer>" in text:
            return True
        return False


def run_generation(model, processor, conversation, audio_list,
                   max_new_tokens, temperature, stop_at_seg=True):
    """Run model generation on a conversation.

    When stop_at_seg is True (default), generation halts as soon as
    </seg> or </answer> is emitted, so the caller can interleave audio.

    Returns (response_text, stop_reason) where stop_reason is one of
    "seg", "answer", "max_tokens_or_eos", or None (when stop_at_seg=False).
    """
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    all_audios = []
    for msg in conversation:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "audio" and "audio" in item:
                    all_audios.append(item["audio"])

    if len(all_audios) == 1:
        inputs = processor(
            text=text, audio=all_audios[0], return_tensors="pt", padding=True
        )
    else:
        inputs = processor(
            text=text, audio=all_audios, return_tensors="pt", padding=True
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "return_audio": False,
        "speaker": None,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    if stop_at_seg:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
            SegStoppingCriteria(processor.tokenizer, prompt_length)
        ])

    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)

    new_tokens = generated[:, prompt_length:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    if stop_at_seg:
        if "</seg>" in response:
            return response, "seg"
        elif "</answer>" in response:
            return response, "answer"
        else:
            return response, "max_tokens_or_eos"
    return response, None


# ── main interleaved loop ──

def run_interleaved(model, processor, audio_path, question, choices,
                    max_rounds=5, max_new_tokens_per_round=128,
                    temperature=0.7, sample_rate=SAMPLE_RATE,
                    tmp_dir="output/interleaved_tmp",
                    duplicate_iou_threshold=0.8,
                    max_duplicate_segments=1,
                    on_duplicate_seg="ignore_continue",
                    finalize_on_stop=True,
                    finalize_max_new_tokens=64,
                    gold_answer=None,
                    continue_mode="prompt"):
    """
    Run audio-interleaved inference with duplicate protection.

    Returns a detailed diagnostic dict (see docstring for format).
    """
    # Validate params
    assert on_duplicate_seg in ("stop", "ignore_continue", "insert_once_continue")

    # Load audio
    audio_full, sr = librosa.load(audio_path, sr=sample_rate)
    duration = librosa.get_duration(path=audio_path)

    prompt = build_initial_prompt(question, choices)

    # State
    all_generated_text = ""
    used_segments = []          # segments actually inserted
    detected_segments = []      # all segments seen (including duplicates)
    duplicate_segments_info = []  # details on duplicate detections
    round_outputs = []
    stop_reason = "max_rounds"
    audio_list = [audio_full]
    seg_count = 0
    duplicate_count = 0
    has_answer_ever = False
    triggered_interleaved = False

    # ── interleaved rounds ──
    for round_idx in range(max_rounds):
        t_start = time.time()
        print(f"\n=== Round {round_idx + 1}/{max_rounds} ===")

        # Count audios before this round
        num_audios_before = len(audio_list) - 1  # exclude full audio

        conversation = build_conversation(
            prompt, all_generated_text, used_segments,
            audio_full, sample_rate, audio_list, round_idx,
            is_finalize=False, continue_mode=continue_mode,
        )

        response, round_stop_reason = run_generation(
            model, processor, conversation, audio_list,
            max_new_tokens_per_round, temperature,
            stop_at_seg=True,
        )

        round_elapsed = time.time() - t_start
        print(f"  生成 ({round_elapsed:.1f}s): {response[:120]}...")

        # ── PRIORITY 1: Extract and process new segs (before answer check) ──
        new_segs = extract_latest_segments(response, seg_count)
        round_duplicate_count = 0
        round_inserted = []

        if new_segs:
            triggered_interleaved = True
            print(f"  检测到 {len(new_segs)} 个新 seg")

        for s, e in new_segs:
            detected_segments.append((s, e))
            clamped = clamp_seg(s, e, duration)
            if clamped is None:
                continue
            start, end = clamped

            # Check duplicate
            dup_seg, iou = is_duplicate_seg(
                start, end, used_segments, duplicate_iou_threshold
            )

            if dup_seg is not None:
                duplicate_count += 1
                round_duplicate_count += 1
                duplicate_segments_info.append({
                    "round": round_idx + 1,
                    "seg": [start, end],
                    "iou": round(iou, 4),
                    "duplicate_of": {"start": dup_seg["start"], "end": dup_seg["end"]},
                })

                if on_duplicate_seg == "stop":
                    print(f"  重复 seg [{start:.2f}, {end:.2f}] (IoU={iou:.2f}) >= threshold, 停止")
                    # Still record the round output
                    round_outputs.append({
                        "round": round_idx + 1,
                        "text": response,
                        "detected_seg_text": f"<seg>{s},{e}</seg>",
                        "parsed_start": start,
                        "parsed_end": end,
                        "stop_reason": "duplicate_seg",
                        "num_audios_before": num_audios_before,
                        "num_audios_after": len(audio_list) - 1,
                        "inserted_audio_paths": [],
                        "duplicate_of_previous": True,
                        "duplicate_iou": round(iou, 4),
                    })

                    if all_generated_text:
                        all_generated_text += " " + response
                    else:
                        all_generated_text = response

                    seg_count += 1
                    stop_reason = "duplicate_seg"
                    break  # break out of seg loop

                elif on_duplicate_seg == "ignore_continue":
                    print(f"  重复 seg [{start:.2f}, {end:.2f}] (IoU={iou:.2f}), 忽略继续")
                    seg_count += 1
                    continue

                elif on_duplicate_seg == "insert_once_continue":
                    # Check if this exact seg was already inserted
                    already_inserted = any(
                        abs(seg["start"] - start) < 0.01 and abs(seg["end"] - end) < 0.01
                        for seg in used_segments
                    )
                    if already_inserted:
                        print(f"  重复 seg [{start:.2f}, {end:.2f}], 已插入过, 跳过")
                        seg_count += 1
                        continue
                    # fall through to insert

            # Insert segment
            seg_path = save_segment_audio(
                audio_full, sr, start, end, tmp_dir,
                round_idx + 1, len(used_segments) + 1
            )
            used_segments.append({
                "round": round_idx + 1,
                "start": start,
                "end": end,
                "segment_path": seg_path,
            })
            round_inserted.append(seg_path)
            seg_count += 1
            print(f"  插入 seg: [{start:.2f}, {end:.2f}] -> {seg_path}")

        # Check if we broke due to duplicate_seg stop
        if stop_reason == "duplicate_seg":
            break

        # Update generated text
        if all_generated_text:
            all_generated_text += " " + response
        else:
            all_generated_text = response

        num_audios_after = len(audio_list) - 1

        if new_segs:
            # Segs were processed — record and continue (skip answer check)
            round_outputs.append({
                "round": round_idx + 1,
                "text": response,
                "detected_seg_text": str([(s, e) for s, e in new_segs]) if new_segs else None,
                "parsed_start": round_inserted[0] if round_inserted else None,
                "parsed_end": round_inserted[-1] if round_inserted else None,
                "stop_reason": "continue",
                "num_audios_before": num_audios_before,
                "num_audios_after": num_audios_after,
                "inserted_audio_paths": round_inserted,
                "duplicate_of_previous": round_duplicate_count > 0,
                "duplicate_iou": duplicate_segments_info[-1]["iou"] if duplicate_segments_info else None,
            })
            continue  # ← answer check skipped when segs were found

        # ── DETECT: segs in response but all already processed ──
        all_segs_this_round = parse_segments(response)
        if all_segs_this_round:
            for s, e in all_segs_this_round:
                duplicate_segments_info.append({
                    "round": round_idx + 1,
                    "seg": [s, e],
                    "iou": 1.0,
                    "duplicate_of": {"start": s, "end": e},
                })
            # Append text and continue — re-referencing evidence during reasoning is normal
            if all_generated_text:
                all_generated_text += " " + response
            else:
                all_generated_text = response
            round_outputs.append({
                "round": round_idx + 1,
                "text": response,
                "detected_seg_text": str([(s, e) for s, e in all_segs_this_round]),
                "parsed_start": None,
                "parsed_end": None,
                "stop_reason": "continue",
                "num_audios_before": num_audios_before,
                "num_audios_after": num_audios_after,
                "inserted_audio_paths": [],
                "duplicate_of_previous": True,
                "duplicate_iou": 1.0,
            })
            print(f"  重复 seg (已处理过，继续推理)")
            continue

        # ── PRIORITY 2: No segs — check for answer ──
        if has_answer(response) or round_stop_reason == "answer":
            has_answer_ever = True
            round_outputs.append({
                "round": round_idx + 1,
                "text": response,
                "detected_seg_text": None,
                "parsed_start": None,
                "parsed_end": None,
                "stop_reason": "has_answer",
                "num_audios_before": num_audios_before,
                "num_audios_after": len(audio_list) - 1,
                "inserted_audio_paths": [],
                "duplicate_of_previous": False,
                "duplicate_iou": None,
            })
            stop_reason = "has_answer"
            print(f"  检测到 </answer>，结束。")
            break

        # ── PRIORITY 3: No segs, no answer ──
        if not response:
            round_outputs.append({
                "round": round_idx + 1,
                "text": response,
                "detected_seg_text": None,
                "parsed_start": None,
                "parsed_end": None,
                "stop_reason": "empty_response",
                "num_audios_before": num_audios_before,
                "num_audios_after": len(audio_list) - 1,
                "inserted_audio_paths": [],
                "duplicate_of_previous": False,
                "duplicate_iou": None,
            })
            stop_reason = "empty_response"
            print(f"  空响应，结束。")
            break

        # Has text but no seg or answer — record and continue
        round_outputs.append({
            "round": round_idx + 1,
            "text": response,
            "detected_seg_text": None,
            "parsed_start": None,
            "parsed_end": None,
            "stop_reason": "continue_no_seg",
            "num_audios_before": num_audios_before,
            "num_audios_after": num_audios_after,
            "inserted_audio_paths": [],
            "duplicate_of_previous": False,
            "duplicate_iou": None,
        })

        if round_idx >= max_rounds - 1:
            stop_reason = "max_rounds"

    # ── finalization round ──
    if finalize_on_stop and stop_reason in ("duplicate_seg", "max_rounds") and not has_answer_ever:
        print(f"\n=== Finalization round (stop_reason={stop_reason}) ===")
        t_start = time.time()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_full},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if all_generated_text.strip():
            conversation.append({
                "role": "assistant",
                "content": all_generated_text.strip(),
            })
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": build_finalize_prompt()}],
        })

        response, _ = run_generation(
            model, processor, conversation, audio_list,
            finalize_max_new_tokens, temperature,
            stop_at_seg=False,
        )

        round_elapsed = time.time() - t_start
        print(f"  Finalization 生成 ({round_elapsed:.1f}s): {response[:120]}...")

        if has_answer(response):
            has_answer_ever = True

        if all_generated_text:
            all_generated_text += " " + response
        else:
            all_generated_text = response

        round_outputs.append({
            "round": "finalize",
            "text": response,
            "detected_seg_text": None,
            "parsed_start": None,
            "parsed_end": None,
            "stop_reason": f"finalize_after_{stop_reason}",
            "num_audios_before": len(audio_list) - 1,
            "num_audios_after": len(audio_list) - 1,
            "inserted_audio_paths": [],
            "duplicate_of_previous": False,
            "duplicate_iou": None,
        })

    # ── extract final answer ──
    pred_answer = extract_answer(all_generated_text)
    has_final_answer = bool(pred_answer)

    answer_correct = None
    if gold_answer is not None and has_final_answer:
        answer_correct = (pred_answer == gold_answer)

    # ── build result ──
    result = {
        "question": question,
        "choices": choices,
        "total_rounds": len([ro for ro in round_outputs if isinstance(ro["round"], int)]),
        "triggered_interleaved": triggered_interleaved,
        "num_detected_segments": len(detected_segments),
        "num_inserted_segments": len(used_segments),
        "num_duplicate_segments": duplicate_count,
        "duplicate_segments": duplicate_segments_info,
        "stop_reason": stop_reason,
        "has_final_answer": has_final_answer,
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "answer_correct": answer_correct,
        "final_response": all_generated_text,
        "used_segments": [
            {k: v for k, v in seg.items() if k != "segment_path"}
            for seg in used_segments
        ],
        "used_segment_paths": [seg["segment_path"] for seg in used_segments],
        "round_outputs": round_outputs,
        "parse_errors": None,
    }
    return result


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Audio-interleaved inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--choices", required=True,
                        help='JSON array, e.g. \'["A","B"]\'')
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--max_new_tokens_per_round", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--tmp_dir", default="output/interleaved_tmp")

    # New parameters
    parser.add_argument("--duplicate_iou_threshold", type=float, default=0.8)
    parser.add_argument("--max_duplicate_segments", type=int, default=1)
    parser.add_argument("--on_duplicate_seg", default="ignore_continue",
                        choices=["stop", "ignore_continue", "insert_once_continue"])
    parser.add_argument("--finalize_on_stop", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--finalize_max_new_tokens", type=int, default=64)
    parser.add_argument("--gold_answer", default=None)
    parser.add_argument("--continue_mode", default="prompt",
                        choices=["prompt", "silent", "context"],
                        help="Continue mode for seg insertion rounds")

    args = parser.parse_args()

    choices = json.loads(args.choices)

    if not os.path.exists(args.audio_path):
        print(f"错误: 音频文件不存在 {args.audio_path}")
        sys.exit(1)

    print(f"加载模型: {args.model_path}")
    model, processor = load_model_and_processor(args.model_path, args.adapter_path)
    print(f"模型加载完成，device: {model.device}")

    result = run_interleaved(
        model, processor,
        audio_path=args.audio_path,
        question=args.question,
        choices=choices,
        max_rounds=args.max_rounds,
        max_new_tokens_per_round=args.max_new_tokens_per_round,
        temperature=args.temperature,
        sample_rate=args.sample_rate,
        tmp_dir=args.tmp_dir,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        max_duplicate_segments=args.max_duplicate_segments,
        on_duplicate_seg=args.on_duplicate_seg,
        finalize_on_stop=args.finalize_on_stop,
        finalize_max_new_tokens=args.finalize_max_new_tokens,
        gold_answer=args.gold_answer,
        continue_mode=args.continue_mode,
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果写入: {args.output_json}")
    print(f"  stop_reason:     {result['stop_reason']}")
    print(f"  total_rounds:    {result['total_rounds']}")
    print(f"  triggered_inter: {result['triggered_interleaved']}")
    print(f"  inserted_segs:   {result['num_inserted_segments']}")
    print(f"  duplicate_segs:  {result['num_duplicate_segments']}")
    print(f"  has_final_answer:{result['has_final_answer']}")
    print(f"  pred_answer:     {result['pred_answer']}")
    print(f"  gold_answer:     {result['gold_answer']}")
    print(f"  answer_correct:  {result['answer_correct']}")


if __name__ == "__main__":
    main()

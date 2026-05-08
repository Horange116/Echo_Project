#!/usr/bin/env python3
"""
最小版 audio-interleaved inference。

实现 Echo 论文推理机制：
1. 输入完整 audio + question + choices
2. 模型生成文本
3. 检测到 <seg>start, end</seg> 时暂停
4. 裁剪音频片段插入上下文
5. 继续生成，循环直到出现 </answer> 或达到 max_rounds

用法:
  python scripts/interleaved_infer.py \\
      --model_path /path/to/Qwen2.5-Omni-7B \\
      --adapter_path /path/to/lora/checkpoint \\
      --audio_path /path/to/audio.wav \\
      --question "At what percentage does the music start?" \\
      --choices '["0%", "50%", "100%"]' \\
      --output_json output/interleaved_result.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")
THINK_ANSWER_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    re.S,
)
SAMPLE_RATE = 16000


class SegStoppingCriteria(StoppingCriteria):
    """在生成到第一个完整 </seg> 或 </answer> 时停止。"""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_reason = None  # "seg", "answer", None

    def __call__(self, input_ids, scores, **kwargs):
        new_ids = input_ids[0][self.prompt_length:]
        if len(new_ids) == 0:
            return False
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # answer 优先（如果 model 直接给出答案不带 seg）
        if "</answer>" in text:
            self.stop_reason = "answer"
            return True

        # 检测到完整 </seg> 就暂停
        if "</seg>" in text:
            self.stop_reason = "seg"
            return True

        return False


def build_initial_prompt(question, choices):
    """构造初始 prompt，要求模型输出 seg 和 answer 格式。"""
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
    """插入音频片段后的继续推理 prompt。"""
    return (
        "I have listened to the audio segment you referenced. "
        "Continue your reasoning and provide the final answer. "
        "Use <seg>start, end</seg> if you need to reference more segments."
    )


def parse_segments(text):
    """从文本中提取所有 <seg>start, end</seg>。"""
    return [(float(s), float(e)) for s, e in SEG_PATTERN.findall(text)]


def has_answer(text):
    """检查是否已包含完整 answer。"""
    return bool(THINK_ANSWER_PATTERN.search(text))


def extract_latest_segments(text, known_count):
    """提取文本中超出 known_count 的新 seg 对。"""
    all_segs = parse_segments(text)
    if len(all_segs) > known_count:
        return all_segs[known_count:]
    return []


def clamp_seg(start, end, duration):
    """将 seg 时间 clamp 到 [0, duration]。"""
    start = max(0.0, min(float(start), duration))
    end = max(0.0, min(float(end), duration))
    if start >= end:
        return None
    return (start, end)


def save_segment_audio(audio, sr, start, end, output_dir, round_idx, seg_idx):
    """裁剪音频片段并保存。"""
    os.makedirs(output_dir, exist_ok=True)
    start_s = int(start * sr)
    end_s = int(end * sr)
    segment = audio[start_s:end_s]
    path = os.path.join(output_dir, f"round{round_idx}_seg{seg_idx}.wav")
    import soundfile as sf
    sf.write(path, segment, sr)
    return path


def load_model_and_processor(model_path, adapter_path=None):
    """加载模型和 processor。"""
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


def run_interleaved(model, processor, audio_path, question, choices,
                    max_rounds=5, max_new_tokens_per_round=128,
                    temperature=0.7, sample_rate=SAMPLE_RATE, tmp_dir="output/interleaved_tmp"):
    """执行 audio-interleaved 推理（Echo 机制）。

    每轮生成时通过 SegStoppingCriteria 在第一个完整 </seg> 处立即暂停：
    1. 解析 <seg>start, end</seg>（不论是否在 <think> 内部）
    2. 从原始音频裁剪对应片段
    3. 将裁剪的音频插入下一轮上下文中
    4. 继续生成直到出现 </answer> 或达到 max_rounds
    """
    # 加载完整音频
    audio_full, sr = librosa.load(audio_path, sr=sample_rate)
    duration = librosa.get_duration(path=audio_path)

    # 初始 prompt
    prompt = build_initial_prompt(question, choices)

    # 状态
    all_generated_text = ""
    used_segments = []  # [{"round": ..., "start": ..., "end": ..., "segment_path": ...}]
    round_outputs = []  # 每轮完整生成文本
    round_debug = []    # 每轮 debug 信息
    parse_errors = []

    for round_idx in range(max_rounds):
        print(f"\n=== Round {round_idx + 1}/{max_rounds} ===")

        # ---- 构造本轮对话 ----
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_full},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # assistant 历史
        if all_generated_text.strip():
            conversation.append({
                "role": "assistant",
                "content": all_generated_text.strip(),
            })

        # 最新一个裁剪音频片段 + 继续推理 prompt
        if used_segments:
            last_seg = used_segments[-1]
            seg_audio, _ = librosa.load(last_seg["segment_path"], sr=sample_rate)
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio": seg_audio},
                    {"type": "text", "text": build_continue_prompt()},
                ],
            })

        # ---- 编码 ----
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

        num_audios_before = len(all_audios)

        if len(all_audios) == 1:
            inputs = processor(
                text=text, audio=all_audios[0], return_tensors="pt", padding=True
            )
        else:
            inputs = processor(
                text=text, audio=all_audios, return_tensors="pt", padding=True
            )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # ---- 使用 SegStoppingCriteria 生成 ----
        prompt_length = inputs["input_ids"].shape[1]
        seg_stop = SegStoppingCriteria(processor.tokenizer, prompt_length)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens_per_round,
            "return_audio": False,
            "speaker": None,
            "stopping_criteria": StoppingCriteriaList([seg_stop]),
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)

        new_tokens = generated[:, prompt_length:]
        response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        round_outputs.append(response)

        # 判断 stop reason
        stop_reason = seg_stop.stop_reason
        if stop_reason is None:
            # StoppingCriteria 没触发 → EOS 或 max_tokens
            if has_answer(response):
                stop_reason = "answer"
            elif not response:
                stop_reason = "eos"
            else:
                stop_reason = "max_tokens"

        print(f"  生成 ({stop_reason}): {response[:200]}...")

        # 本轮 debug 信息（先填公共字段，seg 字段在下面补）
        round_info = {
            "round": round_idx + 1,
            "round_stop_reason": stop_reason,
            "detected_seg_text": None,
            "parsed_start": None,
            "parsed_end": None,
            "clipped_segment_path": None,
            "clipped_segment_duration": None,
            "insert_success": False,
            "num_audios_before": num_audios_before,
            "num_audios_after": num_audios_before,
            "inserted_audio_paths": None,
            "continued_after_insert": False,
            "generated_text_preview": response[:300],
        }

        # 追加到总文本
        all_generated_text += (" " + response) if all_generated_text else response

        # ============================================================
        #  分支 1: seg 触发 → 裁剪音频 → 插入上下文 → 下一轮
        # ============================================================
        if stop_reason == "seg":
            segs = parse_segments(response)
            if segs:
                s, e = segs[0]  # 每轮只处理第一个 seg
                clamped = clamp_seg(s, e, duration)
                if clamped is not None:
                    start, end = clamped
                    seg_path = save_segment_audio(
                        audio_full, sr, start, end, tmp_dir,
                        round_idx + 1, len(used_segments) + 1
                    )
                    seg_duration = end - start

                    used_segments.append({
                        "round": round_idx + 1,
                        "start": start,
                        "end": end,
                        "segment_path": seg_path,
                    })

                    round_info["detected_seg_text"] = f"<seg>{s}, {e}</seg>"
                    round_info["parsed_start"] = start
                    round_info["parsed_end"] = end
                    round_info["clipped_segment_path"] = seg_path
                    round_info["clipped_segment_duration"] = round(seg_duration, 3)
                    round_info["insert_success"] = True
                    round_info["num_audios_after"] = num_audios_before + 1
                    round_info["inserted_audio_paths"] = [seg_path]

                    if round_idx < max_rounds - 1:
                        round_info["continued_after_insert"] = True

                    print(f"  检测到 seg: [{start:.2f}, {end:.2f}] -> {seg_path}"
                          f"  (dur={seg_duration:.2f}s)")
                else:
                    parse_errors.append(
                        f"round{round_idx}: clamp failed ({s}, {e}), "
                        f"duration={duration}"
                    )
                    print(f"  警告: seg ({s}, {e}) 超出范围，跳过")
                    round_debug.append(round_info)
                    break
            else:
                # 理论上不会发生（stop_reason=seg 但 parse 不到）
                print(f"  警告: stop_reason=seg 但 parse_segments 返回空")
                round_debug.append(round_info)
                break

            round_debug.append(round_info)

            if round_idx < max_rounds - 1:
                continue
            else:
                break

        # ============================================================
        #  分支 2: answer 触发 → 结束
        # ============================================================
        if stop_reason == "answer":
            round_info["inserted_audio_paths"] = (
                [s["segment_path"] for s in used_segments] if used_segments else []
            )
            round_debug.append(round_info)
            print(f"  检测到 </answer>，结束。")
            break

        # ============================================================
        #  分支 3: 其他（max_tokens / eos）
        # ============================================================
        if response:
            print(f"  未检测到 seg 或 answer ({stop_reason})，"
                  f"{'继续' if round_idx < max_rounds - 1 else '结束'}。")
        else:
            print(f"  空响应，结束。")

        round_info["inserted_audio_paths"] = (
            [s["segment_path"] for s in used_segments] if used_segments else []
        )
        round_debug.append(round_info)

        if not response:
            break

    # ---- 从最终文本中提取 answer ----
    final_answer = ""
    answer_match = THINK_ANSWER_PATTERN.search(all_generated_text)
    if answer_match:
        final_answer = answer_match.group("answer").strip()

    result = {
        "question": question,
        "choices": choices,
        "final_response": all_generated_text,
        "final_answer": final_answer,
        "used_segments": [
            {k: v for k, v in seg.items() if k != "segment_path"}
            for seg in used_segments
        ],
        "used_segment_paths": [seg["segment_path"] for seg in used_segments],
        "round_outputs": round_outputs,
        "round_debug": round_debug,
        "num_rounds": len(round_outputs),
        "parse_errors": parse_errors if parse_errors else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="最小版 audio-interleaved inference"
    )
    parser.add_argument("--model_path", required=True, help="Qwen2.5-Omni 基座模型路径")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter 路径（可选）")
    parser.add_argument("--audio_path", required=True, help="输入音频路径")
    parser.add_argument("--question", required=True, help="问题文本")
    parser.add_argument("--choices", required=True, help="选项 JSON 数组，如 '[\"A\",\"B\"]'")
    parser.add_argument("--output_json", required=True, help="输出结果 JSON 路径")
    parser.add_argument("--max_rounds", type=int, default=5, help="最大推理轮次")
    parser.add_argument("--max_new_tokens_per_round", type=int, default=128,
                        help="每轮最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--sample_rate", type=int, default=16000, help="音频采样率")
    parser.add_argument("--tmp_dir", default="output/interleaved_tmp",
                        help="临时音频片段目录")
    args = parser.parse_args()

    # 解析 choices
    try:
        choices = json.loads(args.choices)
    except (json.JSONDecodeError, ValueError):
        print(f"错误: choices 格式无效 {args.choices}")
        sys.exit(1)

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
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果写入: {args.output_json}")
    print(f"  推理轮次: {result['num_rounds']}")
    print(f"  引用段数: {len(result['used_segments'])}")
    print(f"  最终答案: {result['final_answer']}")
    print(f"  解析错误: {result.get('parse_errors')}")
    for rd in result.get('round_debug', []):
        r = rd['round']
        reason = rd['round_stop_reason']
        ins = '✓' if rd['insert_success'] else '✗'
        cont = '✓' if rd['continued_after_insert'] else '✗'
        print(f"  Round {r}: stop={reason} insert={ins} continue={cont} "
              f"audios={rd['num_audios_before']}->{rd['num_audios_after']} "
              f"seg={rd['detected_seg_text'] or '—'}")


if __name__ == "__main__":
    main()

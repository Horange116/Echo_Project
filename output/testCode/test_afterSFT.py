# -*- coding: utf-8 -*-
import os
import re
import json
from datetime import datetime

import librosa
import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

BASE_MODEL_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/"
ADAPTER_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/test_SFT_10_20260426_154454/checkpoint-1"
AUDIO_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/mnt/bn/wdq-base1/data/ALMs/EAQA/audios/AudioSet/audio_72.wav"
OUTPUT_DIR = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/test_afterSFT"

QUESTION = """How long after the man finishes speaking at approximately 4.8 seconds does the next stir event start?
Choose the answer from ['0.2 seconds', '0.5 seconds', '0.8 seconds', '1.0 second']. Think step-by-step. Refer to the
specific audio segments while thinking, and indicate the corresponding timestamps with <seg>start, end</seg>. Answer in
the format of <think>...</think><answer>...</answer>."""

THINK_ANSWER_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.S,
)
SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")


def validate_response_structure(response: str):
    result = {
        "has_think_answer": False,
        "has_seg_in_think": False,
        "seg_format_valid": False,
        "answer_block_nonempty": False,
        "think_text": "",
        "answer_text": "",
        "segments": [],
        "fully_structured": False,
    }

    match = THINK_ANSWER_PATTERN.match(response)
    if not match:
        return result

    think_text = match.group("think").strip()
    answer_text = match.group("answer").strip()
    seg_matches = SEG_PATTERN.findall(think_text)

    result["has_think_answer"] = True
    result["think_text"] = think_text
    result["answer_text"] = answer_text
    result["answer_block_nonempty"] = bool(answer_text)

    if seg_matches:
        result["has_seg_in_think"] = True
        result["seg_format_valid"] = True
        result["segments"] = [[float(start), float(end)] for start, end in seg_matches]

    result["fully_structured"] = all(
        [
            result["has_think_answer"],
            result["has_seg_in_think"],
            result["seg_format_valid"],
            result["answer_block_nonempty"],
        ]
    )
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


def run_single_inference(model, processor):
    audio_data, _ = librosa.load(AUDIO_PATH, sr=16000)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_data},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio_data, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=256, return_audio=False, speaker=None)

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return response


def main():
    model, processor = load_model_and_processor()
    response = run_single_inference(model, processor)
    result = validate_response_structure(response)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"single_infer_{timestamp}.json")
    payload = {
        "base_model_path": BASE_MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "audio_path": AUDIO_PATH,
        "question": QUESTION,
        "raw_response": response,
        "structure_check": result,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("raw_response:")
    print(response)
    print()
    print("structure_check:")
    print(f"has_think_answer: {result['has_think_answer']}")
    print(f"has_seg_in_think: {result['has_seg_in_think']}")
    print(f"seg_format_valid: {result['seg_format_valid']}")
    print(f"answer_block_nonempty: {result['answer_block_nonempty']}")
    print(f"fully_structured: {result['fully_structured']}")
    print(f"segments: {result['segments']}")
    print(f"outfile: {output_path}")


if __name__ == "__main__":
    main()

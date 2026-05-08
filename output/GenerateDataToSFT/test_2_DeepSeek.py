# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path

from openai import OpenAI


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/metadata_with_qwen_audio_info.jsonl"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-HUnMwzbL0kiac1TnUkuk6BunochazpWF32mgTHM5nGDqeaoo")
BASE_URL = "https://yinli.one/v1"
MODEL_NAME = "deepseek-r1"

REQUEST_TIMEOUT = 180.0


PROMPT_TEMPLATE = """You are given simulated information about acoustic events in an audio clip, including:

A1: A description of the audio.
A2: Information about speech (only if present), including possible transcript, emotion, speaker gender, and spoken language.
A3: Information about music (only if present), including genre and instruments.
A4: A list of key audio segments where major sound events occur (with start and end timestamps).
A5: The duration of the audio.
These are used to simulate your understanding as if you had listened to the actual audio. You MUST treat them as your internal interpretation of "the audio", and you MUST NOT reference A1-A5, metadata, labels, annotations, or "the audio description" explicitly in your output.

Your goal is to generate challenging question-answer-chain-of-thought (CoT) triplets in the following valid JSON format:
{
  "question": "A question that requires listening to the audio to answer accurately.",
  "choices": ["a", "b", "c", "d"],
  "answer": "the correct answer, exactly matching one item in choices",
  "cot": "<think>...your reasoning here...</think><answer>...</answer>"
}

REQUIREMENTS:
1. Question Design:
The question must be answerable only by listening to the audio.
It should require non-trivial inference, going beyond surface-level perception.
The question may either involve explicit temporal framing (e.g., asking what happened first, last, or at the same time) or be temporally neutral (e.g., asking about a specific sound, event, or emotion).
However, the answer must require temporal reasoning--such as identifying the order of events, comparing the lengths of different segments, or detecting overlaps in time.

2. Multiple Choice Options:
The correct answer must be unambiguously supported by the audio.
The incorrect answers must be plausible, but clearly incorrect when grounded in the audio.
The "answer" field must exactly match one item in "choices".
The final answer inside <answer>...</answer> must exactly match the "answer" field.

3. Chain-of-Thought (CoT):
The reasoning process must be natural, fluent, and well-structured. Avoid using bullet points or numbering.
Each time a specific piece of evidence from the audio is discussed, the relevant audio segment must be explicitly referenced first, using the format <seg>start_second, end_second</seg> (e.g., <seg>3.3, 5.7</seg>).
Each CoT must include at least one segment reference, unless the provided segments all span the entire audio duration.

4. Use of Simulated Inputs (A1-A5):
Treat A1-A5 as your internal hearing/understanding of the audio.
Refer to any sounds or events by saying they occurred "in the audio". You MUST NOT refer to A1-A5, metadata, labels, annotations, or "the audio description" as external annotations.

A1: __DESCRIPTION__
A2: __SPEECH__
A3: __MUSIC__
A4: __SEGMENTS__
A5: __DURATION__

Ignore the task above. Return only: ok.
"""


def read_first_item(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError("No valid line found in input jsonl.")


def format_segments_for_prompt(events):
    strong_events = []
    for ev in events or []:
        strong_events.append({
            "time_range": "%.3fs - %.3fs" % (float(ev["start"]), float(ev["end"])),
            "label": str(ev["label"]),
        })
    return json.dumps({"strong_events": strong_events}, ensure_ascii=False)


def build_prompt(item):
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("__DESCRIPTION__", str(item.get("a1_description", "")))
    prompt = prompt.replace("__SPEECH__", str(item.get("a2_speech", "")))
    prompt = prompt.replace("__MUSIC__", str(item.get("a3_music", "")))
    prompt = prompt.replace("__SEGMENTS__", format_segments_for_prompt(item.get("events", [])))
    prompt = prompt.replace("__DURATION__", str(item.get("duration", "")))
    return prompt


def main():
    if not API_KEY:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable.")

    item = read_first_item(INPUT_JSONL)
    prompt = build_prompt(item)

    print("segment_id:", item.get("segment_id"))
    print("prompt_chars:", len(prompt))

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )

    start = time.time()
    print("calling api...", flush=True)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
        timeout=REQUEST_TIMEOUT,
    )

    elapsed = time.time() - start

    print("api returned", flush=True)
    print("elapsed_seconds:", round(elapsed, 2))
    print("content:", resp.choices[0].message.content)


if __name__ == "__main__":
    main()

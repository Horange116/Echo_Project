# -*- coding: utf-8 -*-
import os
import json
import time
from openai import OpenAI


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/metadata_with_qwen_audio_info.jsonl"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-HUnMwzbL0kiac1TnUkuk6BunochazpWF32mgTHM5nGDqeaoo")
BASE_URL = "https://yinli.one/v1"
MODEL_NAME = "deepseek-r1"

REQUEST_TIMEOUT = 600.0


def read_first_item(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError("empty input")


def format_segments_for_prompt(events):
    strong_events = []
    for ev in events or []:
        strong_events.append({
            "time_range": "%.3fs - %.3fs" % (float(ev["start"]), float(ev["end"])),
            "label": str(ev["label"]),
        })
    return json.dumps({"strong_events": strong_events}, ensure_ascii=False)


def build_min_long_prompt(item):
    return """You are given simulated information about acoustic events in an audio clip.

A1: %s
A2: %s
A3: %s
A4: %s
A5: %s

Ignore everything above and return only: ok.
""" % (
        str(item.get("a1_description", "")),
        str(item.get("a2_speech", "")),
        str(item.get("a3_music", "")),
        format_segments_for_prompt(item.get("events", [])),
        str(item.get("duration", "")),
    )


def main():
    if not API_KEY:
        raise ValueError("Please set DEEPSEEK_API_KEY.")

    item = read_first_item(INPUT_JSONL)
    prompt = build_min_long_prompt(item)

    print("segment_id:", item.get("segment_id"))
    print("prompt_chars:", len(prompt))

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )

    start = time.time()
    print("calling stream...", flush=True)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )

    first_time = None
    chunks = []

    for chunk in stream:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            if first_time is None:
                first_time = time.time()
                print("first_chunk_seconds:", round(first_time - start, 2), flush=True)
            print(text, end="", flush=True)
            chunks.append(text)

    print()
    print("total_seconds:", round(time.time() - start, 2))
    print("content:", "".join(chunks))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/metadata_with_qwen_audio_info.jsonl"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/deepseek_qa_cot_raw.jsonl"
ERROR_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/deepseek_qa_cot_errors.jsonl"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-HUnMwzbL0kiac1TnUkuk6BunochazpWF32mgTHM5nGDqeaoo")
BASE_URL = "https://yinli.one/v1"
MODEL_NAME = "deepseek-r1"

MAX_ITEMS = 100
CONCURRENCY = 3
MAX_RETRIES = 1
REQUEST_TIMEOUT = 360.0
SLEEP_SECONDS = 0.2

MAX_OUTPUT_TOKENS = 700
MAX_EVENTS_IN_PROMPT = 14
MAX_A1_CHARS = 500
MAX_A2_CHARS = 300
MAX_A3_CHARS = 300


PROMPT_TEMPLATE = """You are given simulated information about acoustic events in an audio clip, including:

A1: A description of the audio.
A2: Information about speech, only if speech is present.
A3: Information about music, only if music is present.
A4: Key audio segments where major sound events occur, with start and end timestamps.
A5: The duration of the audio.

Treat A1-A5 as your internal understanding of the audio. Do not mention A1-A5, metadata, labels, annotations, or "the audio description" in your output.

Generate one challenging audio question-answer-chain-of-thought triplet as one valid JSON object:
{
  "question": "A question that requires listening to the audio to answer accurately.",
  "choices": ["a", "b", "c", "d"],
  "answer": "the correct answer, exactly matching one item in choices",
  "cot": "<think>...reasoning...</think><answer>...</answer>"
}

Requirements:
1. The question must be answerable using audio alone.
2. The answer must require temporal reasoning, such as order, duration comparison, gap, or overlap.
3. The correct answer must be unambiguous and exactly match one choice.
4. The final answer inside <answer>...</answer> must exactly match the answer field.
5. Each concrete audio evidence in cot must start with a segment tag like <seg>3.3, 5.7</seg>.
6. Output only the JSON object. Do not add explanations before or after it.

A1: __DESCRIPTION__
A2: __SPEECH__
A3: __MUSIC__
A4: __SEGMENTS__
A5: __DURATION__
"""


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_done_ids(path):
    done = set()
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = obj.get("segment_id")
                if sid:
                    done.add(str(sid))
            except Exception:
                pass
    return done


def compact_text(text, max_chars):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def select_events(events, duration, max_events):
    clean = []
    duration = float(duration or 10.0)
    for ev in events or []:
        try:
            start = float(ev["start"])
            end = float(ev["end"])
            label = str(ev["label"])
        except Exception:
            continue
        if end <= start:
            continue
        is_full = start <= 0.05 and abs(end - duration) <= 0.05
        clean.append({
            "start": start,
            "end": end,
            "label": label,
            "duration": end - start,
            "is_full": is_full,
        })

    clean = sorted(clean, key=lambda x: (x["start"], x["end"], x["label"]))
    non_full = [x for x in clean if not x["is_full"]]
    full = [x for x in clean if x["is_full"]]

    selected = non_full[:max_events]
    if len(selected) < max_events:
        selected += full[:max_events - len(selected)]
    return sorted(selected, key=lambda x: (x["start"], x["end"], x["label"]))


def format_segments_for_prompt(events, duration):
    selected = select_events(events, duration, MAX_EVENTS_IN_PROMPT)
    strong_events = []
    for ev in selected:
        strong_events.append({
            "time_range": "%.3fs - %.3fs" % (ev["start"], ev["end"]),
            "label": ev["label"],
        })
    return json.dumps({"strong_events": strong_events}, ensure_ascii=False)


def build_prompt(item):
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("__DESCRIPTION__", compact_text(item.get("a1_description", ""), MAX_A1_CHARS))
    prompt = prompt.replace("__SPEECH__", compact_text(item.get("a2_speech", ""), MAX_A2_CHARS))
    prompt = prompt.replace("__MUSIC__", compact_text(item.get("a3_music", ""), MAX_A3_CHARS))
    prompt = prompt.replace("__SEGMENTS__", format_segments_for_prompt(item.get("events", []), item.get("duration", 10.0)))
    prompt = prompt.replace("__DURATION__", str(item.get("duration", "")))
    return prompt


def strip_code_fence(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_object(text):
    text = strip_code_fence(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("JSON object not closed")


def parse_response(text):
    return json.loads(extract_json_object(text))


class GenerationError(Exception):
    def __init__(self, message, raw_response=None, parsed_response=None, reasoning_preview="", prompt_chars=0):
        super().__init__(message)
        self.raw_response = raw_response
        self.parsed_response = parsed_response
        self.reasoning_preview = reasoning_preview
        self.prompt_chars = prompt_chars


def normalize_answer_text(text):
    return " ".join(str(text or "").strip().lower().split())


def repair_cot_answer(cot, answer):
    if "<answer>" not in cot or "</answer>" not in cot:
        return cot
    before = cot.split("<answer>", 1)[0]
    after = cot.split("</answer>", 1)[1]
    return before + "<answer>" + answer + "</answer>" + after


def repair_answer_choice(obj):
    if not isinstance(obj, dict):
        return obj, False

    choices = obj.get("choices", [])
    if not isinstance(choices, list):
        return obj, False

    answer_norm = normalize_answer_text(obj.get("answer", ""))
    for choice in choices:
        if normalize_answer_text(choice) == answer_norm:
            obj["answer"] = choice
            obj["cot"] = repair_cot_answer(obj.get("cot", ""), choice)
            return obj, True

    cot = obj.get("cot", "")
    if "<answer>" in cot and "</answer>" in cot:
        inner = cot.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        inner_norm = normalize_answer_text(inner)
        for choice in choices:
            if normalize_answer_text(choice) == inner_norm:
                obj["answer"] = choice
                obj["cot"] = repair_cot_answer(cot, choice)
                return obj, True

    return obj, False


def validate_triplet(obj):
    required = {"question", "choices", "answer", "cot"}
    missing = required - set(obj.keys())
    if missing:
        return False, "missing keys: %s" % sorted(missing)
    if not isinstance(obj["question"], str) or not obj["question"].strip():
        return False, "bad question"
    if not isinstance(obj["choices"], list) or len(obj["choices"]) != 4:
        return False, "choices must be list of 4"
    if not all(isinstance(x, str) and x.strip() for x in obj["choices"]):
        return False, "bad choices"
    if obj["answer"] not in obj["choices"]:
        return False, "answer not in choices"
    cot = obj["cot"]
    if not isinstance(cot, str):
        return False, "cot not string"
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if tag not in cot:
            return False, "missing tag: %s" % tag
    answer_inside = cot.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if answer_inside != obj["answer"]:
        return False, "cot answer mismatch"
    if "<seg>" not in cot:
        return False, "missing seg"
    return True, ""


def make_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def call_model(prompt):
    client = make_client()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        content_chunks = []
        reasoning_chunks = []
        first_any = None
        first_reasoning = None
        first_content = None
        try:
            print("api attempt %d/%d, prompt chars: %d" % (attempt, MAX_RETRIES, len(prompt)))
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if first_any is None:
                        first_any = time.time()
                        print("first_chunk_seconds:", round(first_any - start, 2))
                    if first_reasoning is None:
                        first_reasoning = time.time()
                        print("first_reasoning_seconds:", round(first_reasoning - start, 2))
                    reasoning_chunks.append(reasoning)

                text = getattr(delta, "content", None)
                if text:
                    if first_any is None:
                        first_any = time.time()
                        print("first_chunk_seconds:", round(first_any - start, 2))
                    if first_content is None:
                        first_content = time.time()
                        print("first_content_seconds:", round(first_content - start, 2))
                    content_chunks.append(text)

            content = "".join(content_chunks).strip()
            reasoning_text = "".join(reasoning_chunks).strip()
            print("total_seconds:", round(time.time() - start, 2),
                  "content chars:", len(content),
                  "reasoning chars:", len(reasoning_text))
            if not content:
                raise RuntimeError("empty final content; reasoning chars=%d" % len(reasoning_text))
            return {
                "content": content,
                "reasoning_preview": reasoning_text[:1000],
            }

        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError, RuntimeError) as e:
            last_error = e
            wait = min(120, 10 * attempt)
            print("api failed attempt %d/%d: %r, sleep %ss" % (attempt, MAX_RETRIES, e, wait))
            time.sleep(wait)
    raise last_error


def process_one(line_no, item):
    sid = str(item.get("segment_id", ""))
    if not sid:
        raise ValueError("missing segment_id")

    prompt = build_prompt(item)
    result = call_model(prompt)
    raw = result["content"]
    reasoning_preview = result.get("reasoning_preview", "")
    prompt_chars = len(prompt)

    try:
        triplet = parse_response(raw)
    except Exception as e:
        raise GenerationError(
            "parse failed: %r" % e,
            raw_response=raw,
            reasoning_preview=reasoning_preview,
            prompt_chars=prompt_chars,
        )

    triplet, repaired = repair_answer_choice(triplet)
    ok, reason = validate_triplet(triplet)
    if not ok:
        raise GenerationError(
            reason,
            raw_response=raw,
            parsed_response=triplet,
            reasoning_preview=reasoning_preview,
            prompt_chars=prompt_chars,
        )

    return {
        "segment_id": sid,
        "audio_path": item.get("audio_path"),
        "source_item": item,
        "qa_cot": triplet,
        "raw_response": raw,
        "reasoning_preview": reasoning_preview,
        "prompt_chars": prompt_chars,
        "auto_repaired": repaired,
    }


def main():
    if not API_KEY:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable.")

    done_ids = load_done_ids(OUTPUT_JSONL)
    print("already done:", len(done_ids))

    tasks = []
    skipped = 0
    for line_no, item in read_jsonl(INPUT_JSONL):
        sid = str(item.get("segment_id", ""))
        if not sid:
            append_jsonl(ERROR_JSONL, {"line_no": line_no, "error": "missing segment_id", "item": item})
            continue
        if sid in done_ids:
            skipped += 1
            continue
        tasks.append((line_no, item))
        if MAX_ITEMS is not None and len(tasks) >= MAX_ITEMS:
            break

    print("pending this run:", len(tasks), "skipped:", skipped, "concurrency:", CONCURRENCY)

    processed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_map = {executor.submit(process_one, line_no, item): (line_no, item) for line_no, item in tasks}
        for future in as_completed(future_map):
            line_no, item = future_map[future]
            sid = str(item.get("segment_id", ""))
            try:
                out = future.result()
                append_jsonl(OUTPUT_JSONL, out)
                done_ids.add(sid)
                processed += 1
                print("processed:", processed, "segment_id:", sid)
                if SLEEP_SECONDS:
                    time.sleep(SLEEP_SECONDS)
            except Exception as e:
                failed += 1
                error_obj = {
                    "line_no": line_no,
                    "segment_id": sid,
                    "audio_path": item.get("audio_path"),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                if isinstance(e, GenerationError):
                    error_obj.update({
                        "raw_response": e.raw_response,
                        "raw_preview": str(e.raw_response or "")[:1000],
                        "parsed_response": e.parsed_response,
                        "reasoning_preview": e.reasoning_preview,
                        "prompt_chars": e.prompt_chars,
                    })
                append_jsonl(ERROR_JSONL, error_obj)
                print("failed:", sid, repr(e))

    print("done")
    print("processed:", processed)
    print("skipped:", skipped)
    print("failed:", failed)
    print("output:", OUTPUT_JSONL)
    print("errors:", ERROR_JSONL)


if __name__ == "__main__":
    main()

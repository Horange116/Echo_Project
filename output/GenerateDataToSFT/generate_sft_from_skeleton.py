# -*- coding: utf-8 -*-
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/qa_skeleton.jsonl"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/eaqa_sft_generated.jsonl"
ERROR_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/eaqa_sft_generated_errors.jsonl"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-HUnMwzbL0kiac1TnUkuk6BunochazpWF32mgTHM5nGDqeaoo")
BASE_URL = "https://yinli.one/v1"
MODEL_NAME = "deepseek-r1"

MAX_ITEMS = 3000
CONCURRENCY = 12
MAX_RETRIES = 3
REQUEST_TIMEOUT = 240.0
MAX_OUTPUT_TOKENS = 500

MAX_A1_CHARS = 350
MAX_A2_CHARS = 220
MAX_A3_CHARS = 220

SEG_PATTERN = re.compile(r"<seg>\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*</seg>")
POINT_SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*</seg>")


PROMPT_TEMPLATE = """You are refining an audio multiple-choice QA sample and writing the assistant answer.

The choices, correct answer, and evidence timestamps are fixed. Do not change the choices or the answer.
You may rewrite the question to sound natural, but it must ask the same temporal question and must still be answered by the fixed answer.

Write only:
<question>...</question>
<think>...</think><answer>__ANSWER__</answer>

Requirements:
1. The rewritten question must include the same event relation as the original question.
2. The reasoning must be natural and concise.
3. Use the evidence segments with <seg>start, end</seg> before discussing each concrete audio event.
4. The final answer must be exactly: __ANSWER__
5. Do not output JSON. Do not output anything before <question> or after </answer>.
6. Do not mention metadata, labels, annotations, A1, A2, A3, A4, or A5.
7. When writing calculations, avoid special math symbols. Use "times" instead of the multiplication sign and "about" instead of the approximately-equal sign.


Audio description:
__DESCRIPTION__

Speech information:
__SPEECH__

Music information:
__MUSIC__

Original question:
__QUESTION__

Choices:
__CHOICES__

Correct answer:
__ANSWER__

Evidence segments:
__EVIDENCE__
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


class ModelOutputError(Exception):
    def __init__(self, message, raw_response="", prompt_chars=0):
        super().__init__(message)
        self.raw_response = raw_response
        self.prompt_chars = prompt_chars


def load_done_ids(path):
    done = set()
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = obj.get("skeleton_id") or obj.get("segment_id")
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


def fmt_time(x):
    x = round(float(x), 3)
    s = ("%.3f" % x).rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def format_evidence(events):
    out = []
    for ev in events or []:
        out.append({
            "segment": "<seg>%s, %s</seg>" % (fmt_time(ev["start"]), fmt_time(ev["end"])),
            "sound": ev["label"],
            "start": fmt_time(ev["start"]),
            "end": fmt_time(ev["end"]),
            "duration": fmt_time(float(ev["end"]) - float(ev["start"])),
        })
    return json.dumps(out, ensure_ascii=False)


def build_prompt(item):
    source = item.get("source_item", {}) or {}
    choices = item.get("choices", [])
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("__DESCRIPTION__", compact_text(source.get("a1_description", ""), MAX_A1_CHARS))
    prompt = prompt.replace("__SPEECH__", compact_text(source.get("a2_speech", ""), MAX_A2_CHARS))
    prompt = prompt.replace("__MUSIC__", compact_text(source.get("a3_music", ""), MAX_A3_CHARS))
    prompt = prompt.replace("__QUESTION__", str(item["question"]))
    prompt = prompt.replace("__CHOICES__", json.dumps(choices, ensure_ascii=False))
    prompt = prompt.replace("__ANSWER__", str(item["answer"]))
    prompt = prompt.replace("__EVIDENCE__", format_evidence(item.get("evidence_events", [])))
    return prompt


def make_user_content(item):
    choices = item["choices"]
    return (
        "<audio>"
        + item["question"]
        + " Choose the answer from "
        + repr(choices)
        + ". Think step-by-step. Refer to the specific audio segments while thinking, and indicate the corresponding timestamps. "
        + "Answer in the format of <think>...</think><answer>...</answer>."
    )

def clean_generated_text(text):
    text = str(text or "")

    replacements = {
        "\u00a1\u00c1": " times ",   # ?¨¢
        "\u00a1\u00d6": " about ",   # ??
        "\u00d7": " times ",         # ¡Á
        "\u2248": " about ",         # ¡Ö
        "\u2264": " less than or equal to ",
        "\u2265": " greater than or equal to ",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"</think>\s+<answer>", "</think><answer>", text)
    text = re.sub(r"<think>\s+", "<think>", text)
    text = re.sub(r"\s+</think>", "</think>", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()



def extract_model_output(text):
    text = str(text or "").strip()
    text = normalize_model_tags(text)
    q_start = text.find("<question>")
    q_end = text.find("</question>")
    if q_start < 0 or q_end < 0 or q_end <= q_start:
        raise ValueError("missing question wrapper")

    question = text[q_start + len("<question>"):q_end].strip()
    if not question:
        raise ValueError("empty rewritten question")

    tail = text[q_end + len("</question>"):].strip()
    assistant = extract_assistant_content(tail)
    return question, assistant


def extract_assistant_content(text):
    text = str(text or "").strip()
    text = normalize_model_tags(text)
    start = text.find("<think>")
    end = text.rfind("</answer>")
    if end < 0:
        raise ValueError("missing think/answer wrapper")
    if start < 0:
        answer_start = text.rfind("<answer>")
        if answer_start < 0 or answer_start > end:
            raise ValueError("missing think/answer wrapper")
        reasoning = text[:answer_start].strip()
        if reasoning.startswith("<question>") and "</question>" in reasoning:
            reasoning = reasoning.split("</question>", 1)[1].strip()
        if not reasoning:
            raise ValueError("missing think/answer wrapper")
        return "<think>%s</think>%s" % (reasoning, text[answer_start:end + len("</answer>")].strip())
    return text[start:end + len("</answer>")].strip()


def normalize_model_tags(text):
    text = str(text or "")
    replacements = {
        "<reasoning>": "<think>",
        "</reasoning>": "</think>",
        "<cot>": "<think>",
        "</cot>": "</think>",
        "[think]": "<think>",
        "[/think]": "</think>",
        "[answer]": "<answer>",
        "[/answer]": "</answer>",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def repair_assistant_content(content, answer):
    content = normalize_model_tags(content).strip()

    if content.startswith("<think>") and "</think>" in content and "<answer>" not in content:
        content = content.rstrip() + "<answer>%s</answer>" % answer
    return content


def repair_point_segments(content, evidence_events):
    def find_interval(point):
        point = float(point)
        candidates = []
        for ev in evidence_events or []:
            try:
                start = float(ev["start"])
                end = float(ev["end"])
            except Exception:
                continue
            if abs(point - start) <= 0.02 or abs(point - end) <= 0.02:
                span = end - start
                candidates.append((span, start, end))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        _, start, end = candidates[0]
        return "<seg>%s, %s</seg>" % (fmt_time(start), fmt_time(end))

    def repl(match):
        fixed = find_interval(match.group(1))
        return fixed if fixed else match.group(0)

    return POINT_SEG_PATTERN.sub(repl, str(content or ""))


def validate_assistant(content, answer):
    if not content.startswith("<think>"):
        return False, "assistant does not start with <think>"
    if not content.endswith("</answer>"):
        return False, "assistant does not end with </answer>"
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if tag not in content:
            return False, "missing tag: %s" % tag
    answer_inside = content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if answer_inside != answer:
        return False, "answer mismatch"
    if not SEG_PATTERN.search(content):
        return False, "missing valid <seg>start, end</seg>"
    return True, ""


def make_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)


def call_model(prompt):
    client = make_client()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        content_chunks = []
        reasoning_chars = 0
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
                if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_chars += len(reasoning)
                text = getattr(delta, "content", None)
                if text:
                    if first_content is None:
                        first_content = time.time()
                        print("first_content_seconds:", round(first_content - start, 2))
                    content_chunks.append(text)
            raw = "".join(content_chunks).strip()
            print("total_seconds:", round(time.time() - start, 2), "content chars:", len(raw), "reasoning chars:", reasoning_chars)
            if not raw:
                raise RuntimeError("empty content")
            return raw
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError, RuntimeError) as e:
            last_error = e
            wait = min(90, 10 * attempt)
            print("api failed attempt %d/%d: %r, sleep %ss" % (attempt, MAX_RETRIES, e, wait))
            time.sleep(wait)
    raise last_error


def process_one(line_no, item):
    skeleton_id = str(item.get("skeleton_id") or item.get("segment_id"))
    prompt = build_prompt(item)
    raw = call_model(prompt)
    try:
        rewritten_question, assistant = extract_model_output(raw)
        rewritten_question = clean_generated_text(rewritten_question)
        assistant = clean_generated_text(assistant)
    except Exception:
        try:
            rewritten_question = item["question"]
            assistant = extract_assistant_content(raw)
            rewritten_question = clean_generated_text(rewritten_question)
            assistant = clean_generated_text(assistant)
        except Exception as e:
            raise ModelOutputError(
                "extract failed: %r" % e,
                raw_response=raw,
                prompt_chars=len(prompt),
            )

    assistant = repair_assistant_content(assistant, item["answer"])
    assistant = repair_point_segments(assistant, item.get("evidence_events", []))
    assistant = clean_generated_text(assistant)

    ok, reason = validate_assistant(assistant, item["answer"])
    if not ok:
        raise ModelOutputError(
            reason,
            raw_response=raw,
            prompt_chars=len(prompt),
        )

    return {
        "skeleton_id": skeleton_id,
        "segment_id": item.get("segment_id"),
        "messages": [
            {"role": "user", "content": make_user_content({**item, "question": rewritten_question})},
            {"role": "assistant", "content": assistant},
        ],
        "audios": [item["audio_path"]],
        "source_skeleton": item,
        "raw_response": raw,
        "prompt_chars": len(prompt),
        "rewritten_question": rewritten_question,
    }


def main():
    if not API_KEY:
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable.")

    done = load_done_ids(OUTPUT_JSONL)
    tasks = []
    skipped = 0
    for line_no, item in read_jsonl(INPUT_JSONL):
        sid = str(item.get("skeleton_id") or item.get("segment_id"))
        if sid in done:
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
            sid = str(item.get("skeleton_id") or item.get("segment_id"))
            try:
                out = future.result()
                append_jsonl(OUTPUT_JSONL, out)
                processed += 1
                print("processed:", processed, "skeleton_id:", sid)
            except Exception as e:
                failed += 1
                error_obj = {
                    "line_no": line_no,
                    "skeleton_id": sid,
                    "segment_id": item.get("segment_id"),
                    "audio_path": item.get("audio_path"),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "source_skeleton": item,
                }
                if isinstance(e, ModelOutputError):
                    error_obj.update({
                        "raw_response": e.raw_response,
                        "raw_preview": str(e.raw_response or "")[:1500],
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

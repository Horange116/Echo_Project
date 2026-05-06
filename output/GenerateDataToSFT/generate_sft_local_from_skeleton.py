# -*- coding: utf-8 -*-
import json
import random
import re
import traceback
from pathlib import Path


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/qa_skeleton.jsonl"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/eaqa_sft_local_generated_3000_12999.jsonl"
ERROR_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/eaqa_sft_local_generated_3000_12999_errors.jsonl"

# DeepSeek is planned to cover skeleton indices [0, 2999].
# This script covers [3000, 12999].
START_INDEX = 3000
MAX_ITEMS = 10000
RANDOM_SEED = 42

SEG_PATTERN = re.compile(r"<seg>\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*</seg>")


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


def fmt_time(x):
    x = round(float(x), 3)
    s = ("%.3f" % x).rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def fmt_seconds(x):
    x = max(0.0, float(x))
    if x < 0.05:
        return "0 seconds"
    val = round(x, 1)
    if abs(val - 1.0) < 1e-9:
        return "1.0 second"
    return "%.1f seconds" % val


def fmt_percent_from_value(x, rounding="nearest"):
    x = max(0.0, min(100.0, float(x)))
    if rounding == "floor":
        val = int(x // 5 * 5)
    else:
        val = int(round(x / 5.0) * 5)
    val = max(0, min(100, val))
    return "%d%%" % val


def seg(ev):
    return "<seg>%s, %s</seg>" % (fmt_time(ev["start"]), fmt_time(ev["end"]))


def phrase(ev):
    if ev.get("phrase"):
        return str(ev["phrase"])
    label = str(ev.get("label", "sound")).strip()
    label = label.replace(", man speaking", "").replace(", woman speaking", "")
    return "the " + label.lower()


def duration(ev):
    if "duration" in ev:
        return float(ev["duration"])
    return float(ev["end"]) - float(ev["start"])


def make_user_content(item):
    return (
        "<audio>"
        + item["question"]
        + " Choose the answer from "
        + repr(item["choices"])
        + ". Think step-by-step. Refer to the specific audio segments while thinking, and indicate the corresponding timestamps. "
        + "Answer in the format of <think>...</think><answer>...</answer>."
    )


def make_assistant(item):
    qa_type = item["qa_type"]
    evs = item.get("evidence_events", [])
    answer = item["answer"]
    duration_total = float(item.get("duration") or 10.0)

    if qa_type == "gap":
        a, b = evs[0], evs[1]
        gap = float(b["start"]) - float(a["end"])
        templates = [
            "%s contains %s and ends at %s seconds. %s contains %s and begins at %s seconds. The gap is %s - %s = %.3f seconds, which rounds to %s.",
            "The relevant first event is %s, where %s ends at %s seconds. The next event is %s, where %s starts at %s seconds. Subtracting the end time from the next start time gives %.3f seconds, so the closest choice is %s.",
        ]
        if random.randrange(2) == 0:
            text = templates[0] % (seg(a), phrase(a), fmt_time(a["end"]), seg(b), phrase(b), fmt_time(b["start"]), fmt_time(b["start"]), fmt_time(a["end"]), gap, answer)
        else:
            text = templates[1] % (seg(a), phrase(a), fmt_time(a["end"]), seg(b), phrase(b), fmt_time(b["start"]), gap, answer)

    elif qa_type == "repeated_event_gap":
        a, b = evs[0], evs[1]
        gap = float(b["start"]) - float(a["end"])
        text = (
            "%s marks the earlier occurrence, which ends at %s seconds. "
            "%s marks the following occurrence, which starts at %s seconds. "
            "The elapsed time between them is %s - %s = %.3f seconds, rounding to %s."
        ) % (seg(a), fmt_time(a["end"]), seg(b), fmt_time(b["start"]), fmt_time(b["start"]), fmt_time(a["end"]), gap, answer)

    elif qa_type == "overlap":
        a, b = evs[0], evs[1]
        start = max(float(a["start"]), float(b["start"]))
        end = min(float(a["end"]), float(b["end"]))
        overlap = max(0.0, end - start)
        text = (
            "%s contains %s from %s to %s seconds. "
            "%s contains %s from %s to %s seconds. "
            "Their shared interval runs from %s to %s seconds, lasting %.3f seconds. Rounded to one decimal place, this is %s."
        ) % (
            seg(a), phrase(a), fmt_time(a["start"]), fmt_time(a["end"]),
            seg(b), phrase(b), fmt_time(b["start"]), fmt_time(b["end"]),
            fmt_time(start), fmt_time(end), overlap, answer,
        )

    elif qa_type == "duration_compare":
        a, b = evs[0], evs[1]
        da, db = duration(a), duration(b)
        longer = phrase(a) if da > db else phrase(b)
        text = (
            "%s contains %s and lasts %.3f seconds. "
            "%s contains %s and lasts %.3f seconds. "
            "Since %.3f seconds is longer than %.3f seconds, the longer event is %s."
        ) % (seg(a), phrase(a), da, seg(b), phrase(b), db, max(da, db), min(da, db), longer)

    elif qa_type == "order":
        a, b = evs[0], evs[1]
        first = phrase(a) if float(a["start"]) <= float(b["start"]) else phrase(b)
        text = (
            "%s shows %s beginning at %s seconds. "
            "%s shows %s beginning at %s seconds. "
            "Comparing the two start times, %s begins first."
        ) % (seg(a), phrase(a), fmt_time(a["start"]), seg(b), phrase(b), fmt_time(b["start"]), first)

    elif qa_type == "start_percentage":
        a = evs[0]
        pct = float(a["start"]) / duration_total * 100.0 if duration_total > 0 else 0.0
        text = (
            "%s contains %s, which begins at %s seconds. "
            "The full audio lasts %s seconds. The start point is %s / %s times 100 = %.2f%% of the total duration. "
            "Using the available choices, this corresponds to %s."
        ) % (seg(a), phrase(a), fmt_time(a["start"]), fmt_time(duration_total), fmt_time(a["start"]), fmt_time(duration_total), pct, answer)

    elif qa_type == "duration_percentage":
        a = evs[0]
        dur = duration(a)
        pct = dur / duration_total * 100.0 if duration_total > 0 else 0.0
        text = (
            "%s contains %s and lasts %.3f seconds. "
            "The full audio lasts %s seconds. Dividing %.3f by %s and multiplying by 100 gives %.2f%%, which is closest to %s."
        ) % (seg(a), phrase(a), dur, fmt_time(duration_total), dur, fmt_time(duration_total), pct, answer)

    elif qa_type == "count_before":
        if len(evs) < 2:
            raise ValueError("count_before requires evidence events")
        target = evs[-1]
        completed = [ev for ev in evs[:-1] if float(ev["end"]) <= float(target["start"]) + 1e-6]
        parts = []
        for idx, ev in enumerate(completed, 1):
            parts.append("%s shows completed event %d ending at %s seconds" % (seg(ev), idx, fmt_time(ev["end"])))
        text = (
            "%s. %s contains the target event, which begins at %s seconds. "
            "There are %d completed timestamped events before that start time, so the answer is %s."
        ) % ("; ".join(parts), seg(target), fmt_time(target["start"]), len(completed), answer)

    else:
        raise ValueError("unsupported qa_type: %s" % qa_type)

    return "<think>%s</think><answer>%s</answer>" % (text.strip(), answer)


def validate_output(obj):
    if "messages" not in obj or "audios" not in obj:
        return False, "missing messages/audios"
    msgs = obj["messages"]
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False, "messages length"
    if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        return False, "bad roles"
    user = msgs[0].get("content", "")
    assistant = msgs[1].get("content", "")
    if not user.startswith("<audio>") or "Choose the answer from" not in user:
        return False, "bad user content"
    if not assistant.startswith("<think>") or not assistant.endswith("</answer>"):
        return False, "bad assistant wrapper"
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if tag not in assistant:
            return False, "missing tag %s" % tag
    if not SEG_PATTERN.search(assistant):
        return False, "missing seg"
    answer_inside = assistant.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    expected = obj.get("source_skeleton", {}).get("answer")
    if expected is not None and answer_inside != expected:
        return False, "answer mismatch"
    return True, ""


def build_output(item):
    assistant = make_assistant(item)
    out = {
        "skeleton_id": item.get("skeleton_id"),
        "segment_id": item.get("segment_id"),
        "messages": [
            {"role": "user", "content": make_user_content(item)},
            {"role": "assistant", "content": assistant},
        ],
        "audios": [item["audio_path"]],
        "source_skeleton": item,
        "generator": "local_template",
    }
    ok, reason = validate_output(out)
    if not ok:
        raise ValueError(reason)
    return out


def main():
    random.seed(RANDOM_SEED)
    done = load_done_ids(OUTPUT_JSONL)
    print("already done:", len(done))

    made = 0
    skipped = 0
    failed = 0

    for idx, item in read_jsonl(INPUT_JSONL):
        if idx < START_INDEX:
            skipped += 1
            continue
        if made >= MAX_ITEMS:
            break

        sid = str(item.get("skeleton_id") or item.get("segment_id"))
        if not sid:
            failed += 1
            append_jsonl(ERROR_JSONL, {"line_no": idx, "error": "missing skeleton_id", "item": item})
            continue
        if sid in done:
            skipped += 1
            continue

        try:
            out = build_output(item)
            append_jsonl(OUTPUT_JSONL, out)
            done.add(sid)
            made += 1
            if made % 1000 == 0:
                print("made:", made)
        except Exception as e:
            failed += 1
            append_jsonl(ERROR_JSONL, {
                "line_no": idx,
                "skeleton_id": sid,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })

    print("done")
    print("made:", made)
    print("skipped:", skipped)
    print("failed:", failed)
    print("output:", OUTPUT_JSONL)
    print("errors:", ERROR_JSONL)


if __name__ == "__main__":
    main()

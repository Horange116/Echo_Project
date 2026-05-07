# -*- coding: utf-8 -*-
import json
import math
import random
from pathlib import Path


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/metadata_with_qwen_audio_info.jsonl"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/qa_skeleton.jsonl"
ERROR_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/qa_skeleton_errors.jsonl"

MAX_ITEMS = None
RANDOM_SEED = 42

# Keep the generated skeleton stream balanced. The values do not need to sum
# to 1; they are used as relative quotas while iterating through the input.
TARGET_QA_WEIGHTS = {
    "start_percentage": 18,
    "duration_percentage": 16,
    "gap": 16,
    "overlap": 16,
    "count_before": 12,
    "repeated_event_gap": 10,
    "duration_compare": 8,
    "order": 4,
}


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


def load_done_ids_and_type_counts(path):
    done = set()
    type_counts = {}
    if not Path(path).exists():
        return done, type_counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = obj.get("skeleton_id") or obj.get("segment_id")
                qa_type = obj.get("qa_type")
                if sid:
                    done.add(str(sid))
                if qa_type:
                    qa_type = str(qa_type)
                    type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
            except Exception:
                pass
    return done, type_counts


def clean_label(label):
    label = str(label or "").strip()
    label = label.replace("_", " ")
    return label


LABEL_PHRASES = {
    "Generic impact sounds": "impact sounds",
    "Male speech, man speaking": "male speech",
    "Female speech, woman speaking": "female speech",
    "Female singing": "female singing",
    "Male singing": "male singing",
    "Human voice": "human voice",
    "Music": "music",
    "Surface contact": "surface contact",
    "Air horn, truck horn": "truck horn",
    "Medium engine (mid frequency)": "engine noise",
    "Accelerating, revving, vroom": "revving engine",
    "Race car, auto racing": "race car sound",
    "Truck": "truck engine",
    "Vehicle": "vehicle noise",
    "Car": "car sound",
    "Engine": "engine noise",
    "Engine starting": "engine starting",
    "Mechanisms": "mechanical noise",
    "Noise": "noise",
    "Radio": "radio audio",
    "Breathing": "breathing",
    "Wind": "wind",
    "Tick": "ticking",
    "Tap": "tapping",
    "Emergency vehicle": "emergency vehicle siren",
}


def label_phrase(label):
    label = clean_label(label)
    if label in LABEL_PHRASES:
        return LABEL_PHRASES[label]
    lowered = label.lower()
    lowered = lowered.replace(", man speaking", "")
    lowered = lowered.replace(", woman speaking", "")
    lowered = lowered.replace("generic ", "")
    return lowered


def event_phrase(ev):
    return label_phrase(ev["label"])


def ordinal(n):
    names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
    }
    return names.get(n, "%dth" % n)


def with_occurrence_phrases(events):
    counts = {}
    totals = {}
    for ev in events:
        phrase = event_phrase(ev)
        totals[phrase] = totals.get(phrase, 0) + 1
    out = []
    for ev in events:
        phrase = event_phrase(ev)
        counts[phrase] = counts.get(phrase, 0) + 1
        ev = dict(ev)
        if totals[phrase] > 1:
            ev["phrase"] = "the %s %s" % (ordinal(counts[phrase]), phrase)
        else:
            ev["phrase"] = "the %s" % phrase
        out.append(ev)
    return out


def is_full_span(ev, duration):
    return ev["start"] <= 0.05 and abs(ev["end"] - duration) <= 0.05


def normalize_events(item):
    duration = float(item.get("duration") or 10.0)
    out = []
    for ev in item.get("events", []) or []:
        try:
            start = float(ev["start"])
            end = float(ev["end"])
            label = clean_label(ev["label"])
        except Exception:
            continue
        if end <= start:
            continue
        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "label": label,
            "duration": round(end - start, 3),
        })
    out.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return out


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


def make_second_choices(answer_value):
    base = max(0.0, round(float(answer_value), 1))
    candidates = {
        base,
        max(0.0, round(base - 0.3, 1)),
        round(base + 0.3, 1),
        round(base + 0.6, 1),
        max(0.0, round(base - 0.6, 1)),
        round(base + 1.0, 1),
    }
    candidates = sorted(candidates, key=lambda x: (abs(x - base), x))
    values = []
    for x in candidates:
        if fmt_seconds(x) not in [fmt_seconds(v) for v in values]:
            values.append(x)
        if len(values) == 4:
            break
    while len(values) < 4:
        values.append(round(base + len(values) + 1, 1))
    random.shuffle(values)
    answer = fmt_seconds(base)
    choices = [fmt_seconds(x) for x in values]
    if answer not in choices:
        choices[0] = answer
        random.shuffle(choices)
    return choices, answer


def fmt_percent(x):
    x = max(0.0, min(100.0, float(x)))
    val = int(round(x / 5.0) * 5)
    return "%d%%" % val


def make_percent_choices(answer_value, rounding="nearest"):
    value = max(0.0, min(100.0, float(answer_value)))
    if rounding == "floor":
        base = int(math.floor(value / 5.0) * 5)
    else:
        base = int(round(value / 5.0) * 5)
    candidates = [
        base,
        max(0, base - 10),
        min(100, base + 10),
        max(0, base - 20),
        min(100, base + 20),
        0,
        50,
        100,
    ]
    values = []
    for x in candidates:
        if x not in values:
            values.append(x)
        if len(values) == 4:
            break
    while len(values) < 4:
        values.append(min(100, len(values) * 25))
        values = list(dict.fromkeys(values))
    random.shuffle(values)
    answer = "%d%%" % base
    choices = ["%d%%" % x for x in values[:4]]
    if answer not in choices:
        choices[0] = answer
        random.shuffle(choices)
    return choices, answer


def make_count_choices(answer_value):
    base = int(answer_value)
    candidates = [base, max(0, base - 1), base + 1, base + 2, max(0, base - 2)]
    values = []
    for x in candidates:
        if x not in values:
            values.append(x)
        if len(values) == 4:
            break
    while len(values) < 4:
        x = len(values)
        if x not in values:
            values.append(x)
    random.shuffle(values)
    answer = str(base)
    choices = [str(x) for x in values[:4]]
    if answer not in choices:
        choices[0] = answer
        random.shuffle(choices)
    return choices, answer


def build_gap(item, events, duration):
    non_full = with_occurrence_phrases([ev for ev in events if not is_full_span(ev, duration)])
    for a in non_full:
        for b in non_full:
            if b["start"] <= a["end"]:
                continue
            gap = b["start"] - a["end"]
            if 0.1 <= gap <= 5.0 and event_phrase(a) != event_phrase(b):
                choices, answer = make_second_choices(gap)
                question = "How long after %s ends does %s begin?" % (a["phrase"], b["phrase"])
                return {
                    "qa_type": "gap",
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "evidence_events": [a, b],
                }
    return None


def build_overlap(item, events, duration):
    non_full = [ev for ev in events if not is_full_span(ev, duration)]
    full = [ev for ev in events if is_full_span(ev, duration)]
    pairs = []
    for source in (non_full, events):
        for i, a in enumerate(source):
            for b in source[i + 1:]:
                pairs.append((a, b))

    for a, b in pairs:
            if event_phrase(a) == event_phrase(b):
                continue
            start = max(a["start"], b["start"])
            end = min(a["end"], b["end"])
            overlap = end - start
            full_pair = is_full_span(a, duration) or is_full_span(b, duration)
            if overlap >= 0.2 and not full_pair:
                choices, answer = make_second_choices(overlap)
                question = "How long do %s and %s overlap?" % ("the " + event_phrase(a), "the " + event_phrase(b))
                return {
                    "qa_type": "overlap",
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "evidence_events": [a, b],
                }
    for a in full:
        for b in non_full:
            if event_phrase(a) == event_phrase(b):
                continue
            start = max(a["start"], b["start"])
            end = min(a["end"], b["end"])
            overlap = end - start
            if overlap >= 0.5:
                choices, answer = make_second_choices(overlap)
                question = "For how long is %s audible while %s is also present?" % ("the " + event_phrase(b), "the " + event_phrase(a))
                return {
                    "qa_type": "overlap",
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "evidence_events": [a, b],
                }
    return None


def build_repeated_event_gap(item, events, duration):
    by_label = {}
    for ev in events:
        if is_full_span(ev, duration):
            continue
        by_label.setdefault(event_phrase(ev), []).append(ev)
    for label, group in by_label.items():
        if len(group) < 2:
            continue
        group = with_occurrence_phrases(sorted(group, key=lambda x: x["start"]))
        for a, b in zip(group, group[1:]):
            gap = b["start"] - a["end"]
            if 0.1 <= gap <= 5.0:
                choices, answer = make_second_choices(gap)
                question = "How much time passes between the end of %s and the start of %s?" % (a["phrase"], b["phrase"])
                return {
                    "qa_type": "repeated_event_gap",
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "evidence_events": [a, b],
                }
    return None


def build_duration_compare(item, events, duration):
    non_full = with_occurrence_phrases([ev for ev in events if not is_full_span(ev, duration)])
    for i, a in enumerate(non_full):
        for b in non_full[i + 1:]:
            diff = abs(a["duration"] - b["duration"])
            if diff < 0.2 or event_phrase(a) == event_phrase(b):
                continue
            a_choice = a["phrase"]
            b_choice = b["phrase"]
            answer = a_choice if a["duration"] > b["duration"] else b_choice
            choices = [a_choice, b_choice, "they last the same amount of time", "neither sound is audible"]
            question = "Which event lasts longer, %s or %s?" % (a["phrase"], b["phrase"])
            return {
                "qa_type": "duration_compare",
                "question": question,
                "choices": choices,
                "answer": answer,
                "evidence_events": [a, b],
            }
    return None


def build_order(item, events, duration):
    non_full = with_occurrence_phrases([ev for ev in events if not is_full_span(ev, duration) and ev["start"] > 0.05])
    for i, a in enumerate(non_full):
        for b in non_full[i + 1:]:
            if event_phrase(a) == event_phrase(b):
                continue
            gap = b["start"] - a["start"]
            if 0.2 <= gap <= 6.0:
                first = a["phrase"]
                second = b["phrase"]
                choices = [first, second, "they start at the same time", "neither sound occurs"]
                question = "Which event begins first, %s or %s?" % (first, second)
                return {
                    "qa_type": "order",
                    "question": question,
                    "choices": choices,
                    "answer": first,
                    "evidence_events": [a, b],
                }
    return None


def build_start_percentage(item, events, duration):
    candidates = with_occurrence_phrases([
        ev for ev in events
        if not is_full_span(ev, duration) and duration > 0 and 0.05 <= ev["start"] <= duration - 0.05
    ])
    if not candidates:
        return None
    candidates.sort(key=lambda ev: abs((ev["start"] / duration * 100.0) - 50.0), reverse=True)
    for ev in candidates:
        percent = ev["start"] / duration * 100.0
        rounded = int(math.floor(percent / 5.0) * 5)
        if 0 <= rounded <= 100:
            choices, answer = make_percent_choices(percent, rounding="floor")
            question = "At what percentage of the audio's total duration does %s begin?" % ev["phrase"]
            return {
                "qa_type": "start_percentage",
                "question": question,
                "choices": choices,
                "answer": answer,
                "evidence_events": [ev],
            }
    return None


def build_duration_percentage(item, events, duration):
    candidates = with_occurrence_phrases([
        ev for ev in events
        if duration > 0 and not is_full_span(ev, duration) and ev["duration"] >= 0.2
    ])
    if not candidates:
        return None
    candidates.sort(key=lambda ev: abs((ev["duration"] / duration * 100.0) - 35.0))
    for ev in candidates:
        percent = ev["duration"] / duration * 100.0
        choices, answer = make_percent_choices(percent)
        question = "About what percentage of the full audio does %s occupy?" % ev["phrase"]
        return {
            "qa_type": "duration_percentage",
            "question": question,
            "choices": choices,
            "answer": answer,
            "evidence_events": [ev],
        }
    return None


def build_count_before(item, events, duration):
    non_full = with_occurrence_phrases([ev for ev in events if not is_full_span(ev, duration)])
    if len(non_full) < 3:
        return None
    for target in non_full[2:]:
        completed = [
            ev for ev in non_full
            if ev is not target and ev["end"] <= target["start"] - 0.02
        ]
        count = len(completed)
        if 1 <= count <= 8:
            choices, answer = make_count_choices(count)
            question = "How many timestamped sound events have finished before %s begins?" % target["phrase"]
            evidence = completed[-2:] + [target]
            return {
                "qa_type": "count_before",
                "question": question,
                "choices": choices,
                "answer": answer,
                "evidence_events": evidence,
            }
    return None


def add_skeleton_metadata(item, qa, duration):
    sid = str(item.get("segment_id"))
    qa.update({
        "skeleton_id": "%s:%s" % (sid, qa["qa_type"]),
        "segment_id": sid,
        "video_id": item.get("video_id"),
        "audio_path": item.get("audio_path"),
        "duration": duration,
        "source_item": item,
    })
    return qa


def build_skeleton_candidates(item):
    duration = float(item.get("duration") or 10.0)
    events = normalize_events(item)
    if len(events) < 2:
        return []

    builders = [
        build_start_percentage,
        build_duration_percentage,
        build_gap,
        build_overlap,
        build_count_before,
        build_repeated_event_gap,
        build_duration_compare,
        build_order,
    ]
    candidates = []
    for builder in builders:
        qa = builder(item, events, duration)
        if qa:
            candidates.append(add_skeleton_metadata(item, qa, duration))
    return candidates


def choose_balanced_candidate(candidates, type_counts):
    if not candidates:
        return None

    total = sum(type_counts.values())

    def score(qa):
        qa_type = qa["qa_type"]
        target = TARGET_QA_WEIGHTS.get(qa_type, 1)
        current = type_counts.get(qa_type, 0)
        expected = (total + 1) * target / float(sum(TARGET_QA_WEIGHTS.values()))
        deficit = expected - current
        return (deficit, target, random.random())

    return max(candidates, key=score)


def build_skeleton(item, type_counts=None):
    candidates = build_skeleton_candidates(item)
    if type_counts is None:
        random.shuffle(candidates)
        return candidates[0] if candidates else None
    return choose_balanced_candidate(candidates, type_counts)


def main():
    random.seed(RANDOM_SEED)
    done, type_counts = load_done_ids_and_type_counts(OUTPUT_JSONL)
    made = 0
    skipped = 0
    failed = 0

    for line_no, item in read_jsonl(INPUT_JSONL):
        if MAX_ITEMS is not None and made >= MAX_ITEMS:
            break
        sid = str(item.get("segment_id", ""))
        if not sid:
            failed += 1
            append_jsonl(ERROR_JSONL, {"line_no": line_no, "error": "missing segment_id", "item": item})
            continue
        if any(x.startswith(sid + ":") for x in done):
            skipped += 1
            continue
        try:
            qa = build_skeleton(item, type_counts)
            if not qa:
                skipped += 1
                append_jsonl(ERROR_JSONL, {"line_no": line_no, "segment_id": sid, "error": "no valid skeleton"})
                continue
            append_jsonl(OUTPUT_JSONL, qa)
            done.add(qa["skeleton_id"])
            qa_type = qa["qa_type"]
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
            made += 1
            if made % 1000 == 0:
                print("made:", made)
        except Exception as e:
            failed += 1
            append_jsonl(ERROR_JSONL, {"line_no": line_no, "segment_id": sid, "error": repr(e)})

    print("done")
    print("made:", made)
    print("skipped:", skipped)
    print("failed:", failed)
    print("type_counts:", type_counts)
    print("output:", OUTPUT_JSONL)
    print("errors:", ERROR_JSONL)


if __name__ == "__main__":
    main()

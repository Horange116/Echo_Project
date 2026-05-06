# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from collections import Counter


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/all_summary_metadata.jsonl"
STATE_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/judge_summary_metadata.jsonl"
REPORT_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/judge_summary_metadata.jsonl"

FULL_REBUILD = False


def new_state():
    return {
        "input_jsonl": INPUT_JSONL,
        "file_offset": 0,
        "processed_lines": 0,
        "bad_lines": 0,
        "samples": 0,
        "audio_path_exists": 0,
        "missing_audio_path": 0,
        "duration_missing": 0,
        "duration_invalid": 0,
        "event_count_hist": {},
        "label_counter": {},
        "samples_events_ge_2": 0,
        "samples_repeated_labels": 0,
        "samples_with_overlap": 0,
        "samples_with_ordered_non_overlap": 0,
        "samples_with_short_events": 0,
        "samples_only_full_span_events": 0,
        "eligible_for_temporal_qa": 0,
    }


def load_state(path):
    path = Path(path)
    if FULL_REBUILD or not path.exists():
        return new_state()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def counter_from_dict(d):
    return Counter({str(k): int(v) for k, v in d.items()})


def counter_to_dict(c):
    return dict(c.most_common())


def normalize_events(events):
    out = []
    if not events:
        return out

    for ev in events:
        try:
            start = float(ev["start"])
            end = float(ev["end"])
            label = str(ev["label"])
        except Exception:
            continue

        if end <= start:
            continue

        out.append({
            "start": start,
            "end": end,
            "label": label,
            "duration": end - start,
        })

    out.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return out


def has_overlap(events):
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a = events[i]
            b = events[j]
            if min(a["end"], b["end"]) > max(a["start"], b["start"]):
                return True
    return False


def has_ordered_non_overlap(events):
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j:
                continue
            if events[i]["end"] < events[j]["start"]:
                return True
    return False


def only_full_span_events(events, duration):
    if not events or duration is None:
        return False

    full_count = 0
    for ev in events:
        if abs(ev["start"] - 0.0) <= 0.05 and abs(ev["end"] - duration) <= 0.05:
            full_count += 1

    return full_count == len(events)


def is_eligible_for_temporal_qa(events, duration):
    if len(events) < 2:
        return False

    if duration is None or duration <= 0:
        return False

    if only_full_span_events(events, duration):
        return False

    if has_overlap(events):
        return True

    if has_ordered_non_overlap(events):
        return True

    labels = [ev["label"] for ev in events]
    if len(labels) != len(set(labels)):
        return True

    return False


def update_state_with_item(state, item, counters):
    state["samples"] += 1

    audio_path = item.get("audio_path")
    if audio_path:
        if os.path.exists(audio_path):
            state["audio_path_exists"] += 1
        else:
            state["missing_audio_path"] += 1
    else:
        state["missing_audio_path"] += 1

    duration = item.get("duration")
    try:
        duration = float(duration)
        if duration <= 0:
            state["duration_invalid"] += 1
            duration = None
    except Exception:
        state["duration_missing"] += 1
        duration = None

    events = normalize_events(item.get("events", []))
    event_count = len(events)

    counters["event_count_hist"][str(event_count)] += 1

    for ev in events:
        counters["label_counter"][ev["label"]] += 1

    if event_count >= 2:
        state["samples_events_ge_2"] += 1

    labels = [ev["label"] for ev in events]
    if len(labels) != len(set(labels)):
        state["samples_repeated_labels"] += 1

    if has_overlap(events):
        state["samples_with_overlap"] += 1

    if has_ordered_non_overlap(events):
        state["samples_with_ordered_non_overlap"] += 1

    if any(ev["duration"] <= 1.0 for ev in events):
        state["samples_with_short_events"] += 1

    if only_full_span_events(events, duration):
        state["samples_only_full_span_events"] += 1

    if is_eligible_for_temporal_qa(events, duration):
        state["eligible_for_temporal_qa"] += 1


def build_report(state):
    samples = max(int(state["samples"]), 1)

    def pct(x):
        return round(float(x) * 100.0 / samples, 2)

    report = dict(state)
    report["rates"] = {
        "audio_path_exists_pct": pct(state["audio_path_exists"]),
        "events_ge_2_pct": pct(state["samples_events_ge_2"]),
        "repeated_labels_pct": pct(state["samples_repeated_labels"]),
        "overlap_pct": pct(state["samples_with_overlap"]),
        "ordered_non_overlap_pct": pct(state["samples_with_ordered_non_overlap"]),
        "short_events_pct": pct(state["samples_with_short_events"]),
        "only_full_span_events_pct": pct(state["samples_only_full_span_events"]),
        "eligible_for_temporal_qa_pct": pct(state["eligible_for_temporal_qa"]),
    }

    return report


def main():
    input_path = Path(INPUT_JSONL)

    state = load_state(STATE_PATH)

    counters = {
        "event_count_hist": counter_from_dict(state.get("event_count_hist", {})),
        "label_counter": counter_from_dict(state.get("label_counter", {})),
    }

    start_offset = 0 if FULL_REBUILD else int(state.get("file_offset", 0))

    file_size = input_path.stat().st_size
    if start_offset > file_size:
        print("Input file is smaller than saved offset. Rebuilding from start.")
        state = new_state()
        counters = {
            "event_count_hist": Counter(),
            "label_counter": Counter(),
        }
        start_offset = 0

    new_lines = 0
    new_samples = 0

    with open(input_path, "rb") as f:
        f.seek(start_offset)

        for raw_line in f:
            new_lines += 1
            state["processed_lines"] += 1

            try:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                item = json.loads(line)
                update_state_with_item(state, item, counters)
                new_samples += 1

            except Exception:
                state["bad_lines"] += 1

        state["file_offset"] = f.tell()

    state["event_count_hist"] = counter_to_dict(counters["event_count_hist"])
    state["label_counter"] = counter_to_dict(counters["label_counter"])

    save_json_atomic(STATE_PATH, state)

    report = build_report(state)
    save_json_atomic(REPORT_PATH, report)

    print("done")
    print("input:", INPUT_JSONL)
    print("new lines:", new_lines)
    print("new samples:", new_samples)
    print("total samples:", state["samples"])
    print("bad lines:", state["bad_lines"])
    print("eligible_for_temporal_qa:", state["eligible_for_temporal_qa"])
    print("report:", REPORT_PATH)
    print("state:", STATE_PATH)


if __name__ == "__main__":
    main()

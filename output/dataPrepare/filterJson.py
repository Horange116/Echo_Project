# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from collections import Counter


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/all_summary_metadata.jsonl"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/filtered_temporal_metadata.jsonl"
REPORT_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/filter_report.json"

MIN_EVENT_DURATION = 0.05
FULL_SPAN_TOL = 0.05
SHORT_EVENT_MAX_DURATION = 3.0
MIN_GAP = 0.05
MAX_GAP = 5.0


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def normalize_events(events):
    out = []
    for ev in events or []:
        try:
            start = float(ev["start"])
            end = float(ev["end"])
            label = str(ev["label"])
        except Exception:
            continue
        if end - start < MIN_EVENT_DURATION:
            continue
        out.append({
            "start": start,
            "end": end,
            "label": label,
            "duration": end - start,
        })
    out.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return out


def is_full_span(ev, duration):
    return abs(ev["start"]) <= FULL_SPAN_TOL and abs(ev["end"] - duration) <= FULL_SPAN_TOL


def non_full_events(events, duration):
    return [ev for ev in events if not is_full_span(ev, duration)]


def has_gap_pair(events):
    for a in events:
        for b in events:
            if a is b:
                continue
            gap = b["start"] - a["end"]
            if MIN_GAP <= gap <= MAX_GAP:
                return True
    return False


def has_overlap_pair(events):
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if overlap > MIN_EVENT_DURATION:
                return True
    return False


def has_duration_compare_pair(events):
    durations = [round(ev["duration"], 2) for ev in events]
    return len(set(durations)) >= 2


def has_repeated_label(events):
    labels = [ev["label"] for ev in events]
    return len(labels) != len(set(labels))


def candidate_types(events, duration):
    core_events = non_full_events(events, duration)
    types = []

    if len(core_events) < 2:
        return types

    if has_gap_pair(core_events):
        types.append("gap")

    if has_overlap_pair(core_events):
        types.append("overlap")

    if has_duration_compare_pair(core_events):
        types.append("duration_compare")

    if has_repeated_label(core_events):
        types.append("repeated_event_gap")

    if len(core_events) >= 3:
        types.append("order")

    return types


def main():
    total = 0
    kept = 0
    skipped = Counter()
    type_counter = Counter()

    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for line_no, item in load_jsonl(INPUT_JSONL):
            total += 1

            audio_path = item.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                skipped["missing_audio"] += 1
                continue

            try:
                duration = float(item.get("duration"))
                if duration <= 0:
                    skipped["bad_duration"] += 1
                    continue
            except Exception:
                skipped["bad_duration"] += 1
                continue

            events = normalize_events(item.get("events", []))
            if len(events) < 2:
                skipped["events_lt_2"] += 1
                continue

            core_events = non_full_events(events, duration)
            if len(core_events) < 2:
                skipped["too_few_non_full_events"] += 1
                continue

            types = candidate_types(events, duration)
            if not types:
                skipped["no_candidate_type"] += 1
                continue

            item["events"] = events
            item["qa_candidate_types"] = types
            item["non_full_event_count"] = len(core_events)

            for t in types:
                type_counter[t] += 1

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "input_jsonl": INPUT_JSONL,
        "output_jsonl": OUTPUT_JSONL,
        "total": total,
        "kept": kept,
        "kept_pct": round(kept * 100.0 / max(total, 1), 2),
        "skipped": dict(skipped),
        "candidate_type_counts": dict(type_counter),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("done")
    print("total:", total)
    print("kept:", kept)
    print("kept_pct:", report["kept_pct"])
    print("skipped:", dict(skipped))
    print("candidate_type_counts:", dict(type_counter))
    print("output:", OUTPUT_JSONL)
    print("report:", REPORT_PATH)


if __name__ == "__main__":
    main()

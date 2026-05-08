#!/usr/bin/env python3
"""
Split judged QA-CoT data into SFT / RL / discard streams.

SFT:  qa_valid=True AND cot_valid=True   (keep all fields)
RL:   qa_valid=True AND cot_valid=False  (strip CoT, keep minimal fields)
DISCARD: qa_valid=False or null          (keep for reference / inspection)
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone


def get_type(item):
    return item.get("type") or item.get("qa_type") or item.get("skeleton_type", "unknown")


def get_id(item):
    return item.get("id") or item.get("skeleton_id") or item.get("segment_id") or ""


RL_KEEP_FIELDS = [
    "id", "audio_path", "question", "choices", "answer",
    "original_index", "judge_source_group",
]


def main():
    parser = argparse.ArgumentParser(
        description="Split judged QA-CoT data into SFT / RL / discard"
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_sft_jsonl", required=True)
    parser.add_argument("--output_rl_jsonl", required=True)
    parser.add_argument("--output_discard_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    args = parser.parse_args()

    # ── Read ──
    items = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # ── Split ──
    sft_records = []
    rl_records = []
    discard_records = []

    type_counts = Counter()
    source_group_counts = Counter()
    source_group_type_counts = Counter()

    for item in items:
        qa_valid = item.get("qa_valid")
        cot_valid = item.get("cot_valid")
        qa_type = get_type(item)
        source_group = item.get("judge_source_group", "unknown")

        type_counts[qa_type] += 1
        source_group_counts[source_group] += 1
        source_group_type_counts[f"{source_group}/{qa_type}"] += 1

        if qa_valid is True and cot_valid is True:
            sft_records.append(item)
        elif qa_valid is True and cot_valid is False:
            rl_record = {}
            for k in RL_KEEP_FIELDS:
                if k in item:
                    rl_record[k] = item[k]
            rl_record["type"] = qa_type  # normalized field name
            rl_records.append(rl_record)
        else:
            discard_records.append(item)

    # ── Write ──
    for records, path in [
        (sft_records, args.output_sft_jsonl),
        (rl_records, args.output_rl_jsonl),
        (discard_records, args.output_discard_jsonl),
    ]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Report ──
    report = {
        "input": args.input_jsonl,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(items),
        "sft_count": len(sft_records),
        "rl_count": len(rl_records),
        "discard_count": len(discard_records),
        "by_type": dict(type_counts),
        "by_source_group": dict(source_group_counts),
        "by_source_group_and_type": dict(
            sorted(source_group_type_counts.items())
        ),
    }

    report_dir = os.path.dirname(args.report_json)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    print(f"Total:  {len(items)}")
    print(f"  SFT:     {len(sft_records)}")
    print(f"  RL:      {len(rl_records)}")
    print(f"  Discard: {len(discard_records)}")


if __name__ == "__main__":
    main()

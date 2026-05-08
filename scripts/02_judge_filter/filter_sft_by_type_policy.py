#!/usr/bin/env python3
"""
Route SFT candidate data by type policy, based on judge diagnostic results.

Rather than permanently dropping low-quality types, this script routes them to
needs_review_or_rewrite for future improvement, while clean samples proceed to SFT.

Routing policy (from 600-sample judge):
  - start_percentage (64% QA pass): route to needs_review_or_rewrite
  - deepseek_polished + overlap (63.2% QA pass): route to needs_review_or_rewrite
  - template_or_unpolished + overlap (86.5% QA pass): keep as SFT candidate
  - All other types: keep as SFT candidate

Usage:
  python scripts/02_judge_filter/filter_sft_by_type_policy.py \
    --input_jsonl output/GeneratedData/qa_skeleton.jsonl \
    --output_sft_jsonl output/judge/sft_candidates.jsonl \
    --output_needs_review_jsonl output/judge/needs_review.jsonl \
    --report_json output/judge/filter_report.json
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone


def get_type_from_item(item):
    for k in ("type", "qa_type", "skeleton_type"):
        v = item.get(k)
        if v:
            return v
    ss = item.get("source_skeleton")
    if isinstance(ss, dict):
        for k in ("qa_type", "type"):
            v = ss.get(k)
            if v:
                return v
    return "unknown"


def build_skeleton_index_map(skeleton_path):
    mapping = {}
    with open(skeleton_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("skeleton_id")
            if sid:
                mapping[sid] = i
    return mapping


def get_original_index(item, skeleton_map):
    idx = item.get("original_index")
    if idx is not None:
        return int(idx)
    sid = item.get("skeleton_id") or item.get("id") or ""
    return skeleton_map.get(sid, -1)


def resolve_source_group(original_index):
    return "deepseek_polished" if original_index < 3000 else "template_or_unpolished"


def parse_pairs(raw):
    if not raw:
        return set()
    pairs = set()
    for token in raw.split(","):
        token = token.strip()
        if ":" in token:
            parts = token.split(":", 1)
            pairs.add((parts[0].strip(), parts[1].strip()))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Route SFT candidate data by type policy"
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_sft_jsonl", required=True)
    parser.add_argument("--output_needs_review_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--skeleton_jsonl",
                        default="output/GeneratedData/qa_skeleton.jsonl",
                        help="qa_skeleton.jsonl for index mapping")
    parser.add_argument("--review_types",
                        default="start_percentage",
                        help="Types to route to needs_review (comma-separated)")
    parser.add_argument("--review_source_type_pairs",
                        default="deepseek_polished:overlap",
                        help="Source:type pairs to route to needs_review")
    parser.add_argument("--sft_types",
                        default="gap,duration_compare,repeated_event_gap,"
                                "count_before,duration_percentage,order,overlap",
                        help="Types eligible for SFT (informational)")
    args = parser.parse_args()

    review_types = {t.strip() for t in args.review_types.split(",") if t.strip()}
    review_pairs = parse_pairs(args.review_source_type_pairs)
    sft_types = {t.strip() for t in args.sft_types.split(",") if t.strip()}

    # ── Build skeleton index mapping ──
    print("Building skeleton index map ...")
    skeleton_map = build_skeleton_index_map(args.skeleton_jsonl)
    print(f"  Mapped {len(skeleton_map)} skeleton IDs")

    # ── Read input ──
    items = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    total_in = len(items)
    print(f"  Input: {total_in} records")

    # ── Route ──
    sft_records = []
    review_records = []

    before_type_counts = Counter()
    after_type_counts = Counter()
    review_type_counts = Counter()
    before_source_type_counts = Counter()
    after_source_type_counts = Counter()
    review_source_type_counts = Counter()

    for item in items:
        qa_type = get_type_from_item(item)
        orig_idx = get_original_index(item, skeleton_map)
        source_group = resolve_source_group(orig_idx)
        source_type_key = f"{source_group}/{qa_type}"

        before_type_counts[qa_type] += 1
        before_source_type_counts[source_type_key] += 1

        # Routing logic:
        # 1. If type is in review_types → needs_review
        # 2. If source:type is in review_pairs → needs_review
        # 3. Otherwise → SFT candidate

        if qa_type in review_types:
            review_records.append((item, f"review_type:{qa_type}"))
            review_type_counts[qa_type] += 1
            review_source_type_counts[source_type_key] += 1
        elif (source_group, qa_type) in review_pairs:
            review_records.append((item, f"review_pair:{source_group}:{qa_type}"))
            review_type_counts[qa_type] += 1
            review_source_type_counts[source_type_key] += 1
        else:
            sft_records.append(item)
            after_type_counts[qa_type] += 1
            after_source_type_counts[source_type_key] += 1

    # ── Write outputs ──
    for records, path in [
        (sft_records, args.output_sft_jsonl),
        ([r[0] for r in review_records], args.output_needs_review_jsonl),
    ]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Report ──
    # Breakdown of review by reason
    review_by_type = Counter()
    review_by_pair = Counter()
    for _, reason in review_records:
        if reason.startswith("review_type:"):
            t = reason.split(":", 1)[1]
            review_by_type[t] += 1
        elif reason.startswith("review_pair:"):
            parts = reason.split(":")
            if len(parts) >= 3:
                key = f"{parts[1]}/{parts[2]}"
                review_by_pair[key] += 1

    report = {
        "input": args.input_jsonl,
        "skeleton_jsonl": args.skeleton_jsonl,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "review_types": args.review_types,
            "review_source_type_pairs": args.review_source_type_pairs,
            "sft_types": args.sft_types,
        },
        "total_input": total_in,
        "sft_candidates": len(sft_records),
        "needs_review": len(review_records),
        "routed_to_review_by_type": dict(review_by_type),
        "routed_to_review_by_source_type_pair": dict(review_by_pair),
        "before_by_type": dict(before_type_counts),
        "after_by_type": dict(after_type_counts),
        "review_by_type": dict(review_type_counts),
        "before_by_source_type": dict(sorted(before_source_type_counts.items())),
        "after_by_source_type": dict(sorted(after_source_type_counts.items())),
        "review_by_source_type": dict(sorted(review_source_type_counts.items())),
    }

    report_dir = os.path.dirname(args.report_json)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    print()
    print(f"Total:         {total_in}")
    print(f"  SFT:          {len(sft_records)}")
    print(f"  Needs review: {len(review_records)}")
    print(f"  → by type:    {sum(review_by_type.values())}")
    for t, c in sorted(review_by_type.items()):
        print(f"      {t}: {c}")
    print(f"  → by pair:    {sum(review_by_pair.values())}")
    for k, c in sorted(review_by_pair.items()):
        print(f"      {k}: {c}")


if __name__ == "__main__":
    main()

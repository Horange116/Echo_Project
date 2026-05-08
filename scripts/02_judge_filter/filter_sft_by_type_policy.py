#!/usr/bin/env python3
"""
Filter SFT candidate data by type policy, based on judge diagnostic results.

Drops low-quality types and source_group × type pairs identified by the 600-sample
judge, while preserving the original record schema.

Default policy (based on judge results):
  - Drop start_percentage entirely              (worst: 64% QA pass)
  - Drop deepseek_polished + overlap            (only 63.2% QA pass in polished)
  - Keep everything else

Usage:
  python scripts/filter_sft_by_type_policy.py \
    --input_jsonl output/GeneratedData/eaqa_sft_generated.jsonl \
    --output_jsonl output/judge/sft_filtered.jsonl \
    --report_json output/judge/filter_report.json
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone


def get_type_from_item(item):
    """Extract type from item, checking source_skeleton if needed."""
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
    """Build skeleton_id → 0-based line index mapping from qa_skeleton.jsonl."""
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
    """Get original_index from item or via skeleton_id mapping."""
    idx = item.get("original_index")
    if idx is not None:
        return int(idx)
    sid = item.get("skeleton_id") or item.get("id") or ""
    return skeleton_map.get(sid, -1)


def resolve_source_group(original_index):
    if original_index < 3000:
        return "deepseek_polished"
    return "template_or_unpolished"


def parse_pairs(raw):
    """Parse 'deepseek_polished:overlap,foo:bar' into set of (source, type)."""
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
        description="Filter SFT candidate data by type policy"
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--skeleton_jsonl",
                        default="output/GeneratedData/qa_skeleton.jsonl",
                        help="qa_skeleton.jsonl for index mapping (default: "
                             "output/GeneratedData/qa_skeleton.jsonl)")
    parser.add_argument("--drop_types",
                        default="start_percentage",
                        help="Comma-separated type names to drop entirely")
    parser.add_argument("--drop_source_type_pairs",
                        default="deepseek_polished:overlap",
                        help="Comma-separated source:type pairs to drop")
    parser.add_argument("--keep_types",
                        default="gap,duration_compare,repeated_event_gap,"
                                "count_before,duration_percentage,order,overlap",
                        help="Types to keep (informational)")
    args = parser.parse_args()

    drop_types = {t.strip() for t in args.drop_types.split(",") if t.strip()}
    drop_pairs = parse_pairs(args.drop_source_type_pairs)
    keep_types = {t.strip() for t in args.keep_types.split(",") if t.strip()}

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

    # ── Filter ──
    kept = []
    dropped = []
    before_type_counts = Counter()
    after_type_counts = Counter()
    before_source_type_counts = Counter()
    after_source_type_counts = Counter()

    for item in items:
        qa_type = get_type_from_item(item)
        orig_idx = get_original_index(item, skeleton_map)
        source_group = resolve_source_group(orig_idx)
        source_type_key = f"{source_group}/{qa_type}"

        before_type_counts[qa_type] += 1
        before_source_type_counts[source_type_key] += 1

        # Apply drop rules
        if qa_type in drop_types:
            dropped.append((item, f"drop_type:{qa_type}"))
            continue

        if (source_group, qa_type) in drop_pairs:
            dropped.append((item, f"drop_pair:{source_group}:{qa_type}"))
            continue

        kept.append(item)
        after_type_counts[qa_type] += 1
        after_source_type_counts[source_type_key] += 1

    # ── Write output ──
    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Report ──
    # Count drop reasons
    drop_reason_counts = Counter(reason for _, reason in dropped)
    # Summary by drop_type / drop_pair
    drop_type_summary = {}
    drop_pair_summary = {}
    for item, reason in dropped:
        qa_type = get_type_from_item(item)
        if reason.startswith("drop_type:"):
            drop_type_summary[qa_type] = drop_type_summary.get(qa_type, 0) + 1
        elif reason.startswith("drop_pair:"):
            # reason = "drop_pair:deepseek_polished:overlap"
            parts = reason.split(":")
            if len(parts) >= 3:
                src, t = parts[1], parts[2]
                key = f"{src}/{t}"
                drop_pair_summary[key] = drop_pair_summary.get(key, 0) + 1

    report = {
        "input": args.input_jsonl,
        "skeleton_jsonl": args.skeleton_jsonl,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "drop_types": args.drop_types,
            "drop_source_type_pairs": args.drop_source_type_pairs,
            "keep_types": args.keep_types,
        },
        "total_input": total_in,
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_by_type": drop_type_summary,
        "dropped_by_source_type_pair": drop_pair_summary,
        "before_by_type": dict(before_type_counts),
        "after_by_type": dict(after_type_counts),
        "before_by_source_type": dict(
            sorted(before_source_type_counts.items())
        ),
        "after_by_source_type": dict(
            sorted(after_source_type_counts.items())
        ),
    }

    report_dir = os.path.dirname(args.report_json)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    print()
    print(f"Total:  {total_in}")
    print(f"  Kept:   {len(kept)}")
    print(f"  Dropped by type: {sum(drop_type_summary.values())}")
    for t, c in sorted(drop_type_summary.items()):
        print(f"    - {t}: {c}")
    print(f"  Dropped by source:type pair: {sum(drop_pair_summary.values())}")
    for k, c in sorted(drop_pair_summary.items()):
        print(f"    - {k}: {c}")
    print(f"  Output: {args.output_jsonl}")
    print(f"  Report: {args.report_json}")


if __name__ == "__main__":
    main()

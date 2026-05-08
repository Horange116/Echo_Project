#!/usr/bin/env python3
"""
Quality statistics on routed SFT candidates, using actual generated CoT data.

Maps skeleton_ids from SFT candidates / needs_review into the generated data
(DeepSeek + template) and computes training-readiness metrics.
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone

SEG_PATTERN = re.compile(r"<seg>\s*([\d.]+)\s*,\s*([\d.]+)\s*</seg>")
THINK_PATTERN = re.compile(r"<think>")
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_cot(item):
    """Extract CoT text from a generated record."""
    msgs = item.get("messages", [])
    if msgs and isinstance(msgs, list):
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return item.get("raw_response", "") or ""


def get_choices(item):
    ss = item.get("source_skeleton", {})
    if isinstance(ss, dict):
        choices = ss.get("choices", [])
        if choices:
            return choices
    return []


def get_answer(item):
    ss = item.get("source_skeleton", {})
    if isinstance(ss, dict):
        ans = ss.get("answer", "")
        if ans:
            return str(ans).strip()
    return ""


def get_duration(item):
    ss = item.get("source_skeleton", {})
    if isinstance(ss, dict):
        return ss.get("duration")
    return None


def get_type(item):
    ss = item.get("source_skeleton", {})
    if isinstance(ss, dict):
        return ss.get("qa_type") or ss.get("type", "unknown")
    return "unknown"


def get_skeleton_id(item):
    sid = item.get("skeleton_id")
    if sid:
        return sid
    ss = item.get("source_skeleton", {})
    if isinstance(ss, dict):
        return ss.get("skeleton_id", "")
    return ""


def load_index_map(path):
    mapping = {}
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            sid = d.get("skeleton_id")
            if sid:
                mapping[sid] = i
    return mapping


def build_skel_id_set(path):
    ids = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("skeleton_id") or d.get("id") or ""
            if sid:
                ids.add(sid)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Quality statistics on SFT candidates"
    )
    parser.add_argument("--sft_candidates_jsonl", required=True)
    parser.add_argument("--needs_review_jsonl", required=True)
    parser.add_argument("--generated_files", nargs="+", required=True,
                        help="Generated JSONL files (DeepSeek + template)")
    parser.add_argument("--skeleton_jsonl",
                        default="output/GeneratedData/qa_skeleton.jsonl")
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--inspection_jsonl", required=True,
                        help="Output 3 random samples per type for inspection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_per_type", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load candidate ID sets ──
    print("Loading SFT candidate IDs ...")
    sft_ids = build_skel_id_set(args.sft_candidates_jsonl)
    review_ids = build_skel_id_set(args.needs_review_jsonl)
    print(f"  SFT candidates: {len(sft_ids)}")
    print(f"  Needs review:   {len(review_ids)}")

    # ── Load generated data ──
    print("Loading generated CoT data ...")
    all_generated = []
    for fp in args.generated_files:
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                all_generated.append(d)
    print(f"  Total generated records: {len(all_generated)}")

    # Build skeleton index map for source_group
    skel_map = load_index_map(args.skeleton_jsonl)

    def source_group(skid):
        idx = skel_map.get(skid, -1)
        return "deepseek_polished" if idx < 3000 else "template_or_unpolished"

    # ── Match generated records to SFT candidates ──
    sft_matched = []
    review_matched = []
    unmatched_sft = 0
    for d in all_generated:
        sid = get_skeleton_id(d)
        if sid in sft_ids:
            sft_matched.append(d)
        elif sid in review_ids:
            review_matched.append(d)
    print(f"  SFT matched:   {len(sft_matched)}")
    print(f"  Review matched: {len(review_matched)}")

    # ── Stats on SFT candidates ──
    stats = {
        "total_candidates": len(sft_ids),
        "total_with_cot": len(sft_matched),
        "by_type": Counter(),
        "by_source_group": Counter(),
        "by_type_and_source": Counter(),
        "has_think_answer": 0,
        "has_seg": 0,
        "fully_structured": 0,
        "answer_in_choices": 0,
        "cot_lengths": [],
        "seg_counts": Counter(),
        "invalid_timestamps": 0,
        "invalid_timestamp_details": [],
    }

    # Inspection samples: dict[type] -> list of records
    inspect_pool = {}

    total_seg_count = 0

    for d in sft_matched:
        sid = get_skeleton_id(d)
        qa_type = get_type(d)
        sg = source_group(sid)
        cot = extract_cot(d)
        cot_len = len(cot)
        choices = get_choices(d)
        answer = get_answer(d)
        duration = get_duration(d)

        stats["by_type"][qa_type] += 1
        stats["by_source_group"][sg] += 1
        stats["by_type_and_source"][f"{sg}/{qa_type}"] += 1
        stats["cot_lengths"].append(cot_len)

        has_think = bool(THINK_PATTERN.search(cot))
        has_answer_tag = bool(ANSWER_PATTERN.search(cot))
        stats["has_think_answer"] += (1 if has_think and has_answer_tag else 0)

        segs = SEG_PATTERN.findall(cot)
        n_seg = len(segs)
        stats["seg_counts"][n_seg] += 1
        total_seg_count += n_seg

        if n_seg > 0:
            stats["has_seg"] += 1
            # Validate timestamps
            for start_s, end_s in segs:
                try:
                    start = float(start_s)
                    end = float(end_s)
                    if start < 0 or end < start:
                        raise ValueError
                    if duration is not None and (start > duration or end > duration):
                        stats["invalid_timestamps"] += 1
                        stats["invalid_timestamp_details"].append(
                            f"{sid}: seg=[{start},{end}] dur={duration}"
                        )
                except (ValueError, TypeError):
                    stats["invalid_timestamps"] += 1
                    stats["invalid_timestamp_details"].append(
                        f"{sid}: seg=[{start_s},{end_s}] parse_error"
                    )

        if has_think and has_answer_tag and n_seg > 0:
            stats["fully_structured"] += 1

        # Answer in choices
        if choices and answer:
            # Normalize and check
            ans_clean = answer.strip().lower().rstrip(".")
            choices_clean = [c.strip().lower().rstrip(".") for c in choices]
            if ans_clean in choices_clean:
                stats["answer_in_choices"] += 1

        # Collect for inspection pool (per type)
        if qa_type not in inspect_pool:
            inspect_pool[qa_type] = []
        inspect_pool[qa_type].append(d)

    n = len(sft_matched)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sft_candidates": args.sft_candidates_jsonl,
            "needs_review": args.needs_review_jsonl,
            "generated_files": args.generated_files,
        },
        "total": {
            "sft_candidate_skeletons": stats["total_candidates"],
            "with_generated_cot": stats["total_with_cot"],
            "cot_coverage_pct": round(
                stats["total_with_cot"] / stats["total_candidates"] * 100, 1
            ) if stats["total_candidates"] else 0,
        },
        "by_type": {k: v for k, v in sorted(stats["by_type"].items())},
        "by_source_group": {
            k: v for k, v in sorted(stats["by_source_group"].items())
        },
        "by_type_and_source": {
            k: v for k, v in sorted(stats["by_type_and_source"].items())
        },
        "quality": {
            "has_think_answer": {
                "count": stats["has_think_answer"],
                "rate_pct": round(
                    stats["has_think_answer"] / n * 100, 1
                ) if n else 0,
            },
            "has_seg": {
                "count": stats["has_seg"],
                "rate_pct": round(stats["has_seg"] / n * 100, 1) if n else 0,
            },
            "fully_structured": {
                "count": stats["fully_structured"],
                "rate_pct": round(
                    stats["fully_structured"] / n * 100, 1
                ) if n else 0,
            },
            "answer_in_choices": {
                "count": stats["answer_in_choices"],
                "rate_pct": round(
                    stats["answer_in_choices"] / n * 100, 1
                ) if n else 0,
            },
            "avg_cot_length": round(
                sum(stats["cot_lengths"]) / len(stats["cot_lengths"]), 1
            ) if stats["cot_lengths"] else 0,
            "median_cot_length": sorted(stats["cot_lengths"])[
                len(stats["cot_lengths"]) // 2
            ] if stats["cot_lengths"] else 0,
            "avg_seg_per_sample": round(
                total_seg_count / n, 2
            ) if n else 0,
        },
        "seg_count_distribution": {
            str(k): v for k, v in sorted(stats["seg_counts"].items())
        },
        "invalid_timestamps": {
            "count": stats["invalid_timestamps"],
            "examples": stats["invalid_timestamp_details"][:10],
        },
    }

    # Also compute stats on needs_review
    if review_matched:
        review_has_both = sum(
            1 for d in review_matched
            if bool(THINK_PATTERN.search(extract_cot(d)))
            and bool(ANSWER_PATTERN.search(extract_cot(d)))
        )
        report["needs_review"] = {
            "total": len(review_ids),
            "with_cot": len(review_matched),
            "has_think_answer": review_has_both,
        }
    else:
        report["needs_review"] = {
            "total": len(review_ids),
            "with_cot": 0,
        }

    # ── Write report ──
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Write inspection samples ──
    os.makedirs(os.path.dirname(args.inspection_jsonl) or ".", exist_ok=True)
    per_type = args.inspect_per_type
    inspection_written = 0
    with open(args.inspection_jsonl, "w", encoding="utf-8") as f:
        for qa_type in sorted(inspect_pool.keys()):
            pool = inspect_pool[qa_type]
            samples = random.sample(pool, min(per_type, len(pool)))
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                inspection_written += 1
    print(f"  Inspection samples written: {inspection_written}")

    # ── Print summary ──
    print()
    print("=" * 60)
    print(f"SFT candidates:       {stats['total_candidates']}")
    print(f"  With generated CoT: {stats['total_with_cot']} "
          f"({report['total']['cot_coverage_pct']}%)")
    print()
    print("Quality (on generated CoT):")
    q = report["quality"]
    print(f"  has_think_answer:   {q['has_think_answer']['count']} "
          f"({q['has_think_answer']['rate_pct']}%)")
    print(f"  has_seg:            {q['has_seg']['count']} "
          f"({q['has_seg']['rate_pct']}%)")
    print(f"  fully_structured:   {q['fully_structured']['count']} "
          f"({q['fully_structured']['rate_pct']}%)")
    print(f"  answer_in_choices:  {q['answer_in_choices']['count']} "
          f"({q['answer_in_choices']['rate_pct']}%)")
    print(f"  avg_cot_length:     {q['avg_cot_length']}")
    print(f"  median_cot_length:  {q['median_cot_length']}")
    print(f"  avg_seg_per_sample: {q['avg_seg_per_sample']}")
    print(f"  invalid_timestamps: {report['invalid_timestamps']['count']}")
    print()
    print(f"Report:  {args.report_json}")
    print(f"Inspect: {args.inspection_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    main()

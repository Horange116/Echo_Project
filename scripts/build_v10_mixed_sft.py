#!/usr/bin/env python3
"""
v10 mixed SFT dataset: 70% diverse CoT / 30% clean CoT with type-aware weights.

Usage:
    python scripts/build_v10_mixed_sft.py

Output:
    output/GeneratedData/eaqa_sft_v10_mixed_70diverse_30clean.jsonl
    output/judge/v10_mixed_data_report.json
"""

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project"

CLEAN_PATH = f"{BASE}/output/GeneratedData/eaqa_sft_v9_clean.jsonl"
DIVERSE_PATH = f"{BASE}/output/GeneratedData/eaqa_sft_v9_clean_diverse_cot.jsonl"
OUTPUT_PATH = f"{BASE}/output/GeneratedData/eaqa_sft_v10_mixed_70diverse_30clean.jsonl"
REPORT_PATH = f"{BASE}/output/judge/v10_mixed_data_report.json"

SEED = 42

# ── classifier (from enhance_cot_templates.py) ──────────────────────

def classify_question(text: str) -> str:
    q = text.lower()
    if re.search(r"repeated\s+event|recurrence|recurring|same\s+(?:event|sound|type).+"
                 r"(?:again|second|another|occurrence|instance)|"
                 r"gap between the (?:first|two|same)", q):
        return "repeated_event_gap"
    if re.search(r"(?:time|temporal)\s+gap|gap\s+between|how\s+much\s+time\s+between|"
                 r"interval\s+between", q):
        return "gap"
    if re.search(r"overlap|simultaneously|at\s+the\s+same\s+time|co.occur|"
                 r"concurrently|occur\s+together|how\s+long\s+do\s+(?:both|they).+overlap", q):
        return "overlap"
    if re.search(r"how\s+many|count|number\s+of.+before|events?\s+before|"
                 r"finish\s+before|end\s+before|completed\s+before|which.+ended\s+before", q):
        return "count_before"
    if re.search(r"which\s+(?:happened|came|one|event)\s+(?:first|earlier)|"
                 r"order|occur\s+first|what\s+first|what\s+happens\s+first|"
                 r"which.+first|occur\s+before|preced", q):
        return "order"
    if re.search(r"which\s+(?:lasts|is)\s+longer|shorter|compare\s+duration|"
                 r"longer\s+duration|duration\s+(?:compar|difference)|"
                 r"which.+longer|last\s+longer|longest", q):
        return "duration_compare"
    if re.search(r"what\s+percentage|percentage\s+of.+duration|what\s+proportion", q):
        return "duration_percentage"
    if re.search(r"at\s+what\s+percentage|what\s+point.+percent|"
                 r"percentage.+start|when.+begin.+percent|how\s+far.+percent", q):
        return "start_percentage"
    # fallback
    if re.search(r"gap|how\s+long\s+(?:after|before|between)", q):
        return "gap"
    if re.search(r"overlap|simultane", q):
        return "overlap"
    if re.search(r"count|how\s+many|number\s+of", q):
        return "count_before"
    if re.search(r"first|earlier|order|before|preced", q):
        return "order"
    if re.search(r"percent|percentage|%", q):
        return "duration_percentage"
    if re.search(r"longer|shorter|compare", q):
        return "duration_compare"
    return "unknown"


# ── type-aware diverse_ratio ────────────────────────────────────────
# Based on v9a vs v9b eval comparison on manifest_500
TYPE_WEIGHTS = {
    "count_before":      0.80,  # seg 98%→100%, acc 58%→60%  (both improved, use diverse)
    "order":             0.80,  # seg 84%→100%, acc 94%→97%  (both improved)
    "duration_compare":  0.50,  # seg 60%→63%, acc 74%→63%  (acc dropped, balanced)
    "repeated_event_gap":0.30,  # seg 85%→52%, acc 35%→15%  (both dropped, prefer clean)
    "overlap":           0.70,  # seg 36%→52%, acc 27%→20%  (seg up, acc slightly down)
    "gap":               0.85,  # seg 9%→69%, acc 35%→33%   (seg huge, acc stable)
    "start_percentage":  0.80,  # seg 17%→79%, acc 24%→18%  (seg huge, acc slight drop)
    "duration_percentage":0.85, # seg 0%→93%, acc 42%→34%   (seg huge, acc moderate drop)
    "unknown":           0.70,  # default fallback
}


def extract_assistant_text(msg):
    """Get assistant response text regardless of format."""
    content = msg.get("content", "")
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return " ".join(texts)
    return str(content)


def compute_stats(text):
    """Compute CoT quality stats for report."""
    think_match = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", text, re.S)
    has_think_answer = bool(think_match)
    if think_match:
        think_content = think_match.group(1)
    else:
        think_content = text

    segs = re.findall(r"<seg>\s*([\d.]+)\s*,\s*([\d.]+)\s*</seg>", think_content)
    has_seg = len(segs) > 0
    seg_count = len(segs)

    answer_text = think_match.group(2).strip() if think_match else ""
    words = think_content.split()
    word_count = len(words)

    # Validate timestamps
    invalid_ts = 0
    for s, e in segs:
        try:
            fs, fe = float(s), float(e)
            if fs < 0 or fe < 0 or fs > fe:
                invalid_ts += 1
        except ValueError:
            invalid_ts += 1

    # answer_in_choices - rough check if answer looks like a choice
    answer_in_choices = 0
    if answer_text:
        # Simple check: answer is reasonably short (looks like a choice)
        if len(answer_text) < 100:
            answer_in_choices = 1

    return {
        "has_think_answer": has_think_answer,
        "has_seg": has_seg,
        "seg_count": seg_count,
        "word_count": word_count,
        "invalid_timestamps": invalid_ts,
        "answer_in_choices": answer_in_choices,
    }


def main():
    random.seed(SEED)

    # Load both files
    clean_rows = []
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                clean_rows.append(json.loads(line))
    print(f"Loaded {len(clean_rows)} clean rows")

    diverse_rows = []
    with open(DIVERSE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                diverse_rows.append(json.loads(line))
    print(f"Loaded {len(diverse_rows)} diverse rows")

    assert len(clean_rows) == len(diverse_rows), \
        f"Row count mismatch: clean={len(clean_rows)} diverse={len(diverse_rows)}"

    total = len(clean_rows)

    # Classify each sample and decide source
    diverse_count = 0
    clean_count = 0
    type_diverse = defaultdict(int)
    type_clean = defaultdict(int)
    type_total = defaultdict(int)

    output_items = []
    all_stats_clean = []
    all_stats_diverse = []

    for idx in range(total):
        clean_item = clean_rows[idx]
        diverse_item = diverse_rows[idx]

        # Extract question text for classification
        user_msg_clean = clean_item["messages"][0]["content"]
        user_msg_diverse = diverse_item["messages"][0]["content"]

        # Sanity check: user messages should match
        q_text = user_msg_clean  # both are same

        q_type = classify_question(q_text)
        diverse_ratio = TYPE_WEIGHTS.get(q_type, 0.70)
        type_total[q_type] += 1

        # Pick source
        use_diverse = random.random() < diverse_ratio

        if use_diverse:
            item = diverse_item
            all_stats_diverse.append(compute_stats(extract_assistant_text(diverse_item["messages"][1])))
            diverse_count += 1
            type_diverse[q_type] += 1
        else:
            item = clean_item
            all_stats_clean.append(compute_stats(extract_assistant_text(clean_item["messages"][1])))
            clean_count += 1
            type_clean[q_type] += 1

        output_items.append(item)

    # Write output
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in output_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output_items)} items to {OUTPUT_PATH}")

    # Compute aggregate stats
    def avg(lists, key):
        vals = [s[key] for s in lists]
        return sum(vals) / len(vals) if vals else 0

    def rate(lists, key):
        vals = [s[key] for s in lists]
        return sum(vals) / len(vals) if vals else 0

    # Combined stats for ALL output items
    output_clean_stats = []
    output_diverse_stats = []
    for item in output_items:
        asst_text = extract_assistant_text(item["messages"][1])
        s = compute_stats(asst_text)
        # Need to know if this is clean or diverse... re-derive
        q_text = item["messages"][0]["content"]
        if isinstance(q_text, list):
            q_text = " ".join(c.get("text", "") for c in q_text if isinstance(c, dict))
        q_type = classify_question(q_text)
        diverse_ratio = TYPE_WEIGHTS.get(q_type, 0.70)
        output_diverse_stats.append(s)  # just aggregate all

    report = {
        "config": {
            "seed": SEED,
            "default_diverse_ratio": 0.70,
            "type_weights": TYPE_WEIGHTS,
        },
        "total": total,
        "diverse_count": diverse_count,
        "diverse_ratio": round(diverse_count / total, 4),
        "clean_count": clean_count,
        "clean_ratio": round(clean_count / total, 4),
        "by_type": {},
        "output_stats": {
            "avg_word_count": round(avg(output_diverse_stats, "word_count"), 1),
            "avg_seg_count": round(avg(output_diverse_stats, "seg_count"), 2),
            "has_think_answer": round(rate(output_diverse_stats, "has_think_answer"), 4),
            "has_seg": round(rate(output_diverse_stats, "has_seg"), 4),
            "answer_in_choices": round(rate(output_diverse_stats, "answer_in_choices"), 4),
            "invalid_timestamp_ratio": round(rate(output_diverse_stats, "invalid_timestamps"), 4),
        },
    }

    for t in sorted(type_total.keys()):
        d = type_diverse.get(t, 0)
        c = type_clean.get(t, 0)
        report["by_type"][t] = {
            "total": type_total[t],
            "diverse": d,
            "clean": c,
            "diverse_ratio": round(d / type_total[t], 4) if type_total[t] else 0,
        }

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {REPORT_PATH}")
    print(json.dumps(report["output_stats"], indent=2))
    print(f"\nDiverse: {diverse_count}/{total} = {diverse_count/total:.1%}")
    print(f"Clean:   {clean_count}/{total} = {clean_count/total:.1%}")


if __name__ == "__main__":
    main()

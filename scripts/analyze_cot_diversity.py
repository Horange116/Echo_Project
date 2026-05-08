#!/usr/bin/env python3
"""
CoT Diversity Analysis for eaqa_sft_v9_clean.jsonl (messages format).
Extracts <think> content from assistant responses, no qa_type available.

Usage:
  python3 scripts/analyze_cot_diversity.py \
    --input output/GeneratedData/eaqa_sft_v9_clean.jsonl \
    --report output/judge/v9_clean_24861_cot_diversity_report.json \
    --samples output/judge/v9_clean_24861_cot_diversity_samples.jsonl
"""

import argparse
import json
import re
import random
from collections import Counter, defaultdict

SEG_PATTERN = re.compile(r"<seg>\s*[\d.]+\s*,\s*[\d.]+\s*</seg>")
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOKEN_SPLIT = re.compile(r"[\w']+|[^\w\s]")

# ── helpers ──

def tokenize(text: str) -> list[str]:
    return TOKEN_SPLIT.findall(text)

def get_think_text(text: str) -> str:
    m = THINK_PATTERN.search(text)
    return m.group(1).strip() if m else ""

def normalize_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_assistant_text(item: dict) -> str:
    """Extract full assistant response text from messages."""
    msgs = item.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "").strip()
    return ""

def generalize_template(text: str) -> str:
    text = re.sub(r'\b\d+\.?\d*\b', 'NUM', text)
    text = re.sub(r'<seg>\s*[\d.]+\s*,\s*[\d.]+\s*</seg>', '<seg>SEG</seg>', text)
    return text

def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ── main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect_per_type", type=int, default=3)
    args = parser.parse_args()
    random.seed(args.seed)

    # ── load ──
    print(f"Loading {args.input} ...")
    data = []
    with open(args.input) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"  Total lines: {len(data)}")

    # ── extract CoT ──
    samples = []
    for item in data:
        assistant_text = extract_assistant_text(item)
        if not assistant_text:
            continue
        think = get_think_text(assistant_text)
        if not think:
            continue
        samples.append({
            "id": item.get("id", ""),
            "qa_type": "unknown",
            "full_response": assistant_text,
            "think": think,
            "tokens": tokenize(think),
            "words": think.split(),
        })
    n = len(samples)
    print(f"  Samples with valid CoT: {n}")

    if n == 0:
        print("ERROR: no CoT found, aborting.")
        return

    # ═══ 1. Average lengths ═══
    word_counts = [len(s["words"]) for s in samples]
    token_counts = [len(s["tokens"]) for s in samples]
    char_counts = [len(s["think"]) for s in samples]

    avg_len_metrics = {
        "avg_word_count": round(sum(word_counts) / n, 1),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts),
        "avg_token_count": round(sum(token_counts) / n, 1),
        "min_token_count": min(token_counts),
        "max_token_count": max(token_counts),
        "avg_char_count": round(sum(char_counts) / n, 1),
    }
    print(f"\n  1. Avg word={avg_len_metrics['avg_word_count']}, "
          f"token={avg_len_metrics['avg_token_count']}, "
          f"char={avg_len_metrics['avg_char_count']}")

    # ═══ 2. (skipped: no qa_type) ═══

    # ═══ 3. Template repetition ═══
    templates = Counter()
    for s in samples:
        templates[generalize_template(s["think"])] += 1

    unique_templates = len(templates)
    template_repeat_count = sum(1 for v in templates.values() if v > 1)
    top_templates = templates.most_common(15)
    template_metrics = {
        "unique_templates": unique_templates,
        "template_repeat_rate_pct": round((1 - unique_templates / n) * 100, 1),
        "repeated_template_count": template_repeat_count,
        "repeated_template_rate_pct": round(template_repeat_count / n * 100, 1),
        "top_templates": [
            {"count": cnt, "pct": round(cnt / n * 100, 1), "template": tpl[:200]}
            for tpl, cnt in top_templates
        ],
    }
    print(f"  3. Unique templates: {unique_templates}/{n}, "
          f"repeat_rate={template_metrics['template_repeat_rate_pct']}%")

    # ═══ 4. Opening patterns Top-20 ═══
    openings = Counter()
    for s in samples:
        op_words = generalize_template(s["think"]).split()[:5]
        openings[" ".join(op_words)] += 1

    top_openings = openings.most_common(20)
    # 前 5 集中度
    opening_concentration = round(
        sum(cnt for _, cnt in openings.most_common(5)) / n * 100, 1
    )
    opening_metrics = {
        "unique_openings": len(openings),
        "top5_concentration_pct": opening_concentration,
        "top20_openings": [
            {"count": cnt, "pct": round(cnt / n * 100, 1), "pattern": pat}
            for pat, cnt in top_openings
        ],
    }
    print(f"  4. Unique openings: {len(openings)}, "
          f"top5 concentration={opening_concentration}%")

    # ═══ 5. <seg> follow-up patterns Top-20 ═══
    SEG_PLUS = re.compile(
        r"<seg>\s*[\d.]+\s*,\s*[\d.]+\s*</seg>\s*(.{10,60}?)(?:<seg|$)", re.DOTALL
    )
    seg_followups = Counter()
    for s in samples:
        for m in SEG_PLUS.finditer(s["think"]):
            fu = m.group(1).strip()
            if len(fu) >= 5:
                seg_followups[fu] += 1

    top_seg_followups = seg_followups.most_common(20)
    seg_followup_metrics = {
        "unique_seg_followups": len(seg_followups),
        "top20_seg_followups": [
            {"count": cnt, "text": text[:100]}
            for text, cnt in top_seg_followups
        ],
    }
    print(f"  5. Unique seg follow-ups: {len(seg_followups)}")

    # ═══ 6. n-gram repetition ═══
    ngram_metrics = {}
    for ng in [2, 3, 4, 5]:
        all_ngrams: list[tuple] = []
        for s in samples:
            all_ngrams.extend(get_ngrams(s["tokens"], ng))
        total = len(all_ngrams)
        unique = len(set(all_ngrams))
        counter = Counter(all_ngrams)
        repeated_count = sum(v for v in counter.values() if v > 1)
        most_common = counter.most_common(3)
        ngram_metrics[f"{ng}gram"] = {
            "total": total,
            "unique": unique,
            "unique_rate_pct": round(unique / total * 100, 1) if total else 0,
            "repeat_rate_pct": round((1 - unique / total) * 100, 1) if total else 0,
            "repeated_occurrence_pct": round(repeated_count / total * 100, 1) if total else 0,
            "top3": [" ".join(ng) for ng, _ in most_common],
        }
        print(f"  6. {ng}-gram: unique_rate={ngram_metrics[f'{ng}gram']['unique_rate_pct']}%")

    # ═══ 7. Exact duplicate CoT ═══
    cot_exact = Counter()
    for s in samples:
        cot_exact[s["think"]] += 1

    exact_dup_groups = sum(1 for v in cot_exact.values() if v > 1)
    exact_dup_samples = sum(v for v in cot_exact.values() if v > 1)
    exact_metrics = {
        "unique_cot_count": len(cot_exact),
        "unique_cot_ratio": round(len(cot_exact) / n, 4),
        "duplicate_groups": exact_dup_groups,
        "duplicate_samples": exact_dup_samples,
        "duplicate_samples_pct": round(exact_dup_samples / n * 100, 1),
        "most_repeated": [
            {"count": cnt, "preview": ct[:120]}
            for ct, cnt in cot_exact.most_common(5) if cnt > 1
        ],
    }
    print(f"  7. Exact dup: {exact_dup_groups} groups, {exact_dup_samples} samples")

    # ═══ 8. Approximate duplicate CoT ═══
    cot_norm = Counter()
    for s in samples:
        cot_norm[normalize_text(s["think"])] += 1

    norm_dup_groups = sum(1 for v in cot_norm.values() if v > 1)
    norm_dup_samples = sum(v for v in cot_norm.values() if v > 1)

    approx_groups = 0
    approx_samples = 0
    for norm, cnt in cot_norm.most_common():
        if cnt > 1:
            origs = set()
            for s in samples:
                if normalize_text(s["think"]) == norm:
                    origs.add(s["think"])
            if len(origs) > 1:
                approx_groups += 1
                approx_samples += cnt

    approx_metrics = {
        "norm_unique": len(cot_norm),
        "norm_dup_groups": norm_dup_groups,
        "norm_dup_samples": norm_dup_samples,
        "norm_dup_pct": round(norm_dup_samples / n * 100, 1),
        "approx_same_groups": approx_groups,
        "approx_same_samples": approx_samples,
        "approx_same_pct": round(approx_samples / n * 100, 1) if n else 0,
    }
    print(f"  8. Approx dup: {approx_groups} groups, {approx_samples} samples")

    # ═══ 9. Segment count ═══
    seg_counts = []
    for s in samples:
        seg_counts.append(len(SEG_PATTERN.findall(s["think"])))

    avg_seg = round(sum(seg_counts) / n, 2)
    seg_dist = Counter(seg_counts)
    zero_seg = sum(1 for c in seg_counts if c == 0)
    seg_metrics = {
        "avg_seg_per_sample": avg_seg,
        "max_seg": max(seg_counts),
        "zero_seg_count": zero_seg,
        "zero_seg_pct": round(zero_seg / n * 100, 1),
        "seg_distribution": {
            str(k): {"count": v, "pct": round(v / n * 100, 1)}
            for k, v in sorted(seg_dist.items())
        },
    }
    print(f"  9. Avg seg: {avg_seg}, zero-seg: {zero_seg} ({seg_metrics['zero_seg_pct']}%)")

    # ═══ 10. Natural language before/after seg ═══
    BEFORE_PATTERN = re.compile(r"(\S.{5,40}?)\s*<seg>", re.DOTALL)
    AFTER_PATTERN = re.compile(r"</seg>\s*(.{5,40}?)(?:<seg>|$)", re.DOTALL)

    has_before = has_after = has_both = has_neither = 0
    for s in samples:
        think = s["think"]
        segs = SEG_PATTERN.findall(think)
        if not segs:
            continue
        b = bool(BEFORE_PATTERN.search(think))
        a = bool(AFTER_PATTERN.search(think))
        if b and a:
            has_both += 1
        elif b:
            has_before += 1
        elif a:
            has_after += 1
        else:
            has_neither += 1

    seg_nl_total = has_before + has_after + has_both + has_neither
    seg_nl_metrics = {
        "samples_with_seg": seg_nl_total,
        "explanation_before_and_after": {
            "count": has_both, "pct": round(has_both / seg_nl_total * 100, 1)
        },
        "explanation_before_only": {
            "count": has_before, "pct": round(has_before / seg_nl_total * 100, 1)
        },
        "explanation_after_only": {
            "count": has_after, "pct": round(has_after / seg_nl_total * 100, 1)
        },
        "no_explanation": {
            "count": has_neither, "pct": round(has_neither / seg_nl_total * 100, 1)
        },
    }
    print(f"  10. Seg+NL both={has_both}, before={has_before}, after={has_after}, neither={has_neither}")

    # ═══ 11. Overall diversity (no type breakdown) ═══
    all_tokens = []
    for s in samples:
        all_tokens.extend(s["tokens"])
    ttr = round(len(set(all_tokens)) / len(all_tokens), 4) if all_tokens else 0

    # template diversity
    tpl_diversity = round(unique_templates / n, 4) if n else 0

    # opening diversity (first 3 words)
    type_openings = Counter()
    for s in samples:
        op = generalize_template(s["think"]).split()[:3]
        type_openings[" ".join(op)] += 1
    opening_diversity = round(len(type_openings) / n, 4) if n else 0

    diversity_metrics = {
        "type_token_ratio": ttr,
        "template_diversity_ratio": tpl_diversity,
        "opening_diversity_ratio": opening_diversity,
        "exact_unique_ratio": round(len(cot_exact) / n, 4),
    }
    print(f"  11. TTR={ttr}, template_diversity={tpl_diversity}")

    # ── Build report ──
    report = {
        "data_source": args.input,
        "total_samples": n,
        "avg_lengths": avg_len_metrics,
        "template_repetition": template_metrics,
        "opening_patterns": opening_metrics,
        "seg_followup_patterns": seg_followup_metrics,
        "ngram_repetition": ngram_metrics,
        "exact_duplicates": exact_metrics,
        "approximate_duplicates": approx_metrics,
        "segment_counts": seg_metrics,
        "segment_context_explanation": seg_nl_metrics,
        "overall_diversity": diversity_metrics,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {args.report}")

    # ── Write inspection samples ──
    with open(args.samples, "w", encoding="utf-8") as f:
        for s in random.sample(samples, min(30, n)):
            out = {
                "id": s["id"],
                "cot": s["full_response"],
                "think_len": len(s["think"]),
                "seg_count": len(SEG_PATTERN.findall(s["think"])),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Samples saved: {args.samples}")

    # ── Summary comparison with 13K ──
    print("\n" + "=" * 70)
    print("SUMMARY vs 13K analysis")
    print("=" * 70)
    print(f"""
  Metric                   13K (n=13000)     24K (n={n})
  ─────────────────────    ─────────────     ────────────
  avg word count           40.6               {avg_len_metrics['avg_word_count']}
  avg token count          78.7               {avg_len_metrics['avg_token_count']}
  avg char count           261.7              {avg_len_metrics['avg_char_count']}
  avg seg count            1.74               {avg_seg}
  template repeat rate     43.4%              {template_metrics['template_repeat_rate_pct']}%
  top opening conc.        38.1% (top4)       {opening_concentration}% (top5)
  ngram 5-gram unique      28.5%              {ngram_metrics['5gram']['unique_rate_pct']}%
  exact_dup (samples)      0.1%               {exact_metrics['duplicate_samples_pct']}%
  seg+before explanation   69.0%              {seg_nl_metrics['explanation_before_only']['pct']}%
  seg+both explanations    4.7%               {seg_nl_metrics['explanation_before_and_after']['pct']}%
  seg no explanation       26.2%              {seg_nl_metrics['no_explanation']['pct']}%
  TTR                      -                  {ttr}
""")


if __name__ == "__main__":
    main()

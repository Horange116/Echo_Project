#!/usr/bin/env python3
"""
Diagnose gibberish responses in base model MMAR evaluation — refined version.

Refined gibberish classification:
1. pérdida_loop: "pérdida" >= 5 occurrences
2. dominating_word: single word > 50% of total tokens
3. excessive_repeat: non-function word > 30x
4. number_loop: digit sequence >= 20x
5. char_repeat: character repeated >= 10x consecutively

Output comprehensive report with audio analysis.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16000

FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'to', 'of', 'in', 'at', 'on', 'and', 'or', 'for', 'with',
    'it', 'that', 'this', 'be', 'are', 'was', 'were', 'been', 'by', 'as', 'from',
    'not', 'no', 'yes', 'so', 'if', 'but', 'has', 'have', 'had', 'do', 'does', 'did',
    'can', 'will', 'would', 'could', 'should', 'may', 'might', 'shall',
    'we', 'he', 'she', 'they', 'them', 'their', 'his', 'her', 'my', 'your', 'our', 'its',
    'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
    'also', 'very', 'just', 'all', 'each', 'every', 'some', 'any', 'more', 'most',
    '0', '00', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    's', 't', 're', 've', 'll', 'm', 'd',
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify_gibberish(text):
    """Refined gibberish classifier. Returns (is_gibberish, reason)."""
    text_lower = text.lower()

    # 1. pérdida loop
    perdida = len(re.findall(r'pérdida', text_lower))
    if perdida >= 5:
        return True, f"pérdida_loop({perdida}x)"

    words = re.findall(r'\w+', text_lower)
    total = len(words)
    if total == 0:
        return True, "empty_response"

    wc = Counter(words)
    top_word, top_cnt = wc.most_common(1)[0]
    top_ratio = top_cnt / total

    # 2. Single word dominates > 50% of response
    if top_ratio > 0.5:
        return True, f"dominating_word({top_word}:{top_cnt}/{total}={top_ratio:.0%})"

    # 3. Non-function word repeated > 30x (content word loop)
    if top_cnt >= 30 and top_word not in FUNCTION_WORDS:
        return True, f"excessive_repeat({top_word}:{top_cnt}x/{total}w,ratio={top_ratio:.0%})"

    # 4. Numerical repetition
    digits = re.findall(r'\d+', text_lower)
    if digits:
        digit_wc = Counter(digits)
        top_digit, top_dcnt = digit_wc.most_common(1)[0]
        if top_dcnt >= 20 and len(top_digit) <= 3:
            return True, f"number_loop({top_digit}:{top_dcnt}x)"

    # 5. Character repetition
    if re.search(r'(.)\1{9,}', text):
        return True, "char_repeat"

    return False, "normal"


def compute_audio_stats(audio_path):
    """Compute RMS energy and duration of an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            return None, None
        rms = float(np.sqrt(np.mean(y ** 2)))
        duration = float(len(y) / sr)
        return rms, duration
    except Exception:
        return None, None


def stats(arr):
    if not arr:
        return {"min": None, "mean": None, "max": None, "count": 0}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def main():
    parser = argparse.ArgumentParser(description="Gibberish diagnostic for MMAR predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output", default="gibberish_report.json")
    args = parser.parse_args()

    # Build audio_url lookup from test_jsonl
    test_data = {}
    for item in read_jsonl(args.test_jsonl):
        test_data[item["id"]] = item

    # Analyze predictions
    predictions = list(read_jsonl(args.predictions))
    total = len(predictions)

    gibberish_items = []
    normal_items = []
    gib_categories = Counter()
    gib_modalities = Counter()
    normal_modalities = Counter()
    gib_categories_full = Counter()

    for pred in predictions:
        resp = pred.get("raw_response", "")
        is_gib, reason = classify_gibberish(resp)

        test_item = test_data.get(pred["id"], {})
        audio_url = test_item.get("audio_url", "")
        audio_path = os.path.join(args.audio_dir, audio_url)
        rms, duration = compute_audio_stats(audio_path)

        mod = pred.get("modality", "unknown")
        cat = pred.get("category", "unknown")

        entry = {
            "id": pred["id"],
            "category": cat,
            "modality": mod,
            "question": pred.get("question", "")[:200],
            "choices": pred.get("choices", []),
            "gold_answer": pred.get("gold_answer", ""),
            "raw_response": resp[:300],
            "gibberish_reason": reason,
            "audio_url": audio_url,
            "duration": duration,
            "rms": rms,
            "has_answer_tag": pred.get("has_answer_tag", False),
            "response_len": len(resp),
        }

        if is_gib:
            gibberish_items.append(entry)
            gib_categories_full[reason] += 1
            gib_modalities[mod] += 1
            gib_categories[cat] += 1
        else:
            normal_items.append(entry)

    gib_count = len(gibberish_items)
    nor_count = len(normal_items)

    # Audio stats
    gib_durations = [it["duration"] for it in gibberish_items if it["duration"] is not None]
    nor_durations = [it["duration"] for it in normal_items if it["duration"] is not None]
    gib_rms = [it["rms"] for it in gibberish_items if it["rms"] is not None]
    nor_rms = [it["rms"] for it in normal_items if it["rms"] is not None]

    # Top 20 gibberish by RMS (quietest first — likely empty/silent audio)
    top20_gibberish = sorted(
        [it for it in gibberish_items if it["rms"] is not None],
        key=lambda x: x["rms"],
    )[:20]

    # Build report
    report = {
        "total": total,
        "gibberish_count": gib_count,
        "gibberish_rate": round(gib_count / max(1, total), 4),
        "gibberish_by_type": dict(gib_categories_full.most_common()),
        "gibberish_by_modality": {
            k: {
                "count": v,
                "total": v + normal_modalities.get(k, 0),
                "rate": round(v / max(1, v + normal_modalities.get(k, 0)), 4),
            }
            for k, v in sorted(gib_modalities.items(), key=lambda x: -x[1])
        },
        "gibberish_by_category": {
            k: {
                "count": v,
                "total": v + sum(1 for it in normal_items if it["category"] == k),
                "rate": round(v / max(1, v + sum(1 for it in normal_items if it["category"] == k)), 4),
            }
            for k, v in sorted(gib_categories.items(), key=lambda x: -x[1])
        },
        "audio_duration": {
            "gibberish": stats(gib_durations),
            "normal": stats(nor_durations),
        },
        "audio_rms": {
            "gibberish": stats(gib_rms),
            "normal": stats(nor_rms),
        },
        "top20_gibberish_by_rms": [
            {
                "id": it["id"],
                "category": it["category"],
                "modality": it["modality"],
                "reason": it["gibberish_reason"],
                "rms": it["rms"],
                "duration": it["duration"],
                "response_len": it["response_len"],
                "has_answer_tag": it["has_answer_tag"],
                "gold_answer": it["gold_answer"],
                "response_preview": it["raw_response"][:150],
            }
            for it in top20_gibberish
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Gibberish Diagnostic Report")
    print(f"{'='*60}")
    print(f"Total:       {total}")
    print(f"Gibberish:   {gib_count} ({gib_count/max(1,total)*100:.1f}%)")
    print(f"Normal:      {nor_count} ({nor_count/max(1,total)*100:.1f}%)")
    print()
    print("By type:")
    for k, v in gib_categories_full.most_common(10):
        print(f"  {k}: {v}")
    if len(gib_categories_full) > 10:
        print(f"  ... and {len(gib_categories_full) - 10} more")
    print()
    print("By modality:")
    for k, v in sorted(gib_modalities.items(), key=lambda x: -x[1]):
        t = v + sum(1 for it in normal_items if it["modality"] == k)
        print(f"  {k}: {v}/{t} ({v/max(1,t)*100:.1f}%)")
    print()
    print("By category:")
    for k, v in sorted(gib_categories.items(), key=lambda x: -x[1]):
        t = v + sum(1 for it in normal_items if it["category"] == k)
        print(f"  {k}: {v}/{t} ({v/max(1,t)*100:.1f}%)")
    print()
    print("Audio duration (seconds):")
    print(f"  Gibberish: {stats(gib_durations)}")
    print(f"  Normal:    {stats(nor_durations)}")
    print()
    print("Audio RMS energy:")
    print(f"  Gibberish: {stats(gib_rms)}")
    print(f"  Normal:    {stats(nor_rms)}")
    print()
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()

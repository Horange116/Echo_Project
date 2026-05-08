#!/usr/bin/env python3
"""
Run the 5 targeted smoke test samples with a single model load.
Produces individual result files + comparison table.
"""

import json
import os
import sys
import time
from datetime import datetime

from interleaved_infer import load_model_and_processor, run_interleaved

# ── 5 targeted samples ──
SAMPLES = [
    {
        "name": "01_gap",
        "audio_path": "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--7UmfOkRbM_30000.flac",
        "question": "How long after the first male speech ends does the first human voice begin?",
        "choices": ["0.4 seconds", "0.7 seconds", "0.1 seconds", "0 seconds"],
        "gold_answer": "0.1 seconds",
        "qa_type": "gap",
    },
    {
        "name": "02_count_before",
        "audio_path": "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--BIwg9KRxI_130000.flac",
        "question": "How many timestamped sound events have finished before the third noise begins?",
        "choices": ["1", "3", "2", "4"],
        "gold_answer": "2",
        "qa_type": "count_before",
    },
    {
        "name": "03_repeated_event_gap",
        "audio_path": "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--CHY2qO5zc_30000.flac",
        "question": "How much time passes between the end of the first alarm clock and the start of the second alarm clock?",
        "choices": ["0.1 seconds", "0.4 seconds", "1.0 second", "0.7 seconds"],
        "gold_answer": "0.1 seconds",
        "qa_type": "repeated_event_gap",
    },
    {
        "name": "04_duration_compare",
        "audio_path": "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--EnKcYsPas_210000.flac",
        "question": "Which event lasts longer, the cough or the whispering?",
        "choices": ["the cough", "the whispering", "they last the same amount of time", "neither sound is audible"],
        "gold_answer": "the whispering",
        "qa_type": "duration_compare",
    },
    {
        "name": "05_gap",
        "audio_path": "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/train-00000-of-00216_9be12406/audios/--N8Xc-3C3k_30000.flac",
        "question": "How long after the first male speech ends does the impact sounds begin?",
        "choices": ["1.0 second", "0.1 seconds", "0.7 seconds", "0.4 seconds"],
        "gold_answer": "",
        "qa_type": "gap",
    },
]

# ── new inference params ──
INFER_KWARGS = dict(
    max_rounds=5,
    max_new_tokens_per_round=128,
    temperature=0.7,
    tmp_dir="output/interleaved_tmp",
    duplicate_iou_threshold=0.8,
    max_duplicate_segments=1,
    on_duplicate_seg="stop",
    finalize_on_stop=True,
    finalize_max_new_tokens=64,
)


def main():
    model_path = sys.argv[1]
    adapter_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "None" else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "output/interleaved/targeted_five"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output/interleaved_tmp", exist_ok=True)

    # Load model once
    print(f"[{datetime.now()}] 加载模型: {model_path}")
    t0 = time.time()
    model, processor = load_model_and_processor(model_path, adapter_path)
    print(f"[{datetime.now()}] 模型加载完成 ({time.time()-t0:.1f}s), device: {model.device}")

    results = []

    for i, s in enumerate(SAMPLES):
        print(f"\n{'='*60}")
        print(f"[{datetime.now()}] [{i+1}/{len(SAMPLES)}] {s['name']} ({s['qa_type']})")
        print(f"  问题: {s['question']}")
        print(f"  音频: {os.path.basename(s['audio_path'])}")
        print(f"{'='*60}")

        t1 = time.time()
        result = run_interleaved(
            model, processor,
            audio_path=s["audio_path"],
            question=s["question"],
            choices=s["choices"],
            gold_answer=s["gold_answer"],
            **INFER_KWARGS,
        )
        elapsed = time.time() - t1

        result["name"] = s["name"]
        result["qa_type"] = s["qa_type"]
        result["elapsed"] = round(elapsed, 1)
        results.append(result)

        # Write individual result
        out_path = os.path.join(output_dir, f"{s['name']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  → {elapsed:.1f}s | stop={result['stop_reason']} | "
              f"rounds={result['total_rounds']} | insert={result['num_inserted_segments']} | "
              f"dup={result['num_duplicate_segments']} | "
              f"ans={'✓' if result['has_final_answer'] else '✗'} | "
              f"pred={result['pred_answer'] or '-'}")

    # ── Comparison table ──
    print(f"\n{'='*90}")
    print(f"  Comparison Table — Targeted v2 (with duplicate protection + finalize)")
    print(f"{'='*90}")
    print(f"{'sample':20s} {'qa_type':22s} {'trig':6s} {'ins':6s} {'dup':6s} {'rounds':6s} "
          f"{'stop_reason':20s} {'answer':10s} {'pred':12s} {'correct':8s}")
    print("-" * 110)

    for r in results:
        trig = "✓" if r["triggered_interleaved"] else "✗"
        ha = "✓" if r["has_final_answer"] else "✗"
        corr = r["answer_correct"]
        corr_s = "✓" if corr is True else ("✗" if corr is False else "-")

        print(f"{r['name']:20s} {r['qa_type']:22s} {trig:6s} "
              f"{r['num_inserted_segments']:6d} {r['num_duplicate_segments']:6d} "
              f"{r['total_rounds']:6d} {r['stop_reason']:20s} {ha:10s} "
              f"{(r['pred_answer'] or '-'):12s} {corr_s:8s}")

    print("-" * 110)

    # Summary of key improvements
    dup_saved = sum(1 for r in results if r['stop_reason'] == 'duplicate_seg')
    finalized = sum(1 for r in results if 'finalize' in str(r['stop_reason']))
    has_ans = sum(1 for r in results if r['has_final_answer'])
    print(f"\n  改进效果:")
    print(f"    - duplicate 检测触发: {dup_saved}/5 (原本会无限循环)")
    print(f"    - finalize 轮触发: {finalized}/5")
    print(f"    - 有最终答案: {has_ans}/5")

    # Write aggregated result
    agg_path = os.path.join(output_dir, "_summary.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "num_samples": len(results),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果目录: {output_dir}")


if __name__ == "__main__":
    main()

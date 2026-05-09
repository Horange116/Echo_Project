"""
Strategy comparison: evaluate 4 interleaved strategies with rollout_reward.

Reads:
  - output/interleaved_eval/strategy_compare/A_ignore_no_final_smoke20.jsonl
  - output/interleaved_eval/strategy_compare/B_stop_finalize_smoke20.jsonl
  - output/interleaved_eval/strategy_compare/C_ignore_finalize_smoke20.jsonl
  - output/interleaved_eval/strategy_compare/D_insert_once_finalize_smoke20.jsonl

Computes rollout_reward per sample and produces comparison tables.
"""

import json
import os
import re
import sys
from collections import Counter
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from echo_rl.rollout_rewards import rollout_reward, rollout_reward_report

STRATEGIES = {
    "A_ignore_no_final": {
        "label": "A: ignore+no_final",
        "desc": "No duplicate guard, no finalize (≈ basic)",
    },
    "B_stop_finalize": {
        "label": "B: stop+finalize",
        "desc": "Stop on first dup + finalize (current default)",
    },
    "C_ignore_finalize": {
        "label": "C: ignore+finalize",
        "desc": "Full multi-round + finalize at max_rounds",
    },
    "D_insert_once_finalize": {
        "label": "D: insert_once+finalize",
        "desc": "Insert each seg once + finalize at max_rounds",
    },
}

DATA_DIR = "output/interleaved_eval/strategy_compare"


def avg(vals):
    return round(mean(vals), 3) if vals else 0.0


def load_results(name):
    path = os.path.join(DATA_DIR, f"{name}_smoke20.jsonl")
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_rollout(r):
    resp = r["final_response"]
    gt = r["ground_truth"]
    used_segs = r.get("used_segments", [])
    unique_seg_count = len(set((s["start"], s["end"]) for s in used_segs))
    total_inserted = len(used_segs)
    dup_count = total_inserted - unique_seg_count
    rounds = r.get("num_rounds", 0)

    meta = {
        "triggered_interleaved": total_inserted > 0,
        "inserted_segments": used_segs,
        "duplicate_seg_count": dup_count,
        "unique_segment_count": unique_seg_count,
        "round_count": rounds,
        "finalize_triggered": r.get("fully_structured", False),
        "stop_reason": r.get("stop_reason", "unknown"),
    }

    rew = rollout_reward(resp, gt, meta)
    rew["_correct"] = r["is_correct"]
    rew["_pred"] = r.get("pred_answer", "")
    rew["_id"] = r["id"]
    rew["_rounds"] = rounds
    rew["_unique_segs"] = unique_seg_count
    rew["_dup_segs"] = dup_count
    rew["_fully_structured"] = r.get("fully_structured", False)
    rew["_has_answer"] = bool(r.get("pred_answer", ""))
    rew["_stop_reason"] = r.get("stop_reason", "?")
    return rew


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

strategy_rewards = {}

for name, info in STRATEGIES.items():
    results = load_results(name)
    rewards = [compute_rollout(r) for r in results]
    strategy_rewards[name] = {
        "info": info,
        "results": results,
        "rewards": rewards,
    }

# ── Summary table ──
print("=" * 90)
print("  INTERLEAVED STRATEGY COMPARISON — v9b-diverse-cot-2epoch")
print("=" * 90)

for name, data in strategy_rewards.items():
    r = data["rewards"]
    info = data["info"]
    n_correct = sum(1 for x in r if x["_correct"])
    n = len(r)
    ba = avg([x["total"] for x in r])
    ra = avg([x["rollout_total"] for x in r])
    avg_rounds = avg([x["_rounds"] for x in r])
    avg_unique = avg([x["_unique_segs"] for x in r])
    avg_dup = avg([x["_dup_segs"] for x in r])
    n_answer = sum(1 for x in r if x["_has_answer"])
    n_finalized = sum(1 for x in r if x["_fully_structured"])

    print(f"\n{'─'*90}")
    print(f"  {info['label']:30s}  {info['desc']}")
    print(f"{'─'*90}")
    print(f"  {'Accuracy':30s}  {n_correct}/{n} = {n_correct/n*100:.1f}%")
    print(f"  {'Has answer':30s}  {n_answer}/{n}")
    print(f"  {'Finalized (fully_structured)':30s}  {n_finalized}/{n}")
    print(f"  {'Avg rounds':30s}  {avg_rounds:.1f}")
    print(f"  {'Avg unique segs':30s}  {avg_unique:.1f}")
    print(f"  {'Avg dup segs':30s}  {avg_dup:.1f}")
    print(f"  {'Base total (avg)':30s}  {ba:+.3f}")
    print(f"  {'Rollout total (avg)':30s}  {ra:+.3f}")
    print(f"  {'Delta':30s}  {ra-ba:+.3f}")

    # Per-component breakdown
    print(f"  {'─'*40}")
    for comp in ("format", "consistency", "accuracy", "segment", "duplicate_penalty", "round_penalty", "finalize_penalty", "unique_segment_bonus"):
        c_avg = avg([x.get(comp, 0) for x in r])
        print(f"  {comp:30s}  {c_avg:+.3f}")

# ── Cross-strategy comparison ──
print(f"\n{'='*90}")
print("  CROSS-STRATEGY COMPARISON")
print(f"{'='*90}")
header = f"  {'Strategy':25s}  {'Acc':>6s}  {'Rounds':>6s}  {'Uniq':>5s}  {'Dup':>5s}  {'Base':>7s}  {'Rollout':>8s}  {'Delta':>6s}"
print(header)
print("  " + "-" * 75)

for name, data in strategy_rewards.items():
    r = data["rewards"]
    info = data["info"]
    n_correct = sum(1 for x in r if x["_correct"])
    n = len(r)
    ba = avg([x["total"] for x in r])
    ra = avg([x["rollout_total"] for x in r])
    avg_rounds = avg([x["_rounds"] for x in r])
    avg_unique = avg([x["_unique_segs"] for x in r])
    avg_dup = avg([x["_dup_segs"] for x in r])
    print(f"  {info['label']:25s}  {n_correct/n*100:>5.1f}%  {avg_rounds:>5.1f}  {avg_unique:>4.1f}  {avg_dup:>4.1f}  {ba:>+6.3f}  {ra:>+7.3f}  {ra-ba:>+5.3f}")

# ── Correct-only comparison ──
print(f"\n{'─'*40}  CORRECT SAMPLES ONLY  {'─'*40}")
header_c = f"  {'Strategy':25s}  {'n':>3s}  {'Base':>7s}  {'Rollout':>8s}  {'Delta':>6s}  {'Rounds':>6s}  {'Uniq':>5s}"
print(header_c)
print("  " + "-" * 65)

for name, data in strategy_rewards.items():
    r = [x for x in data["rewards"] if x["_correct"]]
    info = data["info"]
    if not r:
        print(f"  {info['label']:25s}  {'—':>3s}  {'—':>7s}  {'—':>8s}  {'—':>6s}  {'—':>6s}  {'—':>5s}")
        continue
    ba = avg([x["total"] for x in r])
    ra = avg([x["rollout_total"] for x in r])
    avg_rounds = avg([x["_rounds"] for x in r])
    avg_unique = avg([x["_unique_segs"] for x in r])
    print(f"  {info['label']:25s}  {len(r):>3d}  {ba:>+6.3f}  {ra:>+7.3f}  {ra-ba:>+5.3f}  {avg_rounds:>5.1f}  {avg_unique:>4.1f}")

print("\nDone.")

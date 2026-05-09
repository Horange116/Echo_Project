"""
Reward dry run: compute rollout_reward on real eval outputs.

Reads:
  - output/interleaved_eval/v9b_2epoch_smoke20.jsonl   (full version)
  - output/interleaved_eval/original_smoke20/batch_result.json (basic version)

Derives rollout_metadata from available fields and compares
base_total vs rollout_total across versions and sample subgroups.
"""

import json
import re
import sys
from collections import Counter
from statistics import mean

from echo_rl.rollout_rewards import rollout_reward, rollout_reward_report

SEG_TAG = re.compile(r"<seg>\s*([\d.]+)\s*,\s*([\d.]+)\s*</seg>")


def avg(vals):
    return round(mean(vals), 3) if vals else 0.0


# ── load ──

full_results = []
with open("output/interleaved_eval/v9b_2epoch_smoke20.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            full_results.append(json.loads(line))

with open("output/interleaved_eval/original_smoke20/batch_result.json") as f:
    basic_data = json.load(f)
basic_results = basic_data["results"]

print("=" * 72)
print("  REWARD DRY RUN")
print("=" * 72)

# ════════════════════════════════════════════
#  FULL VERSION (duplicate guard + finalize)
# ════════════════════════════════════════════

print("\n--- Full version (duplicate guard + finalize) ---")

full_rewards = []
for r in full_results:
    resp = r["final_response"]
    gt = r["ground_truth"]

    seg_tags = SEG_TAG.findall(resp)
    approx_rounds = len(seg_tags)
    used_segs = r.get("used_segments", [])
    unique_seg_count = len(set((s["start"], s["end"]) for s in used_segs))
    dup_est = max(0, approx_rounds - unique_seg_count)
    finalized = r.get("fully_structured", False)

    meta = {
        "triggered_interleaved": unique_seg_count > 0,
        "inserted_segments": used_segs,
        "duplicate_seg_count": dup_est,
        "unique_segment_count": unique_seg_count,
        "round_count": approx_rounds,
        "finalize_triggered": finalized,
        "stop_reason": "duplicate_seg" if dup_est > 0 else "answer",
    }

    rew = rollout_reward(resp, gt, meta)
    rew["_correct"] = r["is_correct"]
    rew["_pred"] = r.get("pred_answer", "")
    rew["_id"] = r["id"]
    full_rewards.append(rew)


def subgroup_stats(rewards, name):
    bt = [r["total"] for r in rewards]
    rt = [r["rollout_total"] for r in rewards]
    print(f"  {name:30s}  n={len(rewards):2d}  base={avg(bt):+.3f}  rollout={avg(rt):+.3f}  diff={avg(rt)-avg(bt):+.3f}")


subgroup_stats(full_rewards, "All")
subgroup_stats([r for r in full_rewards if r["_correct"]], "Correct")
subgroup_stats([r for r in full_rewards if not r["_correct"]], "Wrong")
subgroup_stats([r for r in full_rewards if r.get("finalize_triggered")], "Finalize=True")
subgroup_stats([r for r in full_rewards if not r.get("finalize_triggered")], "Finalize=False")
subgroup_stats([r for r in full_rewards if r.get("duplicate_penalty", 0) < 0], "Dup>0")
subgroup_stats([r for r in full_rewards if r.get("duplicate_penalty", 0) == 0], "Dup=0")

# ════════════════════════════════════════════
#  BASIC VERSION (no duplicate guard)
# ════════════════════════════════════════════

print("\n--- Basic version (no duplicate guard, no finalize) ---")

basic_rewards = []
for r in basic_results:
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
        "finalize_triggered": False,
        "stop_reason": "max_rounds" if rounds >= 5 else "answer",
    }

    rew = rollout_reward(resp, gt, meta)
    rew["_correct"] = r["is_correct"]
    rew["_pred"] = r.get("final_answer", "")
    rew["_id"] = r.get("skeleton_id", "?")
    basic_rewards.append(rew)

subgroup_stats(basic_rewards, "All")
subgroup_stats([r for r in basic_rewards if r["_correct"]], "Correct")
subgroup_stats([r for r in basic_rewards if not r["_correct"]], "Wrong")
subgroup_stats([r for r in basic_rewards if r.get("round_penalty", 0) < 0], "Rounds>5")
subgroup_stats([r for r in basic_rewards if r.get("round_penalty", 0) == 0], "Rounds<=5")

# ════════════════════════════════════════════
#  CROSS-VERSION COMPARISON
# ════════════════════════════════════════════

print("\n--- Cross-version comparison ---")
header = f"  {'Split':30s}  {'Base avg':>9s}  {'Rollout avg':>11s}  {'Delta':>7s}"
print(header)
print("  " + "-" * 60)

for label, rlist in [
    ("Full (all)", full_rewards),
    ("  correct", [r for r in full_rewards if r["_correct"]]),
    ("  wrong", [r for r in full_rewards if not r["_correct"]]),
    ("Basic (all)", basic_rewards),
    ("  correct", [r for r in basic_rewards if r["_correct"]]),
    ("  wrong", [r for r in basic_rewards if not r["_correct"]]),
]:
    ba = avg([r["total"] for r in rlist])
    ra = avg([r["rollout_total"] for r in rlist])
    print(f"  {label:30s}  {ba:>+8.3f}    {ra:>+8.3f}    {ra-ba:>+6.3f}")

# ════════════════════════════════════════════
#  BEST / WORST EXAMPLES
# ════════════════════════════════════════════

print("\n--- Full version: worst 3 ---")
for r in sorted(full_rewards, key=lambda x: x["rollout_total"])[:3]:
    print(f"  {rollout_reward_report('', GT_CORRECT if '_' not in str(r.get('_id')) else '', r)}  id={r['_id'][:50]}")

print("\n--- Full version: best 3 ---")
for r in sorted(full_rewards, key=lambda x: x["rollout_total"])[-3:]:
    print(f"  {rollout_reward_report('', GT_CORRECT if '_' not in str(r.get('_id')) else '', r)}  id={r['_id'][:50]}")

print("\n--- Basic version: worst 3 ---")
for r in sorted(basic_rewards, key=lambda x: x["rollout_total"])[:3]:
    print(f"  {rollout_reward_report('', '', r)}  id={r['_id'][:50]}")

print("\n--- Basic version: best 3 ---")
for r in sorted(basic_rewards, key=lambda x: x["rollout_total"])[-3:]:
    print(f"  {rollout_reward_report('', '', r)}  id={r['_id'][:50]}")

print("\nDone.")

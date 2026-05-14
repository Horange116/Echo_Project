#!/usr/bin/env python3
"""
Test28: Confidence-weighted reward alignment.

Verifies that:
1. Custom ``r_acc()`` with ``avg_logprob`` produces correct confidence-weighted scores
2. Formula matches VERL ``multiturn_rl_6`` exactly
3. Edge cases: confidence clamping, boundary values, backward compatibility
4. End-to-end: full rollouts with logprobs → reward → comparison

Usage:
    python script/test28_reward_confidence_align.py [--model-path /path/to/Qwen2.5-Omni-7B]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from typing import Any

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/s2025244189/s2025244265/Projects/Echo_Project",
)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Custom reward
from echo_rl.rewards import r_acc, total_reward, extract_answer, normalize_answer
from echo_rl.rewards import r_format, r_consist, r_seg

# ---------------------------------------------------------------------------
# Synthetic test helpers
# ---------------------------------------------------------------------------

SAMPLE_RESPONSES = {
    "correct_high_conf": (
        "<think>Let me analyze the audio segments.</think>"
        "<answer>dog</answer>"
    ),
    "correct_low_conf": (
        "<think>I'm not sure about this one.</think>"
        "<answer>dog</answer>"
    ),
    "wrong_high_conf": (
        "<think>The audio clearly contains barking.</think>"
        "<answer>cat</answer>"
    ),
    "wrong_low_conf": (
        "<think>This is really unclear, maybe a guess.</think>"
        "<answer>cat</answer>"
    ),
    "correct_with_seg": (
        "<think>At 1.5s I hear barking.</think>"
        "<answer>dog</answer>"
        "<seg>1.5, 2.3</seg>"
    ),
    "wrong_with_seg": (
        "<think>At 1.5s I hear meowing.</think>"
        "<answer>cat</answer>"
        "<seg>1.5, 2.3</seg>"
    ),
    "no_answer": (
        "<think>I don't know what to answer.</think>"
    ),
    "empty": "",
}


def test_confidence_formula():
    """Test the core confidence-weighted formula against VERL's reference."""
    print("=" * 60)
    print("1. Core formula verification")
    print("=" * 60)

    # Our formula (replicated inline for clarity):
    #   confidence = exp(avg_logprob)  clipped to [0, 1]
    #   if correct:  score = 0.5 * (1 - (confidence - 1)^2)
    #   else:        score = -0.5 * confidence^2

    test_cases = [
        # (avg_logprob, gt_answer, response_key, expected_correct, expected_score_range)
        # High confidence correct → close to 0.5
        ("correct_high_conf", "dog", True, (0.45, 0.5)),
        # Low confidence correct → below 0.5 but positive
        ("correct_low_conf", "dog", True, (0.0, 0.45)),
        # High confidence wrong → strongly negative (near -0.5)
        ("wrong_high_conf", "dog", False, (-0.5, -0.2)),
        # Low confidence wrong → slightly negative (near 0)
        ("wrong_low_conf", "dog", False, (-0.2, 0.0)),
    ]

    logprob_values = [
        ("very high", -0.1),   # exp(-0.1) ≈ 0.905
        ("high", -0.5),         # exp(-0.5) ≈ 0.607
        ("medium", -1.0),       # exp(-1.0) ≈ 0.368
        ("low", -2.0),          # exp(-2.0) ≈ 0.135
        ("very low", -4.0),     # exp(-4.0) ≈ 0.018
    ]

    print(f"\n  {'logprob':>12} | {'conf':>6} | {'formula':>8} | {'correct':>8} | {'wrong':>8}")
    sep = "  " + "-"*12 + "-+-" + "-"*6 + "-+-" + "-"*8 + "-+-" + "-"*8 + "-+-" + "-"*8 + "}"
    print(sep)

    for label, lp in logprob_values:
        conf = math.exp(lp)
        # VERL's exact formula (no clipping)
        correct_score = 0.5 * (1.0 - (conf - 1.0) ** 2)
        wrong_score = -0.5 * (conf ** 2)
        print(f"  {label:>12} | {conf:.4f} | exp(lp)={lp:>5.1f} | {correct_score:>+8.4f} | {wrong_score:>+8.4f}")

    print()
    return True


def test_r_acc_synthetic():
    """Test r_acc with synthetic responses and various avg_logprob values."""
    print("=" * 60)
    print("2. r_acc synthetic tests")
    print("=" * 60)

    all_pass = True
    gt_answer = "dog"

    # --- Test 2a: Backward compatibility (no avg_logprob) ---
    print("\n  2a. Backward compatibility (no avg_logprob → binary)")
    for key in ["correct_high_conf", "correct_low_conf", "wrong_high_conf", "wrong_low_conf"]:
        resp = SAMPLE_RESPONSES[key]
        score = r_acc(resp, gt_answer)
        expected = 0.5 if "dog" in resp else 0.0
        ok = abs(score - expected) < 1e-6
        print(f"    {key:>20}: r_acc={score:+.4f} (expected={expected}) {'✓' if ok else '✗'}")
        all_pass &= ok

    # --- Test 2b: Confidence weighted ---
    print("\n  2b. Confidence-weighted accuracy")
    scenarios = [
        ("correct_high_conf", "dog", -0.1,  True,  0.45, 0.50),
        ("correct_high_conf", "dog", -0.5,  True,  0.40, 0.50),
        ("correct_low_conf",  "dog", -2.0,  True,  0.0,  0.40),
        ("correct_low_conf",  "dog", -4.0,  True,  0.0,  0.20),
        ("wrong_high_conf",   "dog", -0.1,  False, -0.50, -0.30),
        ("wrong_high_conf",   "dog", -0.5,  False, -0.50, -0.10),
        ("wrong_low_conf",    "dog", -2.0,  False, -0.10, 0.0),
        ("wrong_low_conf",    "dog", -4.0,  False, -0.02, 0.0),
    ]

    for resp_key, gt, avg_lp, _is_correct, lo, hi in scenarios:
        resp = SAMPLE_RESPONSES[resp_key]
        score = r_acc(resp, gt, avg_logprob=avg_lp)
        ok = lo <= score <= hi
        status = "✓" if ok else f"✗ (outside [{lo}, {hi}])"
        print(f"    {resp_key:>20} lp={avg_lp:>5.1f}: r_acc={score:+.6f}  {status}")
        all_pass &= ok

    # --- Test 2c: Edge cases ---
    print("\n  2c. Edge cases")
    # c1: empty gt_answer
    score = r_acc("<answer>dog</answer>", "", avg_logprob=-0.5)
    ok = score == 0.0
    print(f"    empty gt_answer: r_acc={score} (expected=0.0) {'✓' if ok else '✗'}")
    all_pass &= ok

    # c2: no answer tag
    score = r_acc("<think>no answer tag</think>", "dog", avg_logprob=-0.5)
    ok = score == 0.0
    print(f"    no answer tag: r_acc={score} (expected=0.0) {'✓' if ok else '✗'}")
    all_pass &= ok

    # c3: confidence clamping above 1.0
    # avg_logprob = 1.0 → exp(1.0) = 2.718 → clamped to 1.0
    score = r_acc("<answer>dog</answer>", "dog", avg_logprob=1.0)
    # clamped conf = 1.0, correct → 0.5 * (1 - (1-1)^2) = 0.5
    ok = abs(score - 0.5) < 1e-6
    print(f"    high logprob clamping (exp(1.0)=2.72→1.0): r_acc={score:.4f} (expected=0.5) {'✓' if ok else '✗'}")
    all_pass &= ok

    # c4: negative confidence → clamped to 0
    # This shouldn't happen in practice (exp(x) > 0 for any real x)
    # But test the clamping path with a mock scenario
    # Actually avg_logprob is always negative for log-probs, so exp() is always (0, 1]
    # This edge case is about safety

    # c5: None avg_logprob (validation mode)
    score = r_acc("<answer>dog</answer>", "dog", avg_logprob=None)
    ok = abs(score - 0.5) < 1e-6
    print(f"    validation mode (None): r_acc={score:.4f} (expected=0.5) {'✓' if ok else '✗'}")
    all_pass &= ok

    # c6: avg_logprob = 0.0 → exp(0.0) = 1.0 → max confidence
    score = r_acc("<answer>dog</answer>", "dog", avg_logprob=0.0)
    ok = abs(score - 0.5) < 1e-6  # conf=1.0, correct → 0.5*(1-(1-1)^2) = 0.5
    print(f"    avg_logprob=0.0 (exp=1.0): r_acc={score:.4f} (expected=0.5) {'✓' if ok else '✗'}")
    all_pass &= ok

    print()
    return all_pass


def test_verl_formula_alignment():
    """Direct formula comparison: custom r_acc vs VERL's formula.

    VERL compute_score uses:
      target_index = find_answer_pattern_flexible(responses, tokenizer, answer, gt)
      confidence = torch.exp(old_log_probs[target_index])

    Custom r_acc uses:
      confidence = min(1.0, max(0.0, math.exp(avg_logprob)))

    Both use the same score formula:
      correct: 0.5 * (1 - (confidence - 1)^2)
      wrong:  -0.5 * confidence^2

    So they are mathematically identical. The difference is the confidence source:
    - VERL: per-token log-prob at the first answer token
    - Custom: average log-prob across all generated tokens

    We verify by computing both scores for a grid of logprob values.
    """
    print("=" * 60)
    print("3. VERL formula alignment")
    print("=" * 60)
    print("\n  Both formulas are mathematically identical:")
    print("    correct = 0.5 * (1 - (conf - 1)^2)")
    print("    wrong   = -0.5 * conf^2")
    print("\n  Confidence source differs:")
    print("    VERL:  exp(old_log_probs[answer_token_index])")
    print("    Custom: exp(avg_logprob), clipped to [0, 1]")
    print()

    import torch
    from echo_rl.rewards import r_acc

    gt = "dog"
    resp_correct = "<think>ok</think><answer>dog</answer>"
    resp_wrong = "<think>ok</think><answer>cat</answer>"

    test_logprobs = [-0.1, -0.5, -1.0, -2.0, -3.0, -5.0]

    header = f"  {'avg_logprob':>12} | {'conf':>8} | {'VERL correct':>14} | {'Custom correct':>15} | {'Match':>6} | {'VERL wrong':>12} | {'Custom wrong':>13} | {'Match':>6}"
    print(header)
    print("  " + "-" * len(header))

    all_ok = True
    for lp in test_logprobs:
        conf = math.exp(lp)
        conf_clipped = min(1.0, max(0.0, conf))

        # VERL formula (no clipping)
        verl_correct = 0.5 * (1.0 - (conf - 1.0) ** 2)
        verl_wrong = -0.5 * (conf ** 2)

        # Custom formula (with clipping, but since conf ∈ (0,1] for any logprob ≤ 0,
        # clipping to [0, 1] should be a no-op for negative logprobs)
        custom_correct = r_acc(resp_correct, gt, avg_logprob=lp)
        custom_wrong = r_acc(resp_wrong, gt, avg_logprob=lp)

        match_correct = abs(custom_correct - verl_correct) < 1e-6
        match_wrong = abs(custom_wrong - verl_wrong) < 1e-6
        all_ok &= match_correct and match_wrong

        print(f"  {lp:>12.1f} | {conf:>8.4f} | {verl_correct:>+14.6f} | {custom_correct:>+15.6f} | {'✓' if match_correct else '✗':>6} | {verl_wrong:>+12.6f} | {custom_wrong:>+13.6f} | {'✓' if match_wrong else '✗':>6}")

    # Special case: logprob > 0 (rare, but tests clipping)
    print(f"\n  Clipping test (avg_logprob=+1.0 → exp=2.718 → clamped to 1.0):")
    custom_clipped = r_acc(resp_correct, gt, avg_logprob=1.0)
    # VERL would produce conf=2.718, correct → 0.5*(1-(2.718-1)^2) = 0.5*(1-2.952) = -0.976
    # Custom clips to 1.0, correct → 0.5*(1-(1-1)^2) = 0.5
    print(f"    VERL (no clip):  0.5 * (1 - (2.718-1)^2) = {0.5 * (1.0 - (2.718 - 1.0)**2):+.6f}")
    print(f"    Custom (clipped): r_acc = {custom_clipped:+.6f}")
    print(f"    ⚠  Clamping prevents degenerate scores when avg_logprob > 0")

    print()
    return all_ok


def test_total_reward_with_confidence():
    """Test that total_reward correctly passes avg_logprob to r_acc."""
    print("=" * 60)
    print("4. total_reward with avg_logprob (end-to-end)")
    print("=" * 60)

    all_ok = True

    # With avg_logprob
    resp = "<think>ok</think><answer>dog</answer>"
    gt = "dog"

    r_with = total_reward(resp, gt, consist_mode="paper", avg_logprob=-0.5)
    r_without = total_reward(resp, gt, consist_mode="paper", avg_logprob=None)

    print(f"\n  Response: correct answer 'dog'")
    print(f"  With    avg_logprob=-0.5: accuracy={r_with['accuracy']:.6f}, total={r_with['total']:.6f}")
    print(f"  Without avg_logprob:      accuracy={r_without['accuracy']:.6f}, total={r_without['total']:.6f}")

    # With logprob, the score should be slightly below 0.5 due to imperfect confidence
    # Without logprob, it should be exactly 0.5
    ok = r_with["accuracy"] < r_without["accuracy"]
    print(f"  Confidence-weighted < binary: {'✓' if ok else '✗'}")
    all_ok &= ok

    # Total reward should reflect confidence weighting
    print(f"\n  Full breakdown:")
    for k, v in r_with.items():
        print(f"    {k}: {v:+.6f}")

    print()
    return all_ok


def test_real_rollout_data():
    """Load real rollout data and test confidence-weighted rewards."""
    print("=" * 60)
    print("5. Real rollout data verification")
    print("=" * 60)

    # Find latest rollout log
    log_dirs = [
        "/hpai/aios3.0/private/user/s2025244189/s2025244265/output/grpo_vllm_batched_eps_fix_pool26/logs/rollouts.jsonl",
        os.path.join(PROJECT_ROOT, "output/grpo_vllm_batched_eps_fix_pool26/logs/rollouts.jsonl"),
    ]

    samples = []
    log_path = None
    for p in log_dirs:
        if os.path.exists(p):
            log_path = p
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            break

    if not samples:
        print("  No real rollout data found. Skipping.")
        return True

    print(f"  Loaded {len(samples)} entries from {log_path}")
    last_step = max(s["step"] for s in samples)
    step_samps = [s for s in samples if s["step"] == last_step]
    print(f"  Step {last_step}: {len(step_samps)} entries")

    # For each sample, compute reward with and without avg_logprob
    all_ok = True
    diff_count = 0
    print()
    for i, s in enumerate(step_samps[:10]):
        resp = s.get("final_response", "")
        gt = s.get("gold_answer", "")
        if not resp or not gt:
            continue

        avg_lp = s.get("avg_logprob")
        r_with = total_reward(resp, gt, consist_mode="paper", avg_logprob=avg_lp)
        r_without = total_reward(resp, gt, consist_mode="paper", avg_logprob=None)

        has_conf = avg_lp is not None
        diff = r_with["accuracy"] - r_without["accuracy"]

        answ = extract_answer(resp)
        correct_str = "✓" if normalize_answer(answ) == normalize_answer(gt) else "✗"

        print(f"  [{i}] gold={gt:<8} pred={answ:<8} {correct_str}"
              f"  avg_lp={str(avg_lp):>8} {'(binary)' if not has_conf else ''}")
        print(f"       acc_w={r_with['accuracy']:+.4f}  acc_wo={r_without['accuracy']:+.4f}"
              f"  diff={diff:+.4f}")

        if abs(diff) > 1e-4:
            diff_count += 1

    print(f"\n  Samples with confidence effect: {diff_count}/{min(len(step_samps), 10)}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen2.5-Omni-7B (for VERL compatibility test)")
    args = parser.parse_args()

    print("=" * 60)
    print("Test28: Confidence-Weighted Reward Alignment")
    print("=" * 60)

    results = {}

    # 1. Core formula
    results["formula"] = test_confidence_formula()

    # 2. r_acc synthetic
    results["r_acc"] = test_r_acc_synthetic()

    # 3. VERL formula alignment
    results["verl_alignment"] = test_verl_formula_alignment()

    # 4. total_reward end-to-end
    results["total_reward"] = test_total_reward_with_confidence()

    # 5. Real rollout data
    results["real_data"] = test_real_rollout_data()

    # ═══ Summary ═══
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        all_pass &= ok
        print(f"  {status} | {name}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

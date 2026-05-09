"""
Unit tests for echo_rl.rollout_rewards.

Tests cover penalty/bonus computation and the rollout_total aggregation.
Base rewards (format, consistency, accuracy, segment) are tested in
test_rewards.py — here we only verify that rollout_reward correctly
extends total_reward with rollout-level signals.
"""

from __future__ import annotations

import pytest

from echo_rl.rollout_rewards import rollout_reward, rollout_reward_report

# ===================================================================
# test data
# ===================================================================

RESP_CORRECT = (
    "<think>The gap between events is 0.5 seconds.</think>"
    "<seg>0.0, 2.324</seg><seg>3.5, 5.0</seg>"
    "<answer>0.5 seconds</answer>"
)
GT_CORRECT = "0.5 seconds"

RESP_WRONG = (
    "<think>I think it is 1.0 second.</think>"
    "<seg>0.0, 1.0</seg>"
    "<answer>1.0 second</answer>"
)
GT_WRONG = "0.5 seconds"

RESP_NO_ANSWER = (
    "<think>I'm not sure.</think>"
    "<seg>0.0, 1.0</seg>"
    "Let me think more..."
)

GOOD_META = {
    "triggered_interleaved": True,
    "inserted_segments": [{"start": 0.0, "end": 2.324}, {"start": 3.5, "end": 5.0}],
    "duplicate_seg_count": 0,
    "unique_segment_count": 2,
    "round_count": 3,
    "finalize_triggered": False,
    "stop_reason": "answer",
}


# ===================================================================
# rollout_reward — penalty / bonus correctness
# ===================================================================

class TestRolloutPenalties:
    """Verify each individual penalty/bonus fires independently."""

    def test_no_penalty_good_rollout(self):
        """Correct answer, unique segs, no finalize → no rollout penalty."""
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, GOOD_META)
        assert r["duplicate_penalty"] == 0.0
        assert r["round_penalty"] == 0.0
        assert r["finalize_penalty"] == 0.0
        assert r["unique_segment_bonus"] == 0.2   # 2 unique × 0.10

    def test_duplicate_penalty_applied(self):
        """duplicate_seg_count > 0 should incur a penalty."""
        meta = {**GOOD_META, "duplicate_seg_count": 2}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["duplicate_penalty"] == -0.2

    def test_no_duplicate_penalty_when_zero(self):
        """No penalty when there are no duplicates."""
        meta = {**GOOD_META, "duplicate_seg_count": 0}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["duplicate_penalty"] == 0.0

    def test_finalize_penalty_applied(self):
        """finalize_triggered=True should incur a penalty."""
        meta = {**GOOD_META, "finalize_triggered": True}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["finalize_penalty"] == -0.2

    def test_finalize_penalty_zero_when_not_triggered(self):
        """No finalize penalty when finalize was not needed."""
        meta = {**GOOD_META, "finalize_triggered": False}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["finalize_penalty"] == 0.0

    def test_round_penalty_too_few(self):
        """round_count < min_rounds should incur a penalty."""
        meta = {**GOOD_META, "round_count": 1}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["round_penalty"] == -0.05

    def test_round_penalty_too_many(self):
        """round_count > max_rounds should incur a penalty."""
        meta = {**GOOD_META, "round_count": 7}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["round_penalty"] == -0.1   # 2 rounds over × -0.05

    def test_round_penalty_in_range(self):
        """round_count between min_rounds and max_rounds → no penalty."""
        for rc in (2, 3, 4, 5):
            meta = {**GOOD_META, "round_count": rc}
            r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
            assert r["round_penalty"] == 0.0, f"failed at round_count={rc}"

    def test_unique_segment_bonus_scales(self):
        """unique_segment_count increase gives larger bonus."""
        meta0 = {**GOOD_META, "unique_segment_count": 0}
        meta1 = {**GOOD_META, "unique_segment_count": 1}
        meta3 = {**GOOD_META, "unique_segment_count": 3}
        assert rollout_reward(RESP_CORRECT, GT_CORRECT, meta0)["unique_segment_bonus"] == 0.0
        assert rollout_reward(RESP_CORRECT, GT_CORRECT, meta1)["unique_segment_bonus"] == 0.1
        assert rollout_reward(RESP_CORRECT, GT_CORRECT, meta3)["unique_segment_bonus"] == 0.3

    def test_segment_efficiency_computed(self):
        """segment_efficiency = unique / max(rounds, 1)."""
        meta = {**GOOD_META, "unique_segment_count": 2, "round_count": 4}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        assert r["segment_efficiency"] == 0.5  # 2 / 4


# ===================================================================
# rollout_total composition
# ===================================================================

class TestRolloutTotal:
    """Verify rollout_total = base total + all rollout penalties/bonuses."""

    def test_rollout_total_good_rollout(self):
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, GOOD_META)
        expected = (
            r["total"]
            + r["duplicate_penalty"]
            + r["round_penalty"]
            + r["finalize_penalty"]
            + r["unique_segment_bonus"]
        )
        assert r["rollout_total"] == pytest.approx(expected)

    def test_rollout_total_with_penalties(self):
        meta = {
            "triggered_interleaved": True,
            "inserted_segments": [],
            "duplicate_seg_count": 3,
            "unique_segment_count": 1,
            "round_count": 6,
            "finalize_triggered": True,
            "stop_reason": "duplicate_seg",
        }
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta)
        expected = (
            r["total"]
            + r["duplicate_penalty"]
            + r["round_penalty"]
            + r["finalize_penalty"]
            + r["unique_segment_bonus"]
        )
        assert r["rollout_total"] == pytest.approx(expected)
        # All penalties should be negative or zero
        assert r["duplicate_penalty"] <= 0.0
        assert r["round_penalty"] <= 0.0
        assert r["finalize_penalty"] <= 0.0

    def test_rollout_total_wrong_answer(self):
        """Wrong answer + bad rollout → low rollout_total."""
        meta = {
            "triggered_interleaved": False,
            "inserted_segments": [{"start": 0.0, "end": 1.0}],
            "duplicate_seg_count": 1,
            "unique_segment_count": 1,
            "round_count": 2,
            "finalize_triggered": True,
            "stop_reason": "duplicate_seg",
        }
        r = rollout_reward(RESP_WRONG, GT_WRONG, meta)
        assert r["accuracy"] == 0.0
        assert r["rollout_total"] <= r["total"]  # penalties drag it down

    def test_rollout_total_preserves_base_keys(self):
        """rollout_reward returns all base total_reward keys."""
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, GOOD_META)
        for key in ("format", "consistency", "accuracy", "segment", "total"):
            assert key in r


# ===================================================================
# custom coefficients
# ===================================================================

class TestCustomCoef:
    def test_custom_coefficients(self):
        """coef overrides should change penalty magnitudes."""
        custom_coef = {
            "duplicate_penalty": -0.25,
            "finalize_penalty": -0.5,
            "unique_segment_bonus": 0.2,
        }
        meta = {
            **GOOD_META,
            "duplicate_seg_count": 1,
            "finalize_triggered": True,
        }
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta, coef=custom_coef)
        assert r["duplicate_penalty"] == -0.25
        assert r["finalize_penalty"] == -0.5
        assert r["unique_segment_bonus"] == 0.4  # 2 unique × 0.20


# ===================================================================
# custom min/max rounds
# ===================================================================

class TestCustomRoundWindow:
    def test_custom_min_rounds(self):
        meta = {**GOOD_META, "round_count": 3}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta, min_rounds=4)
        assert r["round_penalty"] == -0.05

    def test_custom_max_rounds(self):
        meta = {**GOOD_META, "round_count": 3}
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, meta, max_rounds=2)
        assert r["round_penalty"] == -0.05


# ===================================================================
# edge cases
# ===================================================================

class TestRolloutEdgeCases:
    def test_empty_metadata(self):
        """All metadata fields missing → round_count=0 triggers low-round penalty."""
        r = rollout_reward(RESP_CORRECT, GT_CORRECT, {})
        assert r["duplicate_penalty"] == 0.0
        assert r["round_penalty"] == -0.05  # round_count=0 < min_rounds=2
        assert r["finalize_penalty"] == 0.0
        assert r["unique_segment_bonus"] == 0.0

    def test_empty_response(self):
        """Empty response should still produce valid reward dict."""
        r = rollout_reward("", GT_CORRECT, GOOD_META)
        assert isinstance(r, dict)
        assert "rollout_total" in r

    def test_no_answer_in_response(self):
        """No answer in response → accuracy=0, seg=0."""
        r = rollout_reward(RESP_NO_ANSWER, GT_CORRECT, GOOD_META)
        assert r["accuracy"] == 0.0
        assert r["segment"] == 0.0


# ===================================================================
# rollout_reward_report
# ===================================================================

class TestRolloutReport:
    def test_report_returns_string(self):
        report = rollout_reward_report(RESP_CORRECT, GT_CORRECT, GOOD_META)
        assert isinstance(report, str)
        assert "total=" in report
        assert "rounds=3" in report

    def test_report_no_error_on_empty(self):
        report = rollout_reward_report("", GT_CORRECT, {})
        assert isinstance(report, str)

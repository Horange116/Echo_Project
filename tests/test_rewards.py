"""
Unit tests for echo_rl.rewards.

Test data is drawn from the real targeted-smoke responses (Job 41640)
so the tests reflect actual model behaviour.
"""

from __future__ import annotations

import pytest

from echo_rl.rewards import (
    extract_answer,
    extract_segments,
    has_think,
    has_answer_tag,
    normalize_answer,
    r_format,
    r_consist,
    r_acc,
    r_seg,
    total_reward,
)

# ===================================================================
# test data — real model outputs from Job 41640
# ===================================================================

# Sample 1: 01_gap — interleaved then duplicate → finalize
S1_RESPONSE = (
    '<think><seg>0.0, 0.879</seg> '
    '<think><seg>0.0, 0.879</seg> '
    '<think><seg>0.0, 0.879</seg> <seg>0.879, 1.285</seg>'
    '<answer>0.7 seconds</answer>'
)
S1_GT = "0.1 seconds"

# Sample 2: 02_count_before — direct answer (no segs), CORRECT
S2_RESPONSE = (
    '<think>The third noise begins at 4.088 seconds. '
    'We need to count the events that finish before this timestamp.\n'
    '- The first noise finishes at 1.266 seconds.\n'
    '- The second noise finishes at 2.498 seconds.\n'
    '- The third noise begins at 4.088 seconds.\n'
    'Thus, 2 events have finished before the third noise begins.</think>\n'
    '<answer>2</answer>'
)
S2_GT = "2"

# Sample 3: 03_repeated_event_gap — interleaved → finalize
S3_RESPONSE = (
    '<think><seg>1.379, 2.272</seg> '
    '<think><seg>1.379, 2.272</seg> '
    '<answer>0.4 seconds</answer>'
)
S3_GT = "0.1 seconds"

# Sample 4: 04_duration_compare — interleaved → finalize, CORRECT
S4_RESPONSE = (
    '<think><seg>3.312, 3.452</seg> '
    '<think><seg>3.312, 3.452</seg> '
    '<answer>the whispering</answer>'
)
S4_GT = "the whispering"

# Sample 5: 05_gap — normal interleaved (seg → answer without repeat)
S5_RESPONSE = (
    '<think><seg>0.094, 2.864</seg> '
    '<think>The first male speech ends at 2.864 seconds. '
    'The impact sounds begin at 0.7 seconds. '
    'Subtracting the end time from the start time gives 0.7 '
    'minus 2.864, which equals -2.164 seconds. '
    'Using 0.7 seconds as the answer:</think>'
    '<answer>0.7 seconds</answer>'
)
S5_GT = ""

# Edge cases
EMPTY = ""
NO_TAGS = "The answer is 42."
MALFORMED_THINK = "<think>no close tag<answer>42</answer>"
MALFORMED_ANSWER = "<think>ok</think><answer>no close tag"
NESTED_XML = "<think><answer>nested</answer></think><answer>real</answer>"

# Consistency edge cases
CONSIST_OK = "<think><seg>0.0, 1.0</seg> therefore the answer is</think><answer>1</answer>"
CONSIST_VIOL_UPPER = "<think><seg>0.0, 1.0</seg> However,</think><answer>1</answer>"
CONSIST_VIOL_TAG = "<think><seg>0.0, 1.0</seg><answer>1</answer>"
CONSIST_MULTI = "<think><seg>0.0, 1.0</seg> Next<seg>1.0, 2.0</seg> Finally</think><answer>1</answer>"
CONSIST_MULTI_VIOL = "<think><seg>0.0, 1.0</seg> Next<seg>1.0, 2.0</seg> <answer>1</answer>"
CONSIST_TRAILING = "<think><seg>0.0, 1.0</seg></think><answer>1</answer>"


# ===================================================================
# extract_answer
# ===================================================================

class TestExtractAnswer:
    def test_basic(self):
        assert extract_answer(S2_RESPONSE) == "2"

    def test_with_trailing_text(self):
        assert extract_answer("<answer>0.4 seconds</answer> and more") == "0.4 seconds"

    def test_nested_picks_first(self):
        # non-greedy regex picks the first </answer>
        ans = extract_answer(NESTED_XML)
        assert ans == "nested"

    def test_empty(self):
        assert extract_answer(EMPTY) == ""

    def test_no_answer_tag(self):
        assert extract_answer(NO_TAGS) == ""

    def test_malformed(self):
        assert extract_answer(MALFORMED_ANSWER) == ""


# ===================================================================
# extract_segments
# ===================================================================

class TestExtractSegments:
    def test_basic(self):
        segs = extract_segments(S1_RESPONSE)
        assert (0.0, 0.879) in segs
        assert (0.879, 1.285) in segs

    def test_no_segments(self):
        assert extract_segments(S2_RESPONSE) == []

    def test_multiple(self):
        segs = extract_segments(
            "<seg>0.0, 1.5</seg><seg>2.0, 3.5</seg>"
        )
        assert segs == [(0.0, 1.5), (2.0, 3.5)]

    def test_whitespace_variations(self):
        segs = extract_segments(
            "<seg>  0.0 , 1.5 </seg><seg>2,3</seg>"
        )
        assert segs == [(0.0, 1.5), (2.0, 3.0)]

    def test_empty(self):
        assert extract_segments(EMPTY) == []

    def test_malformed_missing_comma(self):
        assert extract_segments("<seg>0.0 1.5</seg>") == []


# ===================================================================
# has_think / has_answer_tag
# ===================================================================

class TestHasThink:
    def test_no_close_think(self):
        # S1 has <think> but never closes it → no valid think block
        assert has_think(S1_RESPONSE) is False

    def test_no_think(self):
        assert has_think(NO_TAGS) is False

    def test_malformed(self):
        assert has_think(MALFORMED_THINK) is False


class TestHasAnswerTag:
    def test_basic(self):
        assert has_answer_tag(S1_RESPONSE) is True

    def test_no_answer(self):
        assert has_answer_tag(NO_TAGS) is False

    def test_malformed(self):
        assert has_answer_tag(MALFORMED_ANSWER) is False


# ===================================================================
# normalize_answer
# ===================================================================

class TestNormalizeAnswer:
    def test_strip_lower(self):
        assert normalize_answer("  0.7 Seconds  ") == "0.7"

    def test_trailing_period(self):
        assert normalize_answer("0.7 seconds.") == "0.7"

    def test_collapse_whitespace(self):
        assert normalize_answer("the   coughing") == "the coughing"

    def test_plain_text(self):
        assert normalize_answer("the whispering") == "the whispering"

    def test_numeric_no_unit(self):
        assert normalize_answer("2") == "2"

    def test_already_normal(self):
        assert normalize_answer("0.1 seconds") == "0.1"

    def test_with_second_singular(self):
        assert normalize_answer("1.0 second") == "1.0"


# ===================================================================
# r_format
# ===================================================================

class TestRFormat:
    def test_both_tags(self):
        assert r_format(S2_RESPONSE) == 0.5

    def test_no_tags(self):
        assert r_format(NO_TAGS) == 0.0

    def test_only_think(self):
        assert r_format("<think>ok</think>") == 0.25

    def test_only_answer(self):
        assert r_format("<answer>42</answer>") == 0.25

    def test_malformed_think(self):
        assert r_format(MALFORMED_THINK) == 0.25  # answer is valid

    def test_empty(self):
        assert r_format(EMPTY) == 0.0


# ===================================================================
# r_consist
# ===================================================================

class TestRConsist:
    # ── paper mode (default) ──

    def test_no_seg_paper(self):
        # no </seg> at all → no violation → 0.0
        assert r_consist(S2_RESPONSE) == 0.0

    def test_lowercase_continuation_paper(self):
        # "therefore" starts with lowercase 't' → OK → 0.0
        assert r_consist(CONSIST_OK) == 0.0

    def test_uppercase_continuation_paper(self):
        # "However" starts with 'H' → 1 violation → -0.1
        assert r_consist(CONSIST_VIOL_UPPER) == -0.1

    def test_tag_continuation_paper(self):
        # <answer> starts with '<' after </seg> → 1 violation → -0.1
        assert r_consist(CONSIST_VIOL_TAG) == -0.1

    def test_mixed_paper(self):
        # First </seg> → "Next" (N) → 1 violation
        # Second </seg> → "Finally" (F) → 1 violation
        # Total: 2 violations → -0.2
        assert r_consist(CONSIST_MULTI) == -0.2

    def test_mixed_with_tag_paper(self):
        # First </seg> → "Next" (N) → 1 violation
        # Second </seg> → "<answer>" (<) → 1 violation
        # Total: 2 violations → -0.2
        assert r_consist(CONSIST_MULTI_VIOL) == -0.2

    def test_trailing_seg_no_text_paper(self):
        # </seg> followed by </think> — next non-ws is '<' → -0.1
        assert r_consist(CONSIST_TRAILING) == -0.1

    def test_many_violations_capped_paper(self):
        many = "<seg>0,1</seg> A<seg>1,2</seg> B<seg>2,3</seg> C<seg>3,4</seg> D<seg>4,5</seg> E<seg>5,6</seg> F"
        assert r_consist(many) == -0.5  # capped at -0.5

    def test_real_sample_4_paper(self):
        # S4: after first </seg> → ' ' then '<' → violation
        #      after second </seg> → ' ' then '<' → violation
        # Total: 2 violations → -0.2
        assert r_consist(S4_RESPONSE) == -0.2

    def test_empty_paper(self):
        assert r_consist(EMPTY) == 0.0

    # ── positive mode (legacy compatibility) ──

    def test_no_seg_positive(self):
        assert r_consist(S2_RESPONSE, mode="positive") == 0.5

    def test_violation_positive(self):
        assert r_consist(CONSIST_VIOL_TAG, mode="positive") == 0.4

    def test_many_violations_capped_positive(self):
        many = "<seg>0,1</seg> A<seg>1,2</seg> B<seg>2,3</seg> C<seg>3,4</seg> D<seg>4,5</seg> E<seg>5,6</seg> F"
        assert r_consist(many, mode="positive") == 0.0

    def test_empty_positive(self):
        assert r_consist(EMPTY, mode="positive") == 0.5

    def test_invalid_mode_raises(self):
        import pytest
        with pytest.raises(ValueError):
            r_consist("<answer>x</answer>", mode="unknown")


# ===================================================================
# r_acc
# ===================================================================

class TestRAcc:
    def test_correct_numeric(self):
        assert r_acc(S2_RESPONSE, S2_GT) == 0.5

    def test_wrong_numeric(self):
        assert r_acc(S1_RESPONSE, S1_GT) == 0.0

    def test_correct_text(self):
        assert r_acc(S4_RESPONSE, S4_GT) == 0.5

    def test_no_answer_in_response(self):
        assert r_acc(NO_TAGS, "42") == 0.0

    def test_empty_gt(self):
        assert r_acc(S5_RESPONSE, S5_GT) == 0.0

    def test_normalize_handles_units(self):
        # "0.7 seconds" from response vs "0.7" from GT
        resp = "<answer>0.7 seconds</answer>"
        assert r_acc(resp, "0.7") == 0.5


# ===================================================================
# r_seg
# ===================================================================

class TestRSeg:
    def test_correct_with_seg(self):
        assert r_seg(S4_RESPONSE, S4_GT) == 0.5

    def test_correct_without_seg(self):
        # S2 is correct but has no segs → no segment reward
        assert r_seg(S2_RESPONSE, S2_GT) == 0.0

    def test_wrong_with_seg(self):
        assert r_seg(S1_RESPONSE, S1_GT) == 0.0

    def test_wrong_without_seg(self):
        resp = "<think>nope</think><answer>wrong</answer>"
        assert r_seg(resp, "correct") == 0.0

    def test_empty(self):
        assert r_seg(S5_RESPONSE, S5_GT) == 0.0  # S5_GT is ""


# ===================================================================
# total_reward
# ===================================================================

class TestTotalReward:
    def test_direct_answer_correct(self):
        """S2: direct answer, correct, no segs."""
        r = total_reward(S2_RESPONSE, S2_GT)
        assert r["format"] == 0.5
        assert r["consistency"] == 0.0       # no segs → no penalty
        assert r["accuracy"] == 0.5
        assert r["segment"] == 0.0           # no segs
        assert r["total"] == 1.0

    def test_interleaved_correct(self):
        """S4: interleaved → finalize, correct, has segs, but no </think>."""
        r = total_reward(S4_RESPONSE, S4_GT)
        assert r["format"] == 0.25           # no </think>, only <answer>
        assert r["consistency"] == -0.2      # 2 violations
        assert r["accuracy"] == 0.5
        assert r["segment"] == 0.5           # correct + has segs
        assert r["total"] == 1.05

    def test_interleaved_wrong(self):
        """S1: interleaved → finalize, wrong answer, no </think>."""
        r = total_reward(S1_RESPONSE, S1_GT)
        assert r["format"] == 0.25           # no </think>
        assert r["accuracy"] == 0.0
        assert r["segment"] == 0.0
        # 4 × </seg>: each followed by < → -0.4
        assert r["consistency"] == -0.4
        assert r["total"] == pytest.approx(-0.15)

    def test_positive_mode_compat(self):
        """total_reward with consist_mode='positive' matches old behaviour."""
        r = total_reward(S2_RESPONSE, S2_GT, consist_mode="positive")
        assert r["consistency"] == 0.5
        assert r["total"] == 1.5

    def test_empty(self):
        r = total_reward(EMPTY, "42")
        assert r == {"format": 0.0, "consistency": 0.0,
                     "accuracy": 0.0, "segment": 0.0, "total": 0.0}

    def test_no_tags(self):
        r = total_reward(NO_TAGS, "42")
        assert r["format"] == 0.0
        assert r["consistency"] == 0.0
        assert r["total"] == 0.0

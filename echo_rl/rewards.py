"""
Reward functions for Echo-paper-style interleaved reasoning training.

All functions operate on raw model output text — no model inference, no
verl dependency.  Designed to be composable inside a GRPO reward loop.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# regex patterns
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEG_PATTERN = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")


# ---------------------------------------------------------------------------
# extraction helpers
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    """Return text inside the first <answer>...</answer>, or ''."""
    m = _ANSWER_PATTERN.search(response)
    return m.group(1).strip() if m else ""


def extract_segments(response: str) -> list[tuple[float, float]]:
    """Return all valid (start, end) segment pairs found in response."""
    return [(float(s), float(e)) for s, e in _SEG_PATTERN.findall(response)]


def has_think(response: str) -> bool:
    """Whether response contains a properly closed <think> block."""
    return bool(_THINK_PATTERN.search(response))


def has_answer_tag(response: str) -> bool:
    """Whether response contains a properly closed <answer> block."""
    return bool(_ANSWER_PATTERN.search(response))


# ---------------------------------------------------------------------------
# answer normalisation
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalise an answer string for comparison.

    - strip / lowercase
    - collapse whitespace
    - drop trailing period
    - normalise "X second(s)" → bare number
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    text = re.sub(r"(\d+(?:\.\d+)?)\s*seconds?", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# individual reward components
# ---------------------------------------------------------------------------

def r_format(response: str) -> float:
    """Format reward (max 0.5).

    A correctly closed <think>...</think> AND <answer>...</answer>
    each contribute 0.25.
    """
    score = 0.0
    if has_think(response):
        score += 0.25
    if has_answer_tag(response):
        score += 0.25
    return score


def r_consist(response: str, mode: str = "paper") -> float:
    """Consistency reward / penalty for interleaved reasoning.

    After every ``</seg>`` the model should continue its reasoning
    fluidly (lowercase continuation).  If the next non-whitespace
    character is an uppercase letter or ``<`` the transition is
    considered incoherent — each violation costs 0.1, max 0.5.

    Parameters
    ----------
    mode : {"paper", "positive"}
        ``"paper"`` (default) → return **penalty** in [-0.5, 0]
        where 0 means no violations.

        ``"positive"`` → return [0, 0.5] as a positive reward
        (``0.5 - penalty``), matching the earlier behaviour
        before the paper-mode alignment.
    """
    penalty = 0.0
    parts = response.split("</seg>")
    for i in range(len(parts) - 1):
        remainder = parts[i + 1]
        nxt = _first_non_ws(remainder)
        if nxt and (nxt.isupper() or nxt == "<"):
            penalty += 0.1
    penalty = min(penalty, 0.5)

    if mode == "paper":
        return -penalty          # 0 → -0.5,  0 = no violations
    elif mode == "positive":
        return max(0.0, 0.5 - penalty)  # 0 → 0.5  (legacy)
    else:
        msg = f"r_consist: unknown mode '{mode}' — expected 'paper' or 'positive'"
        raise ValueError(msg)


def _first_non_ws(text: str) -> str:
    for ch in text:
        if not ch.isspace():
            return ch
    return ""


def r_acc(response: str, gt_answer: str) -> float:
    """Accuracy reward (max 0.5).

    The extracted answer must match the ground truth after
    normalisation.
    """
    if not gt_answer:
        return 0.0
    pred = extract_answer(response)
    if not pred:
        return 0.0
    return 0.5 if normalize_answer(pred) == normalize_answer(gt_answer) else 0.0


def r_seg(response: str, gt_answer: str) -> float:
    """Segment-usage reward (max 0.5).

    The answer must be correct AND the response must contain at least
    one valid ``<seg>start,end</seg>``.
    """
    if not gt_answer:
        return 0.0
    pred = extract_answer(response)
    if not pred:
        return 0.0
    if normalize_answer(pred) == normalize_answer(gt_answer) and extract_segments(response):
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# composed total reward
# ---------------------------------------------------------------------------

def total_reward(response: str, gt_answer: str,
                 consist_mode: str = "paper") -> dict[str, Any]:
    """Return a dict with all component rewards and the sum.

    Parameters
    ----------
    consist_mode : str
        Passed through to :func:`r_consist`.  Default ``"paper"`` so
        consistency acts as a penalty (≤ 0).
    """
    rew = {
        "format": r_format(response),
        "consistency": r_consist(response, mode=consist_mode),
        "accuracy": r_acc(response, gt_answer),
        "segment": r_seg(response, gt_answer),
    }
    rew["total"] = sum(rew.values())
    return rew

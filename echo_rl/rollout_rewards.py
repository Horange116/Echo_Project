"""
Rollout-level reward wrapper for interleaved reasoning training.

Extends the base ``total_reward`` (from ``rewards.py``) with penalties
and bonuses computed from rollout metadata, so the GRPO loop can
discourage degenerate behaviour (duplicate loops, finalise overuse)
and encourage efficient multi-round interleaving.
"""

from __future__ import annotations

from typing import Any

from echo_rl.rewards import total_reward

# ---------------------------------------------------------------------------
# default coefficients
# ---------------------------------------------------------------------------

_DEFAULT_COEF: dict[str, float] = {
    "duplicate_penalty": 0.0,          # per duplicate segment (neutral — re-referencing evidence during reasoning is natural)
    "round_penalty_high": -0.05,       # per round above max_rounds
    "round_penalty_low": -0.05,        # penalty if rounds < min_rounds
    "finalize_penalty": -0.20,         # when finalize was triggered
    "unique_segment_bonus": 0.10,      # per unique segment
}


# ---------------------------------------------------------------------------
# rollout reward
# ---------------------------------------------------------------------------

def rollout_reward(
    response: str,
    gt_answer: str,
    rollout_metadata: dict[str, Any],
    consist_mode: str = "paper",
    coef: dict[str, float] | None = None,
    min_rounds: int = 2,
    max_rounds: int = 5,
) -> dict[str, Any]:
    """Combined reward with rollout-level penalties / bonuses.

    Parameters
    ----------
    response:
        Raw model output text.
    gt_answer:
        Ground-truth answer string.
    rollout_metadata:
        Dict with keys (all optional, missing keys treated as zero):
            triggered_interleaved : bool
            inserted_segments    : list[dict]
            duplicate_seg_count  : int
            unique_segment_count : int
            round_count          : int
            finalize_triggered   : bool
            stop_reason          : str
    consist_mode:
        Passed through to ``r_consist``.
    coef:
        Coefficient overrides.  See ``_DEFAULT_COEF`` for defaults.
    min_rounds, max_rounds:
        Expected round-count window.
    """
    meta = rollout_metadata
    c = {**_DEFAULT_COEF, **(coef or {})}

    # --- base rewards (reuse existing total_reward) ---
    base = total_reward(response, gt_answer, consist_mode=consist_mode)

    # --- rollout penalties / bonuses ---
    dup_count = meta.get("duplicate_seg_count", 0)
    unique_count = meta.get("unique_segment_count", 0)
    rounds = meta.get("round_count", 0)
    finalized = meta.get("finalize_triggered", False)

    duplicate_penalty = dup_count * c["duplicate_penalty"] if dup_count > 0 else 0.0

    if rounds > max_rounds:
        round_penalty = (rounds - max_rounds) * c["round_penalty_high"]
    elif rounds < min_rounds:
        round_penalty = c["round_penalty_low"]
    else:
        round_penalty = 0.0

    finalize_penalty = c["finalize_penalty"] if finalized else 0.0
    unique_segment_bonus = unique_count * c["unique_segment_bonus"]

    # segment efficiency ratio (informational)
    seg_eff = unique_count / max(rounds, 1)

    # --- compose ---
    rollout_fields = {
        "duplicate_penalty": round(duplicate_penalty, 4),
        "round_penalty": round(round_penalty, 4),
        "finalize_penalty": round(finalize_penalty, 4),
        "unique_segment_bonus": round(unique_segment_bonus, 4),
        "segment_efficiency": round(seg_eff, 4),
    }
    out = {**base, **rollout_fields}
    out["rollout_total"] = round(
        base["total"]
        + duplicate_penalty
        + round_penalty
        + finalize_penalty
        + unique_segment_bonus,
        4,
    )
    return out


def rollout_reward_report(
    response: str,
    gt_answer: str,
    rollout_metadata: dict[str, Any],
    **kwargs: Any,
) -> str:
    """Human-readable one-line summary of a rollout reward."""
    r = rollout_reward(response, gt_answer, rollout_metadata, **kwargs)
    meta = rollout_metadata
    parts = [
        f"total={r['rollout_total']:+.2f}",
        f"(base={r['total']:+.2f}",
        f"acc={r['accuracy']:.2f}",
        f"seg={r['segment']:.2f}",
        f"dup={r['duplicate_penalty']:+.2f}",
        f"round={r['round_penalty']:+.2f}",
        f"final={r['finalize_penalty']:+.2f}",
        f"uniq={r['unique_segment_bonus']:+.2f})",
        f"rounds={meta.get('round_count','?')}",
        f"uniq_segs={meta.get('unique_segment_count','?')}",
        f"dup_segs={meta.get('duplicate_seg_count','?')}",
        f"finalize={meta.get('finalize_triggered','?')}",
    ]
    return "  ".join(parts)

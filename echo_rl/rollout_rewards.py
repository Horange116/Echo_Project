"""
Rollout-level reward wrapper for interleaved reasoning training.

Wraps the paper-aligned ``total_reward`` (Rformat + Rconsist + Racc + Rseg)
to produce a flat dict compatible with the GRPO training loop.
"""

from __future__ import annotations

from typing import Any

from echo_rl.rewards import total_reward


def rollout_reward(
    response: str,
    gt_answer: str,
    rollout_metadata: dict[str, Any] | None = None,
    consist_mode: str = "paper",
) -> dict[str, Any]:
    """Combined reward aligned with Echo paper Section 4.2.

    ``R(τ) = Rformat(τ) + Rconsist(τ) + Racc(τ) + Rseg(τ)``

    Parameters
    ----------
    response:
        Raw model output text.
    gt_answer:
        Ground-truth answer string.
    rollout_metadata:
        Ignored (kept for API compatibility).
    consist_mode:
        Passed through to ``r_consist``.
    """
    avg_logprob = None
    if rollout_metadata:
        avg_logprob = rollout_metadata.get("avg_logprob")
    base = total_reward(response, gt_answer, consist_mode=consist_mode, avg_logprob=avg_logprob)
    out = {**base}
    out["rollout_total"] = round(base["total"], 4)
    return out


def rollout_reward_report(
    response: str,
    gt_answer: str,
    rollout_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Human-readable one-line summary of a rollout reward."""
    r = rollout_reward(response, gt_answer, rollout_metadata, **kwargs)
    parts = [
        f"total={r['rollout_total']:+.2f}",
        f"(fmt={r['format']:.2f}",
        f"cst={r['consistency']:+.2f}",
        f"acc={r['accuracy']:.2f}",
        f"seg={r['segment']:.2f})",
    ]
    return "  ".join(parts)

#!/usr/bin/env python3
"""Lightweight unified batch schema for GRPO training pipeline.

Replaces scattered list/dict/tensor passing between the three GRPO roles:
  - collect_rollouts  → produces rollout_data + samples
  - compute_rewards   → produces metrics + rollout_rewards
  - build_advantages  → produces advantages
  - encode rollouts   → produces encoded (text-only)
  - update_actor_*    → consumes encoded/advantages/rollout_data

Usage:
    batch = TrainingBatch(
        rollout_data=[...],
        samples=[...],
        num_rollouts=4,
    )
    batch = compute_rewards(batch, global_step)        # batch.metrics populated
    batch = build_advantages_from_metrics(batch, dev)   # batch.advantages populated
    batch = encode_text_rollouts(batch, tokenizer)      # batch.encoded populated
    loss = update_actor_text(policy, ref, opt, batch, args, dev)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TrainingBatch:
    """Unified batch/data object flowing through the GRPO training pipeline.

    Core fields (set at creation):
        rollout_data:  per-trajectory rollout output dicts
        samples:       per-trajectory source sample dicts
        num_rollouts:  number of rollouts per sample (grouping factor)

    Populated by pipeline stages:
        metrics:     reward breakdown (from compute_rewards)
        advantages:  group-normalized advantages (from build_advantages_from_metrics)
        encoded:     tokenized prompt+completion tuples (from encode_text_rollouts)
    """

    rollout_data: List[dict]
    samples: List[dict]
    num_rollouts: int

    # Populated by pipeline stages (initially empty/None)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    advantages: Optional[torch.Tensor] = None
    encoded: Optional[List[Tuple[List[int], int, int]]] = None

    def __post_init__(self) -> None:
        if len(self.rollout_data) != len(self.samples):
            raise ValueError(
                f"rollout_data ({len(self.rollout_data)}) and "
                f"samples ({len(self.samples)}) must have same length"
            )
        if len(self.rollout_data) % self.num_rollouts != 0:
            raise ValueError(
                f"rollout_data size ({len(self.rollout_data)}) must be "
                f"divisible by num_rollouts ({self.num_rollouts})"
            )

    # ── Convenience properties ──

    @property
    def size(self) -> int:
        """Number of trajectories (rollout_data entries)."""
        return len(self.rollout_data)

    @property
    def rollout_rewards(self) -> List[float]:
        """Per-trajectory total reward from metrics, or empty if not computed."""
        if not self.metrics:
            return []
        return [m["rollout_total"] for m in self.metrics]

    @property
    def sample_ids(self) -> List[str]:
        return [s.get("id", "?") for s in self.samples]

    @property
    def completions(self) -> List[str]:
        """Per-trajectory final_response text."""
        return [rd.get("final_response", "") for rd in self.rollout_data]

    @property
    def predictions(self) -> List[str]:
        """Per-trajectory pred_answer text."""
        return [rd.get("pred_answer", "") for rd in self.rollout_data]

    @property
    def avg_logprobs(self) -> List[Optional[float]]:
        """Per-trajectory average log-probability (if captured by worker)."""
        return [rd.get("avg_logprob") for rd in self.rollout_data]

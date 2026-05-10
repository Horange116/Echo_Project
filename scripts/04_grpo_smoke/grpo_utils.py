"""
GRPO utility functions for interleaved audio reasoning training.

Provides:
  - get_per_token_logps: extract per-token log-probs from model forward pass
  - compute_grpo_loss: GRPO loss with clipped surrogate + KL penalty
  - compute_advantages: group-normalized advantages
  - build_rollout_metadata: build rollout_metadata dict from interleaved result
  - build_text_prompt: build text-only prompt for log-prob computation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# token-level log-prob extraction
# ---------------------------------------------------------------------------

def get_per_token_logps(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run forward pass and return per-token log P(token_{t+1} | tokens[:t+1]).

    Args:
        model: PeftModel or Qwen2_5OmniForConditionalGeneration.
               Must have a ``.thinker`` submodule (the text backbone).
        input_ids: (B, T) token IDs.
        attention_mask: (B, T) attention mask.

    Returns:
        (B, T-1) tensor of log-probabilities for each predicted token.

    Notes:
        Qwen2_5OmniForConditionalGeneration does **not** define ``forward()``
        (it inherits ``_forward_unimplemented`` from ``nn.Module``).  We use
        ``model.get_base_model().thinker`` (PeftModel) or ``model.thinker``
        (raw) to get the text-backbone which has a proper forward method.
    """
    # Access the thinker submodule (text backbone with forward())
    if hasattr(model, "get_base_model"):
        thinker = model.get_base_model().thinker
    else:
        thinker = model.thinker
    outputs = thinker(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits                     # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
    # gather log P(token_t | tokens[:t]) for each position
    per_token_logps = log_probs[:, :-1].gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)                                # (B, T-1)
    return per_token_logps


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    ref_logps: Optional[torch.Tensor] = None,
    beta: float = 0.04,
    epsilon: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """GRPO loss with clipped surrogate objective + optional KL penalty.

    Loss per token:
      -L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
      + β * KL(π_ref || π_θ)   [approximate KL per token]

    Args:
        policy_logps: (B, S) log-probs under current policy.
        old_logps: (B, S) log-probs under old policy (before update).
        advantages: (B,) group-normalised advantages.
        ref_logps: (B, S) log-probs under reference model (frozen).
        beta: KL penalty coefficient.
        epsilon: clipping range.
        mask: (B, S) — 1 for completion tokens, 0 for prompt/padding.

    Returns:
        dict with keys ``loss``, ``kl``, ``ratio``, ``pg_loss_unmasked``.
    """
    log_ratio = policy_logps - old_logps          # (B, S)
    ratio = torch.exp(log_ratio)                  # (B, S)

    adv = advantages.unsqueeze(-1)                # (B, 1) -> broadcast
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    pg_loss = -torch.min(surr1, surr2)            # (B, S)

    kl = torch.zeros_like(pg_loss)
    if ref_logps is not None and beta > 0:
        # Schulman et al. approximation:
        #   KL ≈ exp(log q - log p) - (log q - log p) - 1
        log_ref_minus_policy = ref_logps - policy_logps
        kl = torch.exp(log_ref_minus_policy) - log_ref_minus_policy - 1
        pg_loss = pg_loss + beta * kl

    if mask is not None:
        pg_loss = pg_loss * mask
        kl = kl * mask if ref_logps is not None else kl
        n_masked = mask.sum()
        loss = pg_loss.sum() / n_masked if n_masked > 0 else pg_loss.mean()
    else:
        loss = pg_loss.mean()

    return {
        "loss": loss,
        "kl": kl.detach().mean() if ref_logps is not None else torch.tensor(0.0),
        "ratio": ratio.detach().mean(),
        "pg_loss_unmasked": pg_loss.detach().mean(),
    }


# ---------------------------------------------------------------------------
# advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(
    rewards: List[float],
    group_ids: List[int],
) -> torch.Tensor:
    """Group-normalised advantages (mean=0, std=1 per group).

    Args:
        rewards: flat list of scalar rewards (one per rollout).
        group_ids: query-index for each rollout (same query = same group).

    Returns:
        (len(rewards),) tensor of advantages.
    """
    advantages = torch.zeros(len(rewards), dtype=torch.float32)
    for gid in sorted(set(group_ids)):
        indices = [i for i, g in enumerate(group_ids) if g == gid]
        group = [rewards[i] for i in indices]
        mean_r = sum(group) / len(group)
        std_r = (sum((r - mean_r) ** 2 for r in group) / len(group)) ** 0.5
        std_r = max(std_r, 1e-6)
        for idx_in_group, i in enumerate(indices):
            advantages[i] = (group[idx_in_group] - mean_r) / std_r
    return advantages


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------

def build_rollout_metadata(result: dict) -> dict:
    """Build ``rollout_metadata`` dict from an interleaved-inference result.

    Compatible with ``echo_rl.rollout_rewards.rollout_reward``.
    """
    used_segs = result.get("used_segments", [])
    unique = set((s["start"], s["end"]) for s in used_segs)
    unique_count = len(unique)
    total_count = len(used_segs)
    stop_reason = result.get("stop_reason", "unknown")

    return {
        "triggered_interleaved": total_count > 0,
        "inserted_segments": used_segs,
        "duplicate_seg_count": total_count - unique_count,
        "unique_segment_count": unique_count,
        "round_count": result.get("total_rounds", 0),
        "finalize_triggered": stop_reason.startswith("finalize_"),
        "stop_reason": stop_reason,
    }


def build_text_prompt(question: str, choices: list) -> str:
    """Build the text-only initial prompt (mirrors build_initial_prompt)."""
    choices_str = str(choices)
    return (
        question
        + " Choose the answer from "
        + choices_str
        + ". Think step-by-step. Refer to the specific audio segments while thinking, "
        + "and indicate the corresponding timestamps with <seg>start, end</seg>. "
        + "Answer in the format of <think>...</think><answer>...</answer>."
    )


def build_text_inputs(
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    max_length: int = 2048,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Tokenise prompt + completion and create a completion-position mask.

    Returns:
        ``input_ids`` (1, T), ``attention_mask`` (1, T),
        ``completion_mask`` (1, T-1) — 1 for completion-token positions.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = prompt_ids + tokenizer.encode(completion_text, add_special_tokens=False)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    prompt_len = len(prompt_ids)
    completion_mask = torch.zeros_like(input_ids)
    completion_mask[:, prompt_len:] = 1.0
    # shift: position i predicts token i+1
    completion_mask = completion_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
    }

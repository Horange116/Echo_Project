#!/usr/bin/env python3
"""Engine role boundaries for GRPO training.

Extracts the three core GRPO roles as clear functions:
  - collect_rollouts:    trajectory generation (rollout workers)
  - score_ref_logprobs:  reference model KL scoring
  - update_actor:        policy model GRPO update

Plus supporting utilities moved from rollout_smoke_test.py.

Usage:
    from engine_roles import (
        RunReport, load_training_models, unload_training_models,
        compute_rewards, encode_text_rollouts,
        update_actor_text, update_actor_strict,
        score_ref_logprobs, score_ref_logprobs_multimodal,
        check_model_for_nan_inf, ...
    )
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

# Project path setup (mirrors rollout_smoke_test.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_grpo_smoke"))

from batch_schema import TrainingBatch
from echo_rl.rollout_rewards import rollout_reward
from grpo_utils import (
    build_rollout_metadata,
    build_text_prompt,
    compute_advantages,
    compute_grpo_loss,
    get_per_token_logps,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Audio token IDs for Qwen2.5-Omni
# ═══════════════════════════════════════════════════════════════════════════════

AUDIO_BOS_ID = 151647
AUDIO_EOS_ID = 151648
AUDIO_TOKEN_ID = 151646

# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def _diagnose_nan(tag: str, **tensors: torch.Tensor) -> None:
    """Print min/max/any_nan/any_inf for each named tensor."""
    parts = [f"[nan-diag {tag}]"]
    for name, t in tensors.items():
        if t is None or t.numel() == 0:
            parts.append(f"{name}=None/empty")
            continue
        parts.append(
            f"{name}: min={t.min().item():.6g} max={t.max().item():.6g} "
            f"mean={t.mean().item():.6g} nan={bool(t.isnan().any())} "
            f"inf={bool(t.isinf().any())}"
        )
    print("  " + " | ".join(parts))


def check_model_for_nan_inf(
    model: torch.nn.Module, tag: str = "model",
    fatal: bool = False,
    skip_prefixes: tuple = (),
) -> bool:
    """
    Scan all trainable parameters (or all if none trainable) for nan/inf.
    Returns True if clean, False if corruption detected.
    If fatal=True, raises RuntimeError on first corrupted parameter.
    """
    found_nan = False
    found_inf = False
    first_nan = first_inf = ""
    for name, p in model.named_parameters():
        if skip_prefixes and name.startswith(skip_prefixes):
            continue
        if p.numel() == 0:
            continue
        if p.isnan().any():
            if not found_nan:
                first_nan = name
            found_nan = True
            if fatal:
                raise RuntimeError(
                    f"[nan-inf] {tag}: nan detected in {name}"
                )
        if p.isinf().any():
            if not found_inf:
                first_inf = name
            found_inf = True
            if fatal:
                raise RuntimeError(
                    f"[nan-inf] {tag}: inf detected in {name}"
                )
    if found_nan or found_inf:
        msg = f"  [nan-inf] {tag}: "
        if found_nan:
            msg += f"nan (first={first_nan}) "
        if found_inf:
            msg += f"inf (first={first_inf}) "
        print(msg)
        return False
    return True


def check_trainable_grads_for_nan_inf(
    model: torch.nn.Module, tag: str = "grad",
) -> bool:
    """Scan gradients of trainable LoRA parameters for nan/inf."""
    found_nan = False
    found_inf = False
    first_nan = first_inf = ""
    nan_count = 0
    inf_count = 0
    total_grad_elems = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or "lora" not in name:
            continue
        if p.grad is None:
            continue
        g = p.grad
        n = g.numel()
        total_grad_elems += n
        if g.isnan().any():
            found_nan = True
            nan_count += g.isnan().sum().item()
            if not first_nan:
                first_nan = name
        if g.isinf().any():
            found_inf = True
            inf_count += g.isinf().sum().item()
            if not first_inf:
                first_inf = name

    if found_nan or found_inf:
        msg = (
            f"  [diag-grad {tag}] "
            f"{'nan' if found_nan else ''} "
            f"{'inf' if found_inf else ''} "
            f"in grad (total_grad_elems={total_grad_elems})"
        )
        if found_nan:
            msg += f" nan_elems={nan_count} first_nan={first_nan}"
        if found_inf:
            msg += f" inf_elems={inf_count} first_inf={first_inf}"
        for name, p in model.named_parameters():
            if not p.requires_grad or "lora" not in name:
                continue
            if p.grad is None:
                continue
            g = p.grad
            if g.isnan().any() or g.isinf().any():
                msg += (
                    f" | e.g. {name}: grad "
                    f"min={g.min().item():.6g} "
                    f"max={g.max().item():.6g} "
                    f"mean={g.mean().item():.6g} "
                    f"nan={bool(g.isnan().any())} "
                    f"inf={bool(g.isinf().any())}"
                )
                break
        print(msg)
        return False
    else:
        for name, p in model.named_parameters():
            if not p.requires_grad or "lora" not in name:
                continue
            if p.grad is None:
                continue
            g = p.grad
            print(
                f"  [diag-grad {tag}] all clean | e.g. {name}: grad "
                f"min={g.min().item():.6g} "
                f"max={g.max().item():.6g} "
                f"mean={g.mean().item():.6g} "
                f"nan={bool(g.isnan().any())} "
                f"inf={bool(g.isinf().any())}"
            )
            break
        return True


def diagnose_lora_params(model: torch.nn.Module, tag: str = "param") -> None:
    """Scan trainable LoRA params for nan/inf and print first param's stats."""
    for name, p in model.named_parameters():
        if not p.requires_grad or "lora" not in name:
            continue
        pstats = (
            f"min={p.min().item():.6g} "
            f"max={p.max().item():.6g} "
            f"mean={p.mean().item():.6g} "
            f"nan={bool(p.isnan().any())} "
            f"inf={bool(p.isinf().any())}"
        )
        print(f"  [diag-param {tag}] {name}: {pstats}")
        break


def _diagnose_input_features(
    tag: str, input_features, feat_attn,
) -> None:
    """Print input_features and feature_attention_mask stats."""
    if input_features is None:
        print(f"  [diag {tag}] input_features=None")
        return
    print(
        f"  [diag {tag}] input_features: shape={tuple(input_features.shape)} "
        f"dtype={input_features.dtype} "
        f"min={input_features.min().item():.6g} "
        f"max={input_features.max().item():.6g} "
        f"mean={input_features.mean().item():.6g} "
        f"nan={input_features.isnan().any().item()} "
        f"inf={input_features.isinf().any().item()}"
    )
    if feat_attn is not None:
        print(
            f"  [diag {tag}] feature_attention_mask: "
            f"shape={tuple(feat_attn.shape)} "
            f"sum={feat_attn.sum().item()} "
            f"min={feat_attn.min().item()} max={feat_attn.max().item()} "
            f"nan={feat_attn.isnan().any().item()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RunReport
# ═══════════════════════════════════════════════════════════════════════════════


class RunReport:
    def __init__(self):
        self.rollout_success_count = 0
        self.rollout_failed_count = 0
        self.worker_restart_count = 0
        self.rollout_times: List[float] = []
        self.batch_times: List[float] = []
        self.strict_forward_success = 0
        self.strict_forward_failed = 0
        self.peak_memory_mb: List[int] = []

    def print(self):
        print(f"\n  ── Run Report ──")
        print(f"  rollout_success_count:  {self.rollout_success_count}")
        print(f"  rollout_failed_count:   {self.rollout_failed_count}")
        print(f"  worker_restart_count:   {self.worker_restart_count}")
        if self.rollout_times:
            avg = sum(self.rollout_times) / len(self.rollout_times)
            print(f"  avg_rollout_time:       {avg:.1f}s")
        if self.batch_times:
            total = sum(self.batch_times)
            print(f"  total_wall_time:        {total:.1f}s")
        if self.strict_forward_success + self.strict_forward_failed > 0:
            print(f"  strict_forward_success: {self.strict_forward_success}")
            print(f"  strict_forward_failed:  {self.strict_forward_failed}")
        if self.peak_memory_mb:
            print(f"  peak_memory_mb:         {max(self.peak_memory_mb)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading / unloading
# ═══════════════════════════════════════════════════════════════════════════════


def load_training_models(
    model_path: str, adapter_path: str, device: torch.device,
    disable_talker: bool = True,
):
    """Load policy + reference models for training.

    Args:
        disable_talker: If True, disable talker (saves VRAM, used in text_only
                        mode). If False, keep talker (needed for
                        strict_interleaved audio forward).
    """
    from peft import PeftModel
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu",
    )
    policy_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    if disable_talker:
        policy_model.base_model.disable_talker()
    policy_model = policy_model.to(device, dtype=torch.float16)
    for n, p in policy_model.named_parameters():
        p.requires_grad_("lora" in n)
    policy_model.eval()

    base2 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu",
    )
    ref_model = PeftModel.from_pretrained(base2, adapter_path)
    if disable_talker:
        ref_model.base_model.disable_talker()
    ref_model = ref_model.to(device, dtype=torch.float16)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return policy_model, ref_model, processor


def unload_training_models(policy_model, ref_model) -> None:
    """Release all GPU memory held by training models."""
    del policy_model
    del ref_model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Strict interleaved forward helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _count_audio_tokens(ids: List[int]) -> int:
    """Count tokens in audio spans (bos, N x audio_token, eos)."""
    count = 0
    in_audio = False
    for tid in ids:
        if tid == AUDIO_BOS_ID:
            in_audio = True
        if in_audio:
            count += 1
        if tid == AUDIO_EOS_ID:
            in_audio = False
    return count


def build_strict_interleaved_input(
    processor, sample: dict, rollout_data: dict, device: torch.device,
    max_length: int = 2048,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                    Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Reconstruct interleaved multimodal sequence.

    Sequence: [full_audio][prompt][R1_text][seg1_audio][R2_text]...

    Returns:
        input_ids: (1, T)
        attention_mask: (1, T)
        loss_mask: (1, T-1)  — 1 for model-generated text tokens, 0 elsewhere
        input_features: audio features tensor, or None
        feature_attention_mask: audio feature attention mask, or None
    """
    try:
        import librosa

        round_outputs = rollout_data.get("round_outputs", [])
        all_round_texts = []
        segs_by_round = {}
        used_segments = rollout_data.get("used_segments", [])
        used_segment_paths = rollout_data.get("used_segment_paths", [])

        for ro in round_outputs:
            if isinstance(ro.get("round"), int):
                all_round_texts.append(ro.get("text", ""))
        for seg, path in zip(used_segments, used_segment_paths):
            r = seg.get("round", 0)
            segs_by_round.setdefault(r, []).append(path)

        full_audio, sr = librosa.load(sample["audio_path"], sr=16000)

        # Build content: [full_audio][prompt] + interleaved rounds
        content = [
            {"type": "audio", "audio": full_audio},
            {"type": "text", "text": build_text_prompt(
                sample["question"], sample.get("choices", []),
            )},
        ]

        for i, round_text in enumerate(all_round_texts):
            content.append({"type": "text", "text": round_text.strip()})
            for seg_path in segs_by_round.get(i + 1, []):
                seg_audio, _ = librosa.load(seg_path, sr=16000)
                content.append({"type": "audio", "audio": seg_audio})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False,
        )

        # Gather all audios
        all_audios = [full_audio]
        for seg_path in used_segment_paths:
            seg_audio, _ = librosa.load(seg_path, sr=16000)
            all_audios.append(seg_audio)

        if len(all_audios) > 1:
            inputs = processor(
                text=text, audio=all_audios, return_tensors="pt",
                padding=True, sampling_rate=16000,
            )
        else:
            inputs = processor(
                text=text, audio=all_audios[0], return_tensors="pt",
                padding=True, sampling_rate=16000,
            )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(input_ids),
        ).to(device)
        input_features = inputs.get("input_features")
        feature_attention_mask = inputs.get("feature_attention_mask")
        if input_features is not None:
            input_features = input_features.to(device)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device)

        # Build loss mask: find audio token spans and prompt span
        T = input_ids.shape[1]
        loss_mask = torch.ones(T - 1, dtype=torch.float, device=device)

        # Mask audio token spans
        ids = input_ids[0].tolist()
        in_audio = False
        for i, tid in enumerate(ids):
            if tid == AUDIO_BOS_ID:
                in_audio = True
            if in_audio and i < T - 1:
                loss_mask[i] = 0.0
            if tid == AUDIO_EOS_ID:
                in_audio = False
                if i < T - 1:
                    loss_mask[i] = 0.0

        # Mask prompt tokens (before first model-generated text)
        prompt_text = build_text_prompt(
            sample["question"], sample.get("choices", []),
        )
        prompt_ids = processor.tokenizer.encode(
            prompt_text, add_special_tokens=False,
        )
        prompt_end = len(prompt_ids)
        audio_prefix_len = _count_audio_tokens(ids[:prompt_end + 100])
        mask_start = prompt_end + audio_prefix_len
        for i in range(min(mask_start, T - 1)):
            loss_mask[i] = 0.0

        n_masked_text = loss_mask.sum().item()
        n_audio_masked = (T - 1) - loss_mask.sum().item()
        print(
            f"    [strict] input_ids={input_ids.shape}, "
            f"masked_text_tokens={int(n_masked_text)}, "
            f"masked_audio+prompt={int(n_audio_masked)}",
            file=sys.stderr,
        )

        return (
            input_ids, attention_mask, loss_mask,
            input_features, feature_attention_mask,
        )

    except Exception as e:
        print(
            f"    [strict] build_strict_interleaved_input failed: {e}",
            file=sys.stderr,
        )
        return None


def get_per_token_logps_multimodal(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    input_features: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get per-token log-probs using thinker with optional audio features.

    Falls back to text-only if input_features is None.
    """
    thinker = model.get_base_model().thinker

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if input_features is not None:
        kwargs["input_features"] = input_features
    if feature_attention_mask is not None:
        kwargs["feature_attention_mask"] = feature_attention_mask

    outputs = thinker(**kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    print(
        f"  [diag thinker_logits] shape={tuple(logits.shape)} "
        f"min={logits.min().item():.6g} max={logits.max().item():.6g} "
        f"mean={logits.mean().item():.6g} "
        f"nan={bool(logits.isnan().any())} inf={bool(logits.isinf().any())}"
    )
    log_probs = logits.log_softmax(dim=-1)
    print(
        f"  [diag log_softmax] "
        f"min={log_probs.min().item():.6g} max={log_probs.max().item():.6g} "
        f"mean={log_probs.mean().item():.6g} "
        f"nan={bool(log_probs.isnan().any())} inf={bool(log_probs.isinf().any())}"
    )
    per_token_logps = log_probs[:, :-1].gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    return per_token_logps  # (B, T-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Role: Score Reference Logprobs
# ═══════════════════════════════════════════════════════════════════════════════


def score_ref_logprobs(
    ref_model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Score tokens under the frozen reference model (text-only)."""
    with torch.no_grad():
        return get_per_token_logps(ref_model, input_ids, attention_mask)


def score_ref_logprobs_multimodal(
    ref_model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    input_features: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score tokens under the frozen reference model (multimodal)."""
    with torch.no_grad():
        return get_per_token_logps_multimodal(
            ref_model, input_ids, attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Role: Encode Text Rollouts (pre-processing for text-only update)
# ═══════════════════════════════════════════════════════════════════════════════


def encode_text_rollouts(
    batch: TrainingBatch,
    tokenizer,
    max_length: int = 2048,
) -> TrainingBatch:
    """Encode text-only rollouts into (token_ids, seq_len, prompt_len) tuples.

    Each tuple stores:
       token_ids: prompt + completion token IDs (list of ints)
       seq_len:   total sequence length
       prompt_len: length of the prompt portion (non-trainable prefix)

    Populates and returns ``batch`` with ``batch.encoded`` set.
    Used by update_actor_text to build padded micro-batches.
    """
    encoded = []
    for rollout_data, sample in zip(batch.rollout_data, batch.samples):
        prompt = build_text_prompt(
            sample["question"], sample.get("choices", []),
        )
        completion = rollout_data.get("final_response", "")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = prompt_ids + tokenizer.encode(
            completion, add_special_tokens=False,
        )
        full_ids = full_ids[:max_length]
        encoded.append((full_ids, len(full_ids), len(prompt_ids)))
    batch.encoded = encoded
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Role: Compute Rewards and Advantages
# ═══════════════════════════════════════════════════════════════════════════════


def compute_rewards(
    batch: TrainingBatch,
    global_step: int,
) -> TrainingBatch:
    """Compute rollout rewards for all trajectories.

    Populates and returns ``batch`` with ``batch.metrics`` set.
    """
    all_metrics: List[Dict[str, Any]] = []
    for rollout_data, sample in zip(batch.rollout_data, batch.samples):
        resp = rollout_data.get("final_response", "")
        meta = build_rollout_metadata(rollout_data)
        rew = rollout_reward(resp, sample.get("answer", ""), meta)
        all_metrics.append({
            "step": global_step,
            "sample_id": sample.get("id", "?"),
            "question": sample.get("question", "")[:200],
            "gold_answer": sample.get("answer", ""),
            "rollout_total": rew.get("rollout_total", 0.0),
            "format": rew.get("format", 0.0),
            "consistency": rew.get("consistency", 0.0),
            "accuracy": rew.get("accuracy", 0.0),
            "segment": rew.get("segment", 0.0),
            "pred_answer": rollout_data.get("pred_answer", ""),
            "has_answer": int(
                bool(rollout_data.get("pred_answer", ""))
            ),
            "is_correct": int(
                rollout_data.get("pred_answer", "")
                == sample.get("answer", "")
            ),
            "round_count": rollout_data.get("total_rounds", 0),
            "stop_reason": rollout_data.get("stop_reason", "error"),
            "triggered_interleaved": rollout_data.get(
                "triggered_interleaved", False
            ),
            "has_final_answer": rollout_data.get("has_final_answer", False),
            "answer_correct": rollout_data.get("answer_correct", False),
            "avg_logprob": rollout_data.get("avg_logprob"),
            "final_response": resp,
            "used_segments": rollout_data.get("used_segments", []),
            "round_outputs": rollout_data.get("round_outputs", []),
        })
    batch.metrics = all_metrics
    return batch


def build_advantages_from_metrics(
    batch: TrainingBatch,
    device: torch.device,
) -> TrainingBatch:
    """Group-normalized advantages from reward metrics on the batch.

    Populates and returns ``batch`` with ``batch.advantages`` set.
    Uses grpo_utils.compute_advantages internally.
    """
    rollout_totals = [m["rollout_total"] for m in batch.metrics]
    group_ids = [i // batch.num_rollouts for i in range(len(batch.metrics))]
    batch.advantages = compute_advantages(rollout_totals, group_ids).to(device)
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Role: Update Actor — text_only mode
# ═══════════════════════════════════════════════════════════════════════════════


def update_actor_text(
    policy_model,
    ref_model,
    optimizer,
    batch: TrainingBatch,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor, bool]:
    """GRPO policy update with text-only thinker forward.

    Micro-batches rollouts (padded to equal length within each micro-batch),
    runs policy + ref forward on each micro-batch, computes GRPO clipped
    surrogate loss + KL penalty, and backpropagates.

    Reads ``batch.encoded`` and ``batch.advantages``.

    Returns:
        (agg_loss, agg_kl, grad_norm, weights_healthy)
    """
    encoded = batch.encoded  # List[Tuple[List[int], int, int]]
    advantages = batch.advantages

    B_total = len(encoded)
    total_masked_tokens = sum(
        max(0, (sl - 1) - (pl - 1)) for _, sl, pl in encoded
    )

    policy_model.train()
    micro_bs = args.policy_forward_micro_batch_size
    num_micro = (B_total + micro_bs - 1) // micro_bs
    optimizer.zero_grad()

    log_loss_sum = log_kl_sum = 0.0
    log_tokens = 0

    for mb_idx in range(0, B_total, micro_bs):
        mb_encoded = encoded[mb_idx:mb_idx + micro_bs]
        mb_adv = advantages[mb_idx:mb_idx + micro_bs]
        mb_bs = len(mb_encoded)
        mb_max_len = max(sl for _, sl, _ in mb_encoded)

        mb_ids = torch.zeros(
            (mb_bs, mb_max_len), dtype=torch.long, device=device,
        )
        mb_attn = torch.zeros(
            (mb_bs, mb_max_len), dtype=torch.long, device=device,
        )
        mb_cmask = torch.zeros(
            (mb_bs, mb_max_len - 1), dtype=torch.float, device=device,
        )
        for j, (full_ids, seq_len, prompt_len) in enumerate(mb_encoded):
            mb_ids[j, :seq_len] = torch.tensor(
                full_ids, dtype=torch.long, device=device,
            )
            mb_attn[j, :seq_len] = 1
            if seq_len - 1 > prompt_len - 1:
                mb_cmask[j, prompt_len - 1:seq_len - 1] = 1.0

        # Policy forward + compute per-token logprobs
        mb_policy_logps = get_per_token_logps(policy_model, mb_ids, mb_attn)
        mb_old_logps = mb_policy_logps.detach()

        # Reference model scoring (frozen, no grad)
        mb_ref_logps = score_ref_logprobs(ref_model, mb_ids, mb_attn)

        # GRPO loss
        loss_dict = compute_grpo_loss(
            mb_policy_logps, mb_old_logps, mb_adv,
            ref_logps=mb_ref_logps, beta=args.kl_coef, epsilon=0.2,
            mask=mb_cmask,
        )
        n_tokens = mb_cmask.sum()
        if n_tokens > 0 and total_masked_tokens > 0:
            scale = n_tokens / total_masked_tokens
        else:
            scale = 1.0 / max(num_micro, 1)
        (loss_dict["loss"] * scale).backward()

        nt = max(n_tokens.item(), 1)
        log_loss_sum += loss_dict["loss"].detach().item() * nt
        log_kl_sum += loss_dict["kl"].detach().item() * nt
        log_tokens += nt

        del mb_policy_logps, mb_old_logps, mb_ref_logps, loss_dict

    # Gradient clipping and optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy_model.parameters(), args.max_grad_norm,
    )
    optimizer.step()
    weights_healthy = check_model_for_nan_inf(policy_model, "text_post_step")
    agg_loss = log_loss_sum / max(log_tokens, 1)
    agg_kl = log_kl_sum / max(log_tokens, 1)

    return agg_loss, agg_kl, grad_norm, weights_healthy


# ═══════════════════════════════════════════════════════════════════════════════
# Role: Update Actor — strict_interleaved mode
# ═══════════════════════════════════════════════════════════════════════════════


def update_actor_strict(
    policy_model,
    ref_model,
    processor,
    optimizer,
    batch: TrainingBatch,
    args: argparse.Namespace,
    report: RunReport,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor, bool]:
    """GRPO policy update with full multimodal (strict interleaved) forward.

    Pre-builds interleaved inputs, then processes each rollout individually
    (micro_batch_size forced to 1). Runs policy + ref forward with audio
    features, computes GRPO loss with audio-aware loss mask, and backpropagates.

    Reads ``batch.rollout_data``, ``batch.samples``, and ``batch.advantages``.

    Returns:
        (agg_loss, agg_kl, grad_norm, weights_healthy)
    """
    policy_model.train()
    micro_bs = args.policy_forward_micro_batch_size
    if micro_bs > 1:
        print(f"  [strict] forcing micro_batch_size=1")
        micro_bs = 1
    B_total = batch.size
    optimizer.zero_grad()

    log_loss_sum = log_kl_sum = 0.0
    log_tokens = 0
    total_masked_tokens = 0

    # Pre-build interleaved inputs
    strict_inputs = []
    for rollout_data, sample in zip(batch.rollout_data, batch.samples):
        inp = build_strict_interleaved_input(
            processor, sample, rollout_data, device,
        )
        strict_inputs.append(inp)

    for i, inp in enumerate(strict_inputs):
        if inp is None:
            report.strict_forward_failed += 1
            continue
        input_ids, attn_mask, loss_mask, input_features, feat_attn = inp
        total_masked_tokens += loss_mask.sum().item()
        report.strict_forward_success += 1

        _diagnose_input_features("pre_forward", input_features, feat_attn)

        try:
            # Policy forward with optional audio features
            policy_logps = get_per_token_logps_multimodal(
                policy_model, input_ids, attn_mask,
                input_features=input_features,
                feature_attention_mask=feat_attn,
            )
            old_logps = policy_logps.detach()

            # Reference model scoring (multimodal)
            ref_logps = score_ref_logprobs_multimodal(
                ref_model, input_ids, attn_mask,
                input_features=input_features,
                feature_attention_mask=feat_attn,
            )

            # Text-only control diagnostic (same input, no audio features)
            with torch.no_grad():
                text_logps = get_per_token_logps_multimodal(
                    policy_model, input_ids, attn_mask,
                    input_features=None,
                    feature_attention_mask=None,
                )
            print(
                f"  [diag text_control] "
                f"min={text_logps.min().item():.6g} "
                f"max={text_logps.max().item():.6g} "
                f"mean={text_logps.mean().item():.6g} "
                f"nan={bool(text_logps.isnan().any())} "
                f"inf={bool(text_logps.isinf().any())}"
            )
            del text_logps

            adv_i = batch.advantages[i:i + 1]

            # Pre-loss diagnostic
            _ratio = torch.exp(policy_logps - old_logps)
            _ref_minus_policy = ref_logps - policy_logps
            _kl = torch.exp(_ref_minus_policy) - _ref_minus_policy - 1
            _diagnose_nan(
                "strict_pre_loss",
                advantages=batch.advantages,
                adv_i=adv_i,
                policy_logps=policy_logps,
                old_logps=old_logps,
                ref_logps=ref_logps,
                ref_minus_policy=_ref_minus_policy,
                kl_tensor=_kl,
                ratio=_ratio,
                loss_mask=loss_mask,
            )

            # GRPO loss with audio-aware mask
            loss_dict = compute_grpo_loss(
                policy_logps, old_logps, adv_i,
                ref_logps=ref_logps, beta=args.kl_coef,
                epsilon=0.2, mask=loss_mask.unsqueeze(0),
            )
            n_tokens = loss_mask.sum()
            if n_tokens > 0 and total_masked_tokens > 0:
                scale = n_tokens / total_masked_tokens
            else:
                scale = 1.0 / max(B_total, 1)

            if loss_dict["loss"].isnan() or loss_dict["loss"].isinf():
                print(
                    f"  [nan-inf] strict loss={loss_dict['loss'].item():.6g}, "
                    f"skipping backward"
                )
                del policy_logps, old_logps, ref_logps, loss_dict
                torch.cuda.empty_cache()
                continue

            (loss_dict["loss"] * scale).backward()

            nt = max(n_tokens.item(), 1)
            log_loss_sum += loss_dict["loss"].detach().item() * nt
            log_kl_sum += loss_dict["kl"].detach().item() * nt
            log_tokens += nt

            if hasattr(torch.cuda, 'max_memory_allocated'):
                report.peak_memory_mb.append(
                    torch.cuda.max_memory_allocated(device) // 1024 // 1024
                )

            del policy_logps, old_logps, ref_logps, loss_dict

        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                print(
                    f"  [strict] CUDA error on rollout {i}: "
                    f"{str(e)[:100]}"
                )
                report.strict_forward_failed += 1
                torch.cuda.empty_cache()
                continue
            raise

    if log_tokens > 0:
        check_trainable_grads_for_nan_inf(policy_model, "strict_pre_step")
        diagnose_lora_params(policy_model, "strict_pre_step")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(), args.max_grad_norm,
        )
        optimizer.step()

        if not check_model_for_nan_inf(policy_model, "strict_post_step"):
            weights_healthy = False
        else:
            weights_healthy = True
        diagnose_lora_params(policy_model, "strict_post_step")

        agg_loss = log_loss_sum / log_tokens
        agg_kl = log_kl_sum / log_tokens
    else:
        grad_norm = torch.tensor(0.0)
        agg_loss = 0.0
        agg_kl = 0.0
        weights_healthy = True

    return agg_loss, agg_kl, grad_norm, weights_healthy

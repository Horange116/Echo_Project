#!/usr/bin/env python3
"""
GRPO smoke test for interleaved audio reasoning.

This is a custom training loop (not using ms-swift's GRPOTrainer) because
the interleaved audio inference pipeline is incompatible with the standard
text-generation interface that ``trl.GRPOTrainer`` expects.

Pipeline per training step:
  1. Generate N rollouts per query via ``run_interleaved`` (strategy B)
  2. Compute ``rollout_reward`` for each rollout
  3. Compute text-approximate per-token log-probs (policy + reference)
  4. Group-normalise advantages
  5. GRPO loss (clipped surrogate + KL penalty) → backward → update

Limitation
----------
Per-token log-probs are computed on *text-only* inputs (no audio features).
This is an approximation — the true generative distribution depends on
interleaved audio context. Suitable for a smoke test; for production RL
use verl with the full multi-modal forward pass.

Logged metrics (TensorBoard + stdout)
-------------------------------------
  reward/*           rollout_total, base_total, accuracy, segment, format, consistency
  rollout/*          triggered_interleaved_rate, unique_segment_count,
                     duplicate_seg_count, finalize_rate, answer_rate,
                     answer_correct_rate, round_count
  train/*            loss, approx_kl, ratio, grad_norm, learning_rate, epoch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gc

import numpy as np
import torch
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Project-internal imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from echo_rl.rollout_rewards import rollout_reward, rollout_reward_report
from scripts.interleaved_infer import run_interleaved, load_model_and_processor

from grpo_utils import (
    build_rollout_metadata,
    build_text_inputs,
    build_text_prompt,
    compute_advantages,
    compute_grpo_loss,
    get_per_token_logps,
)


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO smoke test — interleaved audio")
    # paths
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--data_path", default="output/judge/split_rl.jsonl")
    p.add_argument("--output_dir", default="output/grpo_smoke")
    # training
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--kl_coef", type=float, default=0.04)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # rollout
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
    # log-prob text-only (max context length)
    p.add_argument("--max_text_length", type=int, default=2048)
    # micro-batch forward (avoid CUDA OOM with large B×T)
    p.add_argument("--policy_forward_micro_batch_size", type=int, default=4,
                   help="Max batch size for a single policy thinker() forward (grad). "
                        "Exp A shows bs<=4 is safe; bs>=8 crashes at T≈500.")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Stop after this many training steps (-1 = full dataset).")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Limit dataset to first N samples (-1 = use all).")
    p.add_argument("--checkpoint_every", type=int, default=25,
                   help="Save checkpoint every N steps.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_policy_model(
    model_path: str,
    adapter_path: str,
) -> Tuple[PeftModel, Qwen2_5OmniProcessor]:
    """Load policy model: enable training on SFT checkpoint's LoRA weights.

    We train the existing LoRA adapters directly (no merge/unload) to
    avoid PEFT forward-compatibility issues.
    """
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.base_model.disable_talker()
    model = model.to("cuda:0", dtype=torch.float16)

    # Freeze everything except LoRA parameters
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.2f}%)")
    model.eval()
    return model, processor


def load_reference_model(
    model_path: str,
    adapter_path: str,
    device: torch.device = torch.device("cuda"),
) -> PeftModel:
    """Load a frozen reference model on the target device."""
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.base_model.disable_talker()
    model = model.to(device, dtype=torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[dict]:
    """Load split_rl.jsonl and filter samples with missing audio files."""
    import os
    samples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                # Normalise field names: EAQA_RL.jsonl uses "multi_choice"
                if "multi_choice" in s and "choices" not in s:
                    s["choices"] = s["multi_choice"]
                if os.path.exists(s.get("audio_path", "")):
                    samples.append(s)
                else:
                    skipped += 1
                    print(f"  Skipping {s.get('id','?')}: audio not found")
    print(f"  Dataset: {len(samples)} loaded, {skipped} skipped (missing audio)")
    return samples


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------

def pad_and_stack(tensors: List[torch.Tensor], max_len: int, pad_value: float = float("-inf")) -> torch.Tensor:
    """Pad a list of (1, L_i) tensors to (B, max_len)."""
    batch = []
    for t in tensors:
        L = t.shape[-1]
        if L < max_len:
            pad = t.new_full((1, max_len - L), pad_value)
            batch.append(torch.cat([t, pad], dim=-1))
        else:
            batch.append(t[:, :max_len])
    return torch.cat(batch, dim=0)


def parse_rollout_metrics(
    result: dict, sample: dict, reward_dict: dict
) -> Dict[str, Any]:
    """Extract a flat dict of metrics for one rollout."""
    meta = build_rollout_metadata(result)
    return {
        "rollout_total": reward_dict.get("rollout_total", 0.0),
        "base_total": reward_dict.get("total", 0.0),
        "accuracy": reward_dict.get("accuracy", 0.0),
        "segment": reward_dict.get("segment", 0.0),
        "format": reward_dict.get("format", 0.0),
        "consistency": reward_dict.get("consistency", 0.0),
        "triggered_interleaved": int(meta["triggered_interleaved"]),
        "unique_segment_count": meta["unique_segment_count"],
        "duplicate_seg_count": meta["duplicate_seg_count"],
        "finalize_triggered": int(meta["finalize_triggered"]),
        "has_answer": int(bool(result.get("pred_answer", ""))),
        "is_correct": int(result.get("pred_answer", "") == sample.get("answer", "")),
        "round_count": meta["round_count"],
        "stop_reason": meta["stop_reason"],
    }


def log_metrics(
    writer: SummaryWriter,
    step: int,
    loss_dict: Dict[str, torch.Tensor],
    all_metrics: List[Dict[str, Any]],
    extra: Dict[str, float] = None,
):
    """Log all metrics to TensorBoard and print one-line summary."""
    # Loss / train
    writer.add_scalar("train/loss", loss_dict["loss"].item(), step)
    writer.add_scalar("train/approx_kl", loss_dict["kl"].item(), step)
    writer.add_scalar("train/ratio", loss_dict["ratio"].item(), step)
    if extra:
        for k, v in extra.items():
            writer.add_scalar(f"train/{k}", v, step)

    # Reward components
    for key in ("rollout_total", "base_total", "accuracy", "segment",
                "format", "consistency"):
        vals = [m[key] for m in all_metrics]
        writer.add_scalar(f"reward/{key}", sum(vals) / len(vals), step)

    # Rollout diagnostics — rates
    n = len(all_metrics)
    for key, label in [
        ("triggered_interleaved", "triggered_interleaved_rate"),
        ("finalize_triggered", "finalize_rate"),
        ("has_answer", "answer_rate"),
        ("is_correct", "answer_correct_rate"),
    ]:
        writer.add_scalar(f"rollout/{label}",
                          sum(m[key] for m in all_metrics) / n, step)

    # Counts
    for key in ("unique_segment_count", "duplicate_seg_count", "round_count"):
        vals = [m[key] for m in all_metrics]
        writer.add_scalar(f"rollout/{key}", sum(vals) / len(vals), step)

    # Console one-liner (paper-aligned)
    avg_rt = sum(m["rollout_total"] for m in all_metrics) / n
    avg_fmt = sum(m["format"] for m in all_metrics) / n
    avg_cst = sum(m["consistency"] for m in all_metrics) / n
    avg_acc = sum(m["accuracy"] for m in all_metrics) / n
    avg_seg = sum(m["segment"] for m in all_metrics) / n
    n_correct = sum(m["is_correct"] for m in all_metrics)
    avg_uniq = sum(m["unique_segment_count"] for m in all_metrics) / n
    avg_dup = sum(m["duplicate_seg_count"] for m in all_metrics) / n
    kl_val = loss_dict["kl"].item()

    print(
        f"  step {step:3d} | loss {loss_dict['loss'].item():.4f} | "
        f"R {avg_rt:+.3f} (fmt {avg_fmt:+.2f} cst {avg_cst:+.2f} "
        f"acc {avg_acc:.2f} seg {avg_seg:.2f}) | "
        f"correct {n_correct}/{n} | uniq {avg_uniq:.1f} dup {avg_dup:.1f} | "
        f"KL {kl_val:.4f}"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the most recent checkpoint path, or None."""
    ckpt_root = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    steps = []
    for name in os.listdir(ckpt_root):
        if name.startswith("step_") or name == "final":
            p = os.path.join(ckpt_root, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "adapter_config.json")):
                steps.append((os.path.getmtime(p), p))
    steps.sort(reverse=True)
    return steps[0][1] if steps else None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # Enable expandable segments for better GPU memory management
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Output: {args.output_dir}")
    print(f"  Config: {vars(args)}")

    # ── seed ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── load models ──
    print(f"[{datetime.now()}] Loading policy model (trainable LoRA from SFT checkpoint) ...")
    t0 = time.time()
    policy_model, processor = load_policy_model(
        args.model_path, args.adapter_path,
    )
    print(f"  Policy model loaded in {time.time() - t0:.1f}s")

    # Reference model (same checkpoint, frozen, kept on GPU throughout)
    print(f"[{datetime.now()}] Loading reference model ...")
    t0 = time.time()
    ref_model = load_reference_model(args.model_path, args.adapter_path, device)
    print(f"  Reference model loaded in {time.time() - t0:.1f}s")

    # ── load data ──
    dataset = load_dataset(args.data_path)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
        print(f"  Limited to first {args.max_samples} samples")
    print(f"  Data columns: {list(dataset[0].keys()) if dataset else 'empty'}")

    # ── optimiser ──
    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=args.learning_rate
    )

    # ── logging ──
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
    # save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── training loop ──
    tokenizer = processor.tokenizer
    global_step = 0
    skipped_samples = set()  # sample IDs that crash every rollout
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    print(f"\n{'='*60}")
    print(f"  Training: {len(dataset)} samples, {args.num_epochs} epoch(s), "
          f"{total_batches} batches/epoch")
    print(f"  Rollouts/query: {args.num_rollouts} → "
          f"{args.num_rollouts * args.batch_size} generations/step")
    print(f"{'='*60}\n")

    for epoch in range(args.num_epochs):
        random.shuffle(dataset)
        epoch_start = time.time()

        for batch_idx in range(0, len(dataset), args.batch_size):
            step_start = time.time()
            batch = dataset[batch_idx:batch_idx + args.batch_size]
            current_batch_size = len(batch)

            # ════════════════════════════════════════════
            # Phase 1: Generate rollouts
            # ════════════════════════════════════════════
            all_results: List[Tuple[dict, dict]] = []  # (result, sample)
            policy_model.eval()

            for sample_idx, sample in enumerate(batch):
                sid = sample.get("id", "?")
                if sid in skipped_samples:
                    print(f"  [{epoch}.{batch_idx//args.batch_size}.{sample_idx}.skip] {sid} (previously failed, skipping)")
                    for _ in range(args.num_rollouts):
                        all_results.append(({
                            "final_response": "", "pred_answer": "",
                            "total_rounds": 0, "used_segments": [],
                            "stop_reason": "skipped",
                        }, sample))
                    continue
                sample_failed = False
                for r_idx in range(args.num_rollouts):
                    if sample_failed:
                        print(f"  [{epoch}.{batch_idx//args.batch_size}.{sample_idx}.{r_idx}] "
                              f"{sid[:50]}  SKIP (sample already failed)")
                        all_results.append(({
                            "final_response": "", "pred_answer": "",
                            "total_rounds": 0, "used_segments": [],
                            "stop_reason": "error",
                        }, sample))
                        continue
                    print(f"  [{epoch}.{batch_idx//args.batch_size}.{sample_idx}.{r_idx}] "
                          f"{sid[:50]}", end="", flush=True)
                    t_gen = time.time()
                    try:
                        with torch.no_grad():
                            result = run_interleaved(
                                policy_model, processor,
                                audio_path=sample["audio_path"],
                                question=sample["question"],
                                choices=sample["choices"],
                                gold_answer=sample["answer"],
                                max_rounds=args.max_rounds,
                                max_new_tokens_per_round=args.max_new_tokens,
                                temperature=args.temperature,
                                tmp_dir=os.path.join(args.output_dir, "tmp"),
                                on_duplicate_seg="stop",
                                finalize_on_stop=True,
                                finalize_max_new_tokens=args.finalize_max_new_tokens,
                            )
                            all_results.append((result, sample))
                        gen_time = time.time() - t_gen
                        pred = result.get("pred_answer", "?")
                        rounds = result.get("total_rounds", 0)
                        segs = len(result.get("used_segments", []))
                        print(f"  {gen_time:.1f}s  rounds={rounds}  segs={segs}  pred={pred[:40]}")
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        # Flush async CUDA errors to avoid contamination
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        sample_failed = True
                        all_results.append(({
                            "final_response": "",
                            "pred_answer": "",
                            "total_rounds": 0,
                            "used_segments": [],
                            "stop_reason": "error",
                        }, sample))

            # Detect CUDA device-side asserts during rollouts
            cuda_error_detected = False
            for r, s in all_results:
                if r.get("stop_reason") == "error":
                    cuda_error_detected = True
                    break

            # Free GPU memory from rollout audio/generation artefacts
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                print(f"  WARNING: CUDA synchronize failed (async errors): {e}")
                cuda_error_detected = True
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"  WARNING: CUDA empty_cache failed: {e}")
                cuda_error_detected = True
            gc.collect()

            # Check for samples where ALL rollouts failed → skip in future
            failed_ids = set()
            for i in range(current_batch_size):
                sample_rollouts = all_results[i * args.num_rollouts:(i + 1) * args.num_rollouts]
                all_failed = all(r.get("stop_reason") == "error" for r, _ in sample_rollouts)
                if all_failed and sample_rollouts:
                    sid = sample_rollouts[0][1].get("id", "?")
                    failed_ids.add(sid)
                    print(f"  SKIP-LIST: '{sid}' (all {len(sample_rollouts)} rollouts failed)")
            if failed_ids:
                skipped_samples.update(failed_ids)

            # ── CUDA error recovery: reload models & skip batch ──
            if cuda_error_detected:
                print(f"  CUDA error detected in batch — reloading models to recover ...")
                # Release all CUDA resources
                del all_results
                del encoded
                del policy_model
                del ref_model
                gc.collect()
                for _ in range(3):
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                gc.collect()

                try:
                    latest_ckpt = _find_latest_checkpoint(args.output_dir)
                    load_path = latest_ckpt if latest_ckpt else args.adapter_path
                    print(f"  Reloading policy from: {load_path}")
                    policy_model, processor = load_policy_model(args.model_path, load_path)
                    ref_model = load_reference_model(args.model_path, args.adapter_path, device)
                    tokenizer = processor.tokenizer
                    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
                    print(f"  Models reloaded, skipping batch")
                except Exception as e:
                    print(f"  Model reload failed: {e}")
                    print(f"  CUDA context unrecoverable — exiting training")
                    break
                continue  # skip training for this batch

            # Switch to train mode and keep it throughout Phase 3–5.
            # This avoids a CUDA state corruption seen when calling thinker()
            # in eval mode after many generate() calls, then switching to train.
            policy_model.train()

            # ════════════════════════════════════════════
            # Phase 2: Compute rewards
            # ════════════════════════════════════════════
            all_metrics: List[Dict[str, Any]] = []
            for result, sample in all_results:
                meta = build_rollout_metadata(result)
                rew = rollout_reward(
                    result.get("final_response", ""),
                    sample.get("answer", ""),
                    meta,
                )
                all_metrics.append(parse_rollout_metrics(result, sample, rew))

            # ════════════════════════════════════════════
            # Phase 3–5: Micro-batched log-probs + GRPO loss + update
            # ════════════════════════════════════════════
            # Evidence from Exp A/B/C:
            #   - thinker() forward with requires_grad=True crashes above a
            #     (B, T) threshold (~8×538 with 80 GB A800)
            #   - This is NOT caused by rollout-phase GPU fragmentation (Exp B)
            #     nor by audio token patterns (Exp C)
            #   - bs=4 is the safe ceiling; micro-batch + gradient accumulation
            #     gives mathematically equivalent gradients to a full batch
            #     forward (see tasks/08_grpo_smoke_plan.md §22-23)

            # Encode all rollouts
            tokenizer = processor.tokenizer
            encoded = []  # (full_ids, seq_len, prompt_len)
            for result, sample in all_results:
                prompt = build_text_prompt(sample["question"], sample.get("choices", []))
                completion = result.get("final_response", "")
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
                if len(full_ids) > args.max_text_length:
                    full_ids = full_ids[:args.max_text_length]
                seq_len = len(full_ids)
                encoded.append((full_ids, seq_len, len(prompt_ids)))

            B_total = len(encoded)  # = batch_size × num_rollouts

            # Pre-compute total masked tokens for loss normalisation
            # (needed before the micro-batch loop so each sub-loss is
            #  scaled correctly: loss_mb * n_mb / N_total)
            total_masked_tokens = 0
            for _full_ids, seq_len, prompt_len in encoded:
                if seq_len - 1 > prompt_len - 1:
                    total_masked_tokens += (seq_len - 1) - (prompt_len - 1)

            # Advantages (Phase 4 — still group-normalised per query)
            rollout_totals = [m["rollout_total"] for m in all_metrics]
            group_ids = [i // args.num_rollouts for i in range(len(all_results))]
            advantages = compute_advantages(rollout_totals, group_ids).to(device)

            micro_bs = args.policy_forward_micro_batch_size
            num_micro = (B_total + micro_bs - 1) // micro_bs

            policy_model.train()
            optimizer.zero_grad()

            # Accumulators for logging (weighted by n_tokens)
            log_loss_sum = 0.0
            log_kl_sum = 0.0
            log_ratio_sum = 0.0
            log_tokens = 0

            for mb_idx in range(0, B_total, micro_bs):
                mb_encoded = encoded[mb_idx:mb_idx + micro_bs]
                mb_adv = advantages[mb_idx:mb_idx + micro_bs]
                mb_bs = len(mb_encoded)

                # Re-pad to this micro-batch's own max length (saves memory)
                mb_max_len = max(seq_len for _full_ids, seq_len, _pl in mb_encoded)
                mb_ids = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
                mb_attn = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
                mb_cmask = torch.zeros((mb_bs, mb_max_len - 1), dtype=torch.float, device=device)
                for j, (full_ids, seq_len, prompt_len) in enumerate(mb_encoded):
                    mb_ids[j, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
                    mb_attn[j, :seq_len] = 1
                    if seq_len - 1 > prompt_len - 1:
                        mb_cmask[j, prompt_len - 1: seq_len - 1] = 1.0

                gc.collect()
                torch.cuda.synchronize()

                # Policy forward (WITH grad) — micro-batch
                mb_policy_logps = get_per_token_logps(policy_model, mb_ids, mb_attn)
                mb_old_logps = mb_policy_logps.detach()

                # Reference forward (NO grad) — micro-batch
                with torch.no_grad():
                    mb_ref_logps = get_per_token_logps(ref_model, mb_ids, mb_attn)

                # GRPO loss for this micro-batch
                loss_dict = compute_grpo_loss(
                    mb_policy_logps, mb_old_logps, mb_adv,
                    ref_logps=mb_ref_logps, beta=args.kl_coef, epsilon=0.2,
                    mask=mb_cmask,
                )

                n_tokens = mb_cmask.sum()
                if n_tokens > 0 and total_masked_tokens > 0:
                    scale = n_tokens / total_masked_tokens
                    scaled_loss = loss_dict["loss"] * scale
                elif total_masked_tokens == 0:
                    scaled_loss = loss_dict["loss"] / num_micro
                else:
                    scaled_loss = loss_dict["loss"]

                scaled_loss.backward()  # gradient accumulation

                # Accumulate logging metrics (detach)
                log_loss_sum += loss_dict["loss"].detach().item() * n_tokens.item()
                log_kl_sum += loss_dict["kl"].detach().item() * n_tokens.item()
                log_ratio_sum += loss_dict["ratio"].detach().item() * n_tokens.item()
                log_tokens += n_tokens.item()

                # Free micro-batch tensors
                del mb_policy_logps, mb_old_logps, mb_ref_logps, loss_dict, scaled_loss
                torch.cuda.empty_cache()

            # ── Single step after all micro-batches ──
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            # Build aggregated loss_dict for logging
            agg_loss_dict = {
                "loss": torch.tensor(log_loss_sum / max(log_tokens, 1)),
                "kl": torch.tensor(log_kl_sum / max(log_tokens, 1)),
                "ratio": torch.tensor(log_ratio_sum / max(log_tokens, 1)),
            }

            # Print micro-batch diagnostic
            print(f"  micro-batches {num_micro} × bs≤{micro_bs} | "
                  f"max_T={max(seq_len for _, seq_len, _ in encoded)} | "
                  f"total_masked_tokens={total_masked_tokens}")

            policy_model.eval()
            torch.cuda.empty_cache()

            # ════════════════════════════════════════════
            # Phase 6: Log
            # ════════════════════════════════════════════
            step_time = time.time() - step_start
            extra = {
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "learning_rate": args.learning_rate,
                "epoch": epoch + batch_idx / len(dataset),
                "step_time_s": step_time,
                "num_microbatches": num_micro,
                "micro_batch_size": micro_bs,
            }
            log_metrics(writer, global_step, agg_loss_dict, all_metrics, extra)

            # Log per-rollout details to JSONL
            log_path = os.path.join(args.output_dir, "logs", "rollouts.jsonl")
            with open(log_path, "a") as f:
                for m in all_metrics:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            # ── save checkpoint every 5 steps ──
            if global_step > 0 and global_step % args.checkpoint_every == 0:
                ckpt_dir = os.path.join(
                    args.output_dir, "checkpoints", f"step_{global_step}"
                )
                policy_model.save_pretrained(ckpt_dir)
                print(f"  → Checkpoint saved: {ckpt_dir}")

            global_step += 1
            if 0 < args.max_steps <= global_step:
                print(f"\n  Reached --max_steps {args.max_steps}, stopping.")
                break

        if 0 < args.max_steps <= global_step:
            break

        # End of epoch
        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch + 1} done in {epoch_time:.0f}s\n")

    # ── final save ──
    final_ckpt = os.path.join(args.output_dir, "checkpoints", "final")
    policy_model.save_pretrained(final_ckpt)
    print(f"  Final checkpoint: {final_ckpt}")
    print(f"  Done. Logs: {os.path.join(args.output_dir, 'logs')}")


if __name__ == "__main__":
    main()

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
                for r_idx in range(args.num_rollouts):
                    print(f"  [{epoch}.{batch_idx//args.batch_size}.{sample_idx}.{r_idx}] "
                          f"{sample.get('id', '?')[:50]}", end="", flush=True)
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
                        # Placeholder result so group indexing stays valid
                        dummy = {
                            "final_response": "",
                            "pred_answer": "",
                            "total_rounds": 0,
                            "used_segments": [],
                            "stop_reason": "error",
                        }
                        all_results.append((dummy, sample))

            # Free GPU memory from rollout audio/generation artefacts
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

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
            # Phase 3+5: Text log-probs + GRPO loss + update
            # ════════════════════════════════════════════
            # Build batched inputs for all rollouts, then run a SINGLE
            # thinker() forward per model.  This avoids the CUDA error
            # triggered by 14+ separate forward() calls on the thinker
            # with gradient tracking enabled.
            text_pairs: List[Tuple[str, str]] = []
            for result, sample in all_results:
                prompt = build_text_prompt(sample["question"], sample.get("choices", []))
                completion = result.get("final_response", "")
                text_pairs.append((prompt, completion))

            # Encode all pairs
            tokenizer = processor.tokenizer
            encoded = []  # (full_ids, seq_len, prompt_len)
            max_seq_len = 0
            for prompt, completion in text_pairs:
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
                if len(full_ids) > args.max_text_length:
                    full_ids = full_ids[:args.max_text_length]
                seq_len = len(full_ids)
                max_seq_len = max(max_seq_len, seq_len)
                encoded.append((full_ids, seq_len, len(prompt_ids)))

            # Pad to max_seq_len × B
            B = len(encoded)
            padded_ids = torch.zeros((B, max_seq_len), dtype=torch.long, device=device)
            padded_attn = torch.zeros((B, max_seq_len), dtype=torch.long, device=device)
            padded_cmask = torch.zeros((B, max_seq_len - 1), dtype=torch.float, device=device)
            for i, (full_ids, seq_len, prompt_len) in enumerate(encoded):
                padded_ids[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
                padded_attn[i, :seq_len] = 1
                # completion_mask (shifted): position t predicts token t+1
                # Completion starts at prompt_len, so mask positions [prompt_len-1 .. seq_len-2]
                if seq_len - 1 > prompt_len - 1:
                    padded_cmask[i, prompt_len - 1: seq_len - 1] = 1.0

            policy_model.train()
            optimizer.zero_grad()

            gc.collect()
            torch.cuda.synchronize()

            # Single batched forward — policy (with grad)
            try:
                policy_logps_padded = get_per_token_logps(
                    policy_model, padded_ids, padded_attn
                )
            except RuntimeError as e:
                print(f"\n  [Batched ERROR] policy_model logps:")
                print(f"    padded_ids shape: {padded_ids.shape} device: {padded_ids.device}")
                print(f"    padded_attn shape: {padded_attn.shape} device: {padded_attn.device}")
                for n, p in policy_model.named_parameters():
                    if p.requires_grad:
                        print(f"    LoRA param {n}: dtype={p.dtype} device={p.device} isnan={torch.isnan(p).any()} isinf={torch.isinf(p).any()}")
                raise

            old_logps_padded = policy_logps_padded.detach()

            # Single batched forward — reference (no grad)
            with torch.no_grad():
                ref_logps_padded = get_per_token_logps(
                    ref_model, padded_ids, padded_attn
                )

            masks_padded = padded_cmask

            # ════════════════════════════════════════════
            # Phase 4: Advantages
            # ════════════════════════════════════════════
            rollout_totals = [m["rollout_total"] for m in all_metrics]
            group_ids = [i // args.num_rollouts for i in range(len(all_results))]
            advantages = compute_advantages(rollout_totals, group_ids).to(device)

            # ════════════════════════════════════════════
            # Loss + backward + step
            # ════════════════════════════════════════════
            loss_dict = compute_grpo_loss(
                policy_logps_padded,
                old_logps_padded,
                advantages,
                ref_logps=ref_logps_padded,
                beta=args.kl_coef,
                epsilon=0.2,
                mask=masks_padded,
            )

            loss = loss_dict["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            # Free computation graph and intermediate tensors
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
            }
            log_metrics(writer, global_step, loss_dict, all_metrics, extra)

            # Log per-rollout details to JSONL
            log_path = os.path.join(args.output_dir, "logs", "rollouts.jsonl")
            with open(log_path, "a") as f:
                for m in all_metrics:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            # ── save checkpoint every 5 steps ──
            if global_step > 0 and global_step % 5 == 0:
                ckpt_dir = os.path.join(
                    args.output_dir, "checkpoints", f"step_{global_step}"
                )
                policy_model.save_pretrained(ckpt_dir)
                print(f"  → Checkpoint saved: {ckpt_dir}")

            global_step += 1

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

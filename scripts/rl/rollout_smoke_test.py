#!/usr/bin/env python3
"""
GRPO smoke test with subprocess-isolated rollouts.

Each sample's rollouts run in a separate Python subprocess so that
CUDA device-side asserts cannot contaminate the main training process.

Design:
  - Workers run FIRST (main process has NO models on GPU)
  - After workers complete, main process loads models → trains → unloads
  - This ensures complete CUDA context isolation
"""
from __future__ import annotations

import argparse
import json
import os
import gc
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_grpo_smoke"))
from echo_rl.rollout_rewards import rollout_reward
from grpo_utils import (
    build_rollout_metadata,
    build_text_inputs,
    build_text_prompt,
    compute_advantages,
    compute_grpo_loss,
    get_per_token_logps,
)

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "isolated_rollout_worker.py")


# ── args ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--data_path", default="dataJson/NAQA/EAQA_RL.jsonl")
    p.add_argument("--output_dir", default="output/grpo_isolated_smoke")
    p.add_argument("--max_samples", type=int, default=20)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--kl_coef", type=float, default=0.04)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rounds", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
    p.add_argument("--worker_timeout", type=int, default=600)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--policy_forward_micro_batch_size", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--checkpoint_every", type=int, default=30)
    return p.parse_args()


# ── data ──

def load_dataset(path: str, max_samples: int) -> List[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if "multi_choice" in s and "choices" not in s:
                s["choices"] = s["multi_choice"]
            if os.path.exists(s.get("audio_path", "")):
                samples.append(s)
            if len(samples) >= max_samples:
                break
    return samples


# ── worker runner ──

def run_worker(
    sample: dict, model_path: str, adapter_path: str,
    gpu_id: int,
    max_rounds: int, max_new_tokens: int, num_rollouts: int,
    temperature: float, finalize_max_new_tokens: int, timeout: int,
) -> dict:
    sample_id = sample.get("id", "?")

    cmd = [
        sys.executable, "-u", WORKER_SCRIPT,
        "--sample_json", json.dumps(sample, ensure_ascii=False),
        "--model_path", model_path,
        "--adapter_path", adapter_path,
        "--max_rounds", str(max_rounds),
        "--max_new_tokens", str(max_new_tokens),
        "--num_generations", str(num_rollouts),
        "--temperature", str(temperature),
        "--finalize_max_new_tokens", str(finalize_max_new_tokens),
        "--timeout", str(timeout - 10),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["QWEN_OMNI_SKIP_SPK"] = "1"

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        else:
            return {
                "sample_id": sample_id, "rollouts": [],
                "worker_error": f"no stdout; stderr={proc.stderr[-300:]}",
            }
    except subprocess.TimeoutExpired:
        return {"sample_id": sample_id, "rollouts": [], "worker_error": "timeout"}
    except json.JSONDecodeError:
        return {"sample_id": sample_id, "rollouts": [],
                "worker_error": f"bad json; stdout={proc.stdout[-300:]} stderr={proc.stderr[-300:]}"}
    except Exception as e:
        return {"sample_id": sample_id, "rollouts": [], "worker_error": str(e)[:300]}


# ── model load/unload for training ──

def load_training_models(model_path: str, adapter_path: str, device: torch.device):
    """Load policy + reference models for text-only training."""
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )
    policy_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    policy_model.base_model.disable_talker()
    policy_model = policy_model.to(device, dtype=torch.float16)
    for n, p in policy_model.named_parameters():
        p.requires_grad_("lora" in n)
    policy_model.eval()

    base2 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )
    ref_model = PeftModel.from_pretrained(base2, adapter_path)
    ref_model.base_model.disable_talker()
    ref_model = ref_model.to(device, dtype=torch.float16)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return policy_model, ref_model, processor


def unload_training_models(policy_model, ref_model):
    """Release all GPU memory held by training models."""
    del policy_model
    del ref_model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


# ── main ──

def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Config: {vars(args)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.data_path, args.max_samples)
    print(f"  Dataset: {len(dataset)} samples")

    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    print(f"\n{'='*60}")
    print(f"  Isolated rollout smoke test")
    print(f"  {len(dataset)} samples, {args.num_epochs} epoch(s), "
          f"{total_batches} batches")
    print(f"  Workers per batch: {args.batch_size}")
    print(f"  Rollouts per sample: {args.num_rollouts}")
    print(f"{'='*60}\n")

    for epoch in range(args.num_epochs):
        random.shuffle(dataset)
        epoch_start = time.time()

        for batch_idx in range(0, len(dataset), args.batch_size):
            batch_start = time.time()
            batch = dataset[batch_idx:batch_idx + args.batch_size]
            current_batch_size = len(batch)

            # ═══ Phase 1: Run workers (NO models on GPU yet) ═══
            print(f"\n  ── Batch {batch_idx // args.batch_size}: "
                  f"spawning {current_batch_size} workers ──")
            worker_results: List[dict] = []
            for sample in batch:
                sid = sample.get("id", "?")
                t_w = time.time()
                wres = run_worker(
                    sample, args.model_path, args.adapter_path,
                    args.gpu_id,
                    args.max_rounds, args.max_new_tokens, args.num_rollouts,
                    args.temperature, args.finalize_max_new_tokens,
                    args.worker_timeout,
                )
                w_elapsed = time.time() - t_w
                n_ok = sum(1 for r in wres.get("rollouts", [])
                          if r.get("stop_reason") != "error")
                err = wres.get("worker_error", "")
                status = "OK" if not err else f"ERR: {err[:60]}"
                print(f"    {sid[:40]}: {n_ok}/{args.num_rollouts} ok  "
                      f"{w_elapsed:.1f}s  {status}")
                worker_results.append(wres)

            # Count successes
            total_worker_ok = sum(
                1 for wr in worker_results if not wr.get("worker_error")
            )
            print(f"  Workers done: {total_worker_ok}/{current_batch_size} succeeded")

            # ═══ Phase 2: Flatten results ═══
            all_results: List[Tuple[dict, dict]] = []
            for wres, sample in zip(worker_results, batch):
                rollouts = wres.get("rollouts", [])
                while len(rollouts) < args.num_rollouts:
                    rollouts.append({
                        "final_response": "", "pred_answer": "",
                        "total_rounds": 0, "used_segments": [],
                        "stop_reason": "error",
                    })
                for r in rollouts:
                    all_results.append((r, sample))

            # ═══ Phase 3: Load models, rewards, training ═══
            print(f"\n  [{datetime.now()}] Loading training models ...")
            t_load = time.time()
            policy_model, ref_model, processor = load_training_models(
                args.model_path, args.adapter_path, device,
            )
            tokenizer = processor.tokenizer
            optimizer = torch.optim.AdamW(
                policy_model.parameters(), lr=args.learning_rate,
            )
            trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
            print(f"  Models loaded in {time.time() - t_load:.1f}s "
                  f"(trainable: {trainable:,})")

            try:
                # Rewards
                all_metrics: List[Dict[str, Any]] = []
                for rollout_data, sample in all_results:
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
                        "has_answer": int(bool(rollout_data.get("pred_answer", ""))),
                        "is_correct": int(rollout_data.get("pred_answer", "") == sample.get("answer", "")),
                        "round_count": rollout_data.get("total_rounds", 0),
                        "stop_reason": rollout_data.get("stop_reason", "error"),
                        "triggered_interleaved": rollout_data.get("triggered_interleaved", False),
                        "has_final_answer": rollout_data.get("has_final_answer", False),
                        "answer_correct": rollout_data.get("answer_correct", False),
                        "final_response": resp,
                        "used_segments": rollout_data.get("used_segments", []),
                        "round_outputs": rollout_data.get("round_outputs", []),
                    })

                # Encode
                encoded = []
                for rollout_data, sample in all_results:
                    prompt = build_text_prompt(
                        sample["question"], sample.get("choices", []),
                    )
                    completion = rollout_data.get("final_response", "")
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    full_ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
                    full_ids = full_ids[:2048]
                    encoded.append((full_ids, len(full_ids), len(prompt_ids)))

                B_total = len(encoded)
                total_masked_tokens = sum(
                    max(0, (sl - 1) - (pl - 1)) for _, sl, pl in encoded
                )

                # Advantages
                rollout_totals = [m["rollout_total"] for m in all_metrics]
                group_ids = [i // args.num_rollouts for i in range(len(all_results))]
                advantages = compute_advantages(rollout_totals, group_ids).to(device)

                # Micro-batch forward/backward
                policy_model.train()
                micro_bs = args.policy_forward_micro_batch_size
                num_micro = (B_total + micro_bs - 1) // micro_bs
                optimizer.zero_grad()

                log_loss_sum = log_kl_sum = log_ratio_sum = 0.0
                log_tokens = 0

                for mb_idx in range(0, B_total, micro_bs):
                    mb_encoded = encoded[mb_idx:mb_idx + micro_bs]
                    mb_adv = advantages[mb_idx:mb_idx + micro_bs]
                    mb_bs = len(mb_encoded)
                    mb_max_len = max(sl for _, sl, _ in mb_encoded)

                    mb_ids = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
                    mb_attn = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
                    mb_cmask = torch.zeros((mb_bs, mb_max_len - 1), dtype=torch.float, device=device)
                    for j, (full_ids, seq_len, prompt_len) in enumerate(mb_encoded):
                        mb_ids[j, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
                        mb_attn[j, :seq_len] = 1
                        if seq_len - 1 > prompt_len - 1:
                            mb_cmask[j, prompt_len - 1:seq_len - 1] = 1.0

                    mb_policy_logps = get_per_token_logps(policy_model, mb_ids, mb_attn)
                    mb_old_logps = mb_policy_logps.detach()
                    with torch.no_grad():
                        mb_ref_logps = get_per_token_logps(ref_model, mb_ids, mb_attn)

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
                    log_ratio_sum += loss_dict["ratio"].detach().item() * nt
                    log_tokens += nt

                    del mb_policy_logps, mb_old_logps, mb_ref_logps, loss_dict

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), args.max_grad_norm,
                )
                optimizer.step()

                # ── Log ──
                agg_loss = log_loss_sum / max(log_tokens, 1)
                agg_kl = log_kl_sum / max(log_tokens, 1)

                n = len(all_metrics)
                avg_rt = sum(m["rollout_total"] for m in all_metrics) / max(n, 1)
                avg_fmt = sum(m["format"] for m in all_metrics) / max(n, 1)
                avg_cst = sum(m["consistency"] for m in all_metrics) / max(n, 1)
                avg_acc = sum(m["accuracy"] for m in all_metrics) / max(n, 1)
                avg_seg = sum(m["segment"] for m in all_metrics) / max(n, 1)
                n_correct = sum(m["is_correct"] for m in all_metrics)

                batch_time = time.time() - batch_start
                print(f"  step {global_step:3d} | loss {agg_loss:.4f} | "
                      f"R {avg_rt:+.3f} (fmt {avg_fmt:+.2f} cst {avg_cst:+.2f} "
                      f"acc {avg_acc:.2f} seg {avg_seg:.2f}) | "
                      f"correct {n_correct}/{n} | KL {agg_kl:.4f} | "
                      f"{batch_time:.1f}s")

                writer.add_scalar("train/loss", agg_loss, global_step)
                writer.add_scalar("train/approx_kl", agg_kl, global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                for key in ("rollout_total", "format", "consistency", "accuracy", "segment"):
                    vals = [m[key] for m in all_metrics]
                    writer.add_scalar(f"reward/{key}", sum(vals) / max(len(vals), 1), global_step)

                if global_step > 0 and global_step % args.checkpoint_every == 0:
                    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
                    policy_model.save_pretrained(ckpt_dir)
                    print(f"  → Checkpoint: {ckpt_dir}")

                log_path = os.path.join(args.output_dir, "logs", "rollouts.jsonl")
                with open(log_path, "a") as f:
                    for m in all_metrics:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")

            finally:
                # Always unload training models to free GPU for next workers
                unload_training_models(policy_model, ref_model)
                torch.cuda.empty_cache()
                gc.collect()

            global_step += 1
            if 0 < args.max_steps <= global_step:
                print(f"\n  Reached --max_steps {args.max_steps}, stopping.")
                break

        if 0 < args.max_steps <= global_step:
            break

        print(f"\n  Epoch {epoch + 1} done in {time.time() - epoch_start:.0f}s\n")

    print(f"  Done. Logs: {os.path.join(args.output_dir, 'logs')}")


if __name__ == "__main__":
    main()

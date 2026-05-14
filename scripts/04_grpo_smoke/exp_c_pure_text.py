#!/usr/bin/env python3
"""Experiment C: pure-text forward without any interleaved audio.

No rollout phase. No audio files. Load model fresh, create synthetic
text prompts + completions, do thinker() forward.

Tests whether the CUDA error is tied to:
  a) The interleaved audio pipeline (rollout phase corrupts CUDA state)
  b) The specific token patterns from interleaved generation
  c) Purely the forward pass itself with (B, T) padded inputs

Sweeps batch sizes: 1, 2, 4, 8, 16, 32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import gc
import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from grpo_utils import get_per_token_logps


def parse_args():
    p = argparse.ArgumentParser(description="Exp C: pure-text forward, no audio")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--output_dir", default="output/grpo_smoke/exp_c")
    p.add_argument("--prompt_len", type=int, default=200)
    p.add_argument("--completion_len", type=int, default=400)
    p.add_argument("--max_text_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def log_gpu_mem(tag: str) -> dict:
    return {
        "tag": tag,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    report = {
        "experiment": "C: pure-text forward, no audio interleaving",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "sweep_results": [],
    }

    print(f"[{datetime.now()}] Exp C: pure text forward")
    print(f"  Device: {device}")

    # ── Load model (FRESH, no rollouts ever) ──
    print(f"[{datetime.now()}] Loading policy model ...")
    t0 = time.time()
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    model.base_model.disable_talker()
    model = model.to("cuda:0", dtype=torch.float16)
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    report["mem_after_load"] = log_gpu_mem("after_load")

    # ── Build synthetic text inputs (no audio, no rollouts) ──
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Use a realistic math/QA prompt (mirrors actual task structure)
    base_prompt = (
        "A 10-second audio contains four distinct events. "
        "Event A occurs from 0.0s to 3.5s. Event B occurs from 2.0s to 6.0s. "
        "Event C occurs from 5.0s to 8.0s. Event D occurs from 7.0s to 10.0s. "
        "You are asked to find the total duration during which events A and B overlap. "
        "Choose the answer from ['1.5 seconds', '2.0 seconds', '3.5 seconds', '4.0 seconds']. "
        "Think step-by-step. Refer to the specific audio segments while thinking."
    )

    base_completion = (
        "<think>Event A spans 0.0-3.5s. Event B spans 2.0-6.0s. "
        "The overlap is from max(0.0, 2.0)=2.0s to min(3.5, 6.0)=3.5s. "
        "Duration = 3.5 - 2.0 = 1.5 seconds. "
        "Therefore the overlap duration is 1.5 seconds.</think>"
        "<answer>1.5 seconds</answer>"
    )

    # Create 32 samples (enough for full batch sweep) with varying lengths
    synthetic_samples = []
    for i in range(32):
        # Vary completion length to create realistic length distribution
        extra = " ".join([f"Step {j}: checking calculation..." for j in range(i % 8)])
        completion = base_completion + " " + extra
        prompt_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        full_ids = prompt_ids + completion_ids
        if len(full_ids) > args.max_text_length:
            full_ids = full_ids[:args.max_text_length]
        synthetic_samples.append((full_ids, len(prompt_ids)))

    # Pad to max length
    max_seq_len = max(len(ids) for ids, _ in synthetic_samples)
    B_total = len(synthetic_samples)
    print(f"  Synthetic samples: {B_total}, max_seq_len: {max_seq_len}")
    print(f"  Sequence lengths: {[len(ids) for ids, _ in synthetic_samples]}")

    padded_ids = torch.zeros((B_total, max_seq_len), dtype=torch.long, device=device)
    padded_attn = torch.zeros((B_total, max_seq_len), dtype=torch.long, device=device)
    for i, (full_ids, _prompt_len) in enumerate(synthetic_samples):
        seq_len = len(full_ids)
        padded_ids[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
        padded_attn[i, :seq_len] = 1

    report["input"] = {
        "total_samples": B_total,
        "max_seq_len": max_seq_len,
        "seq_lens": [len(ids) for ids, _ in synthetic_samples],
        "prompt_lens": [pl for _, pl in synthetic_samples],
        "base_prompt": base_prompt[:200] + "...",
        "base_completion": base_completion[:200] + "...",
    }

    # ── Sweep batch sizes ──
    print(f"\n{'='*60}")
    print(f"  Batch size sweep (pure text, no audio ever)")
    print(f"{'='*60}")

    batch_sizes = [1, 2, 4, 8, 16, 32]

    for bs in batch_sizes:
        if bs > B_total:
            continue

        sub_ids = padded_ids[:bs]
        sub_attn = padded_attn[:bs]

        result_entry = {
            "batch_size": bs,
            "input_shape": list(sub_ids.shape),
            "max_seq_len": max_seq_len,
        }

        print(f"\n  bs={bs}: shape={list(sub_ids.shape)}", end="", flush=True)

        # Policy forward (WITH grad)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()

        mem_before = log_gpu_mem(f"bs{bs}_before")
        result_entry["mem_before"] = mem_before

        try:
            logps = get_per_token_logps(model, sub_ids, sub_attn)
            torch.cuda.synchronize()
            mem_after = log_gpu_mem(f"bs{bs}_after")
            result_entry["policy_ok"] = True
            result_entry["logps_shape"] = list(logps.shape)
            result_entry["mem_after"] = mem_after
            print(f"  OK shape={list(logps.shape)} "
                  f"peak_alloc={mem_after['max_allocated_mb']:.0f}MB", end="")
            del logps
        except RuntimeError as e:
            torch.cuda.synchronize()
            result_entry["policy_ok"] = False
            result_entry["error"] = str(e)[:500]
            print(f"  CRASH: {str(e)[:200]}", end="")
            model.eval()
            report["sweep_results"].append(result_entry)
            continue

        model.eval()
        torch.cuda.empty_cache()

        # Ref forward (NO grad)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_before_ref = log_gpu_mem(f"bs{bs}_ref_before")
        result_entry["mem_before_ref"] = mem_before_ref

        try:
            with torch.no_grad():
                ref_logps = get_per_token_logps(model, sub_ids, sub_attn)
            torch.cuda.synchronize()
            mem_after_ref = log_gpu_mem(f"bs{bs}_ref_after")
            result_entry["ref_ok"] = True
            result_entry["mem_after_ref"] = mem_after_ref
            print(f" | ref_ok peak={mem_after_ref['max_allocated_mb']:.0f}MB")
            del ref_logps
        except RuntimeError as e:
            torch.cuda.synchronize()
            result_entry["ref_ok"] = False
            result_entry["ref_error"] = str(e)[:500]
            print(f" | REF CRASH: {str(e)[:200]}")

        torch.cuda.empty_cache()
        report["sweep_results"].append(result_entry)

    # ── Save report ──
    report_path = os.path.join(args.output_dir, "exp_c_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n\n  Report saved: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Pure-text forward (no audio interleaving)")
    print(f"{'='*60}")
    print(f"  {'bs':<6} {'shape':<18} {'policy':<8} {'ref':<8} {'peak_mem_mb':<14}")
    for r in report["sweep_results"]:
        shape_str = str(r.get("input_shape", "?"))
        policy_str = "OK" if r.get("policy_ok") else "CRASH"
        ref_str = "OK" if r.get("ref_ok") else "CRASH"
        peak = max(
            r.get("mem_after", {}).get("max_allocated_mb", 0),
            r.get("mem_after_ref", {}).get("max_allocated_mb", 0),
        )
        print(f"  {r['batch_size']:<6} {shape_str:<18} {policy_str:<8} {ref_str:<8} {peak:<14.0f}")

    crashed = any(not r.get("policy_ok", True) or not r.get("ref_ok", True)
                  for r in report["sweep_results"])
    if crashed:
        print(f"\n  [Exp C] CRASH detected")
        sys.exit(1)
    else:
        print(f"\n  [Exp C] All batch sizes passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment A: batch size sweep for thinker() forward.

Uses the SAME rollout results for all batch sizes.
Sweeps batch_size = 1, 2, 4, 8, 16, 32 with fixed max_seq_len.
Records: crash, GPU memory, input shape.
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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.interleaved_infer import run_interleaved
from grpo_utils import get_per_token_logps


def parse_args():
    p = argparse.ArgumentParser(description="Exp A: batch size sweep")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--data_path", default="output/judge/split_rl.jsonl")
    p.add_argument("--output_dir", default="output/grpo_smoke/exp_a")
    p.add_argument("--num_rollouts", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
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
        "experiment": "A: batch size sweep",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "sweep_results": [],
    }

    print(f"[{datetime.now()}] Exp A start")
    print(f"  Device: {device}")

    # ── Load policy model ──
    print(f"[{datetime.now()}] Loading policy model ...")
    t0 = time.time()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
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
    print(f"  Policy model loaded in {time.time() - t0:.1f}s")
    report["mem_after_load"] = log_gpu_mem("after_policy_load")

    # ── Load data ──
    samples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                if os.path.exists(s.get("audio_path", "")):
                    samples.append(s)
    print(f"  Dataset: {len(samples)} valid samples")

    # ── Phase 1: Rollouts ──
    batch = samples[:args.batch_size]
    all_results = []  # list of (result_dict, sample_dict)
    model.eval()

    print(f"\n{'='*60}")
    print(f"  Generating {len(batch) * args.num_rollouts} rollouts "
          f"({len(batch)} queries × {args.num_rollouts} each)")
    print(f"{'='*60}")

    for sample in batch:
        for r_idx in range(args.num_rollouts):
            print(f"  [{sample.get('id','?')[:40]}.{r_idx}]", end="", flush=True)
            t_gen = time.time()
            try:
                with torch.no_grad():
                    result = run_interleaved(
                        model, processor,
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
                print(f" {gen_time:.1f}s rounds={result.get('total_rounds',0)}")
            except Exception as e:
                print(f" ERROR: {e}")
                all_results.append(({
                    "final_response": "", "pred_answer": "",
                    "total_rounds": 0, "used_segments": [],
                    "stop_reason": "error",
                }, sample))

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    report["mem_after_rollouts"] = log_gpu_mem("after_rollouts")

    # ── Build padded inputs (B, max_T) from ALL rollouts ──
    tokenizer = processor.tokenizer
    encoded = []
    max_seq_len = 0
    for result, sample in all_results:
        prompt = (sample["question"] + " Choose the answer from "
                  + str(sample.get("choices", []))
                  + ". Think step-by-step.")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_text = result.get("final_response", "")
        full_ids = prompt_ids + tokenizer.encode(completion_text, add_special_tokens=False)
        if len(full_ids) > args.max_text_length:
            full_ids = full_ids[:args.max_text_length]
        seq_len = len(full_ids)
        max_seq_len = max(max_seq_len, seq_len)
        encoded.append((full_ids, seq_len, len(prompt_ids)))

    B_total = len(encoded)
    print(f"\n  Total rollouts: {B_total}, max_seq_len: {max_seq_len}")

    # Create full padded tensors (fixed max_seq_len for all sweeps)
    padded_ids = torch.zeros((B_total, max_seq_len), dtype=torch.long, device=device)
    padded_attn = torch.zeros((B_total, max_seq_len), dtype=torch.long, device=device)
    for i, (full_ids, seq_len, _prompt_len) in enumerate(encoded):
        padded_ids[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
        padded_attn[i, :seq_len] = 1

    report["input"] = {
        "total_rollouts": B_total,
        "max_seq_len": max_seq_len,
        "seq_lens": [e[1] for e in encoded],
        "prompt_lens": [e[2] for e in encoded],
    }

    # ── Sweep batch sizes ──
    print(f"\n{'='*60}")
    print(f"  Batch size sweep (fixed max_T={max_seq_len})")
    print(f"{'='*60}")

    batch_sizes = [1, 2, 4, 8, 16, 32]

    for bs in batch_sizes:
        if bs > B_total:
            print(f"  bs={bs}: SKIP (only {B_total} rollouts available)")
            continue

        sub_ids = padded_ids[:bs]
        sub_attn = padded_attn[:bs]

        result_entry = {
            "batch_size": bs,
            "input_shape": list(sub_ids.shape),
            "max_seq_len": max_seq_len,
        }

        print(f"\n  bs={bs}: shape={list(sub_ids.shape)}", end="", flush=True)

        # ── Policy forward (WITH grad) ──
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.eval()
        # Switch to train mode to enable grad tracking (matches real training)
        model.train()

        mem_before = log_gpu_mem(f"bs{bs}_before_policy")
        result_entry["mem_before_policy"] = mem_before

        try:
            policy_logps = get_per_token_logps(model, sub_ids, sub_attn)
            torch.cuda.synchronize()
            mem_after_policy = log_gpu_mem(f"bs{bs}_after_policy")
            result_entry["mem_after_policy"] = mem_after_policy
            result_entry["policy_ok"] = True
            result_entry["policy_logps_shape"] = list(policy_logps.shape)
            print(f"  policy_ok shape={list(policy_logps.shape)} "
                  f"peak_alloc={mem_after_policy['max_allocated_mb']:.0f}MB", end="")
        except RuntimeError as e:
            torch.cuda.synchronize()
            result_entry["policy_ok"] = False
            result_entry["policy_error"] = str(e)[:300]
            print(f"  POLICY CRASH: {str(e)[:200]}", end="")
            model.eval()
            report["sweep_results"].append(result_entry)
            continue

        # Clean up policy forward tensors
        del policy_logps
        model.eval()
        torch.cuda.empty_cache()

        # ── Ref forward (NO grad) ──
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_before_ref = log_gpu_mem(f"bs{bs}_before_ref")
        result_entry["mem_before_ref"] = mem_before_ref

        try:
            with torch.no_grad():
                ref_logps = get_per_token_logps(model, sub_ids, sub_attn)
            torch.cuda.synchronize()
            mem_after_ref = log_gpu_mem(f"bs{bs}_after_ref")
            result_entry["mem_after_ref"] = mem_after_ref
            result_entry["ref_ok"] = True
            print(f" | ref_ok peak={mem_after_ref['max_allocated_mb']:.0f}MB")
        except RuntimeError as e:
            torch.cuda.synchronize()
            result_entry["ref_ok"] = False
            result_entry["ref_error"] = str(e)[:300]
            print(f" | REF CRASH: {str(e)[:200]}")

        del ref_logps
        torch.cuda.empty_cache()
        report["sweep_results"].append(result_entry)

    # ── Save report ──
    report_path = os.path.join(args.output_dir, "exp_a_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n\n  Report saved: {report_path}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'bs':<6} {'shape':<18} {'policy':<8} {'ref':<8} {'peak_mem_mb':<14}")
    for r in report["sweep_results"]:
        shape_str = str(r.get("input_shape", "?"))
        policy_str = "OK" if r.get("policy_ok") else "CRASH"
        ref_str = "OK" if r.get("ref_ok") else "CRASH"
        peak = max(
            r.get("mem_after_policy", {}).get("max_allocated_mb", 0),
            r.get("mem_after_ref", {}).get("max_allocated_mb", 0),
        )
        print(f"  {r['batch_size']:<6} {shape_str:<18} {policy_str:<8} {ref_str:<8} {peak:<14.0f}")

    # Non-zero exit if any crash (for easy checking)
    crashed = any(not r.get("policy_ok", True) or not r.get("ref_ok", True)
                  for r in report["sweep_results"])
    if crashed:
        print(f"\n  [Exp A] CRASH detected")
        sys.exit(1)
    else:
        print(f"\n  [Exp A] All batch sizes passed")


if __name__ == "__main__":
    main()

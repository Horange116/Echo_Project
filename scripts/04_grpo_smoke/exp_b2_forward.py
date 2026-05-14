#!/usr/bin/env python3
"""Experiment B2: Forward-only stage. Load rollouts from JSONL, run thinker() forward.

Key question: does the CUDA error still occur when the GPU starts FRESH
(no rollout phase in the same process)?
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
from transformers import Qwen2_5OmniForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from grpo_utils import get_per_token_logps


def parse_args():
    p = argparse.ArgumentParser(description="Exp B2: forward-only from saved rollouts")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--rollout_file", default="output/grpo_smoke/exp_b/rollout_outputs.jsonl")
    p.add_argument("--output_dir", default="output/grpo_smoke/exp_b")
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
        "experiment": "B2: forward-only from saved rollouts",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "results": [],
    }

    # ── Load rollout data ──
    print(f"[{datetime.now()}] Exp B2: forward-only")
    print(f"  Loading rollouts from: {args.rollout_file}")

    if not os.path.exists(args.rollout_file):
        print(f"  ERROR: rollout file not found! Run exp_b1_rollout.py first.")
        sys.exit(1)

    rollouts = []
    with open(args.rollout_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    print(f"  Loaded {len(rollouts)} rollouts")

    # ── Load policy model (FRESH session, no rollout generation) ──
    print(f"[{datetime.now()}] Loading policy model (fresh) ...")
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

    # ── Build padded inputs (same logic as main training script) ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    encoded = []
    max_seq_len = 0
    for r in rollouts:
        prompt = (r["question"] + " Choose the answer from "
                  + str(r["choices"])
                  + ". Think step-by-step.")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = prompt_ids + tokenizer.encode(r.get("final_response", ""),
                                                  add_special_tokens=False)
        if len(full_ids) > args.max_text_length:
            full_ids = full_ids[:args.max_text_length]
        seq_len = len(full_ids)
        max_seq_len = max(max_seq_len, seq_len)
        encoded.append((full_ids, seq_len, len(prompt_ids)))

    B = len(encoded)
    print(f"  Total: {B} rollouts, max_seq_len: {max_seq_len}")

    padded_ids = torch.zeros((B, max_seq_len), dtype=torch.long, device=device)
    padded_attn = torch.zeros((B, max_seq_len), dtype=torch.long, device=device)
    for i, (full_ids, seq_len, _prompt_len) in enumerate(encoded):
        padded_ids[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
        padded_attn[i, :seq_len] = 1

    report["input"] = {
        "total_rollouts": B,
        "max_seq_len": max_seq_len,
        "seq_lens": [e[1] for e in encoded],
    }

    # ── Policy forward (WITH grad) ──
    print(f"\n  Forward: policy model, batch={B}, max_T={max_seq_len}")
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.train()  # enable grad tracking = match training conditions
    mem_before = log_gpu_mem("before_policy_forward")

    try:
        policy_logps = get_per_token_logps(model, padded_ids, padded_attn)
        torch.cuda.synchronize()
        mem_after = log_gpu_mem("after_policy_forward")
        report["policy_ok"] = True
        report["policy_logps_shape"] = list(policy_logps.shape)
        report["mem_after_policy"] = mem_after
        print(f"  Policy forward OK: shape={list(policy_logps.shape)} "
              f"peak_alloc={mem_after['max_allocated_mb']:.0f}MB")
        del policy_logps
    except RuntimeError as e:
        torch.cuda.synchronize()
        report["policy_ok"] = False
        report["policy_error"] = str(e)[:500]
        print(f"  POLICY FORWARD CRASH: {str(e)[:300]}")

    model.eval()
    torch.cuda.empty_cache()

    # ── Ref forward (NO grad) ──
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before_ref = log_gpu_mem("before_ref_forward")
    try:
        with torch.no_grad():
            ref_logps = get_per_token_logps(model, padded_ids, padded_attn)
        torch.cuda.synchronize()
        mem_after_ref = log_gpu_mem("after_ref_forward")
        report["ref_ok"] = True
        report["mem_after_ref"] = mem_after_ref
        print(f"  Ref forward OK: peak_alloc={mem_after_ref['max_allocated_mb']:.0f}MB")
        del ref_logps
    except RuntimeError as e:
        torch.cuda.synchronize()
        report["ref_ok"] = False
        report["ref_error"] = str(e)[:500]
        print(f"  REF FORWARD CRASH: {str(e)[:300]}")

    # ── Save report ──
    report_path = os.path.join(args.output_dir, "exp_b2_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")

    # Summary
    policy_ok = report.get("policy_ok", False)
    ref_ok = report.get("ref_ok", False)
    print(f"  Policy: {'OK' if policy_ok else 'CRASH'} | Ref: {'OK' if ref_ok else 'CRASH'}")
    print(f"  [{datetime.now()}] Exp B2 complete")
    sys.exit(0 if (policy_ok and ref_ok) else 1)


if __name__ == "__main__":
    main()

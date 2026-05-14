#!/usr/bin/env python3
"""Experiment B1: Rollout-only stage. Save all rollout results to JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.interleaved_infer import run_interleaved


def parse_args():
    p = argparse.ArgumentParser(description="Exp B1: rollout only → save JSONL")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--data_path", default="output/judge/split_rl.jsonl")
    p.add_argument("--output_dir", default="output/grpo_smoke/exp_b")
    p.add_argument("--num_rollouts", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    print(f"[{datetime.now()}] Exp B1: rollout only")
    print(f"  Device: {device}")

    # ── Load model ──
    print(f"[{datetime.now()}] Loading model ...")
    t0 = time.time()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=False)
    model.base_model.disable_talker()
    model = model.to("cuda:0", dtype=torch.float16)
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

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

    batch = samples[:args.batch_size]
    rollout_file = os.path.join(args.output_dir, "rollout_outputs.jsonl")

    print(f"\n  Generating {len(batch) * args.num_rollouts} rollouts → {rollout_file}")
    print(f"  (Process will EXIT after saving — no forward/backward)")

    with open(rollout_file, "w") as fout:
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
                    gen_time = time.time() - t_gen
                    # Save essential fields for forward stage
                    entry = {
                        "id": sample.get("id", "?"),
                        "question": sample["question"],
                        "choices": sample.get("choices", []),
                        "answer": sample.get("answer", ""),
                        "final_response": result.get("final_response", ""),
                        "pred_answer": result.get("pred_answer", ""),
                        "total_rounds": result.get("total_rounds", 0),
                        "used_segments": result.get("used_segments", []),
                        "stop_reason": result.get("stop_reason", "unknown"),
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f" {gen_time:.1f}s rounds={result.get('total_rounds',0)}")
                except Exception as e:
                    print(f" ERROR: {e}")
                    entry = {
                        "id": sample.get("id", "?"),
                        "question": sample["question"],
                        "choices": sample.get("choices", []),
                        "answer": sample.get("answer", ""),
                        "final_response": "",
                        "pred_answer": "",
                        "total_rounds": 0,
                        "used_segments": [],
                        "stop_reason": "error",
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()

    print(f"\n  Rollouts saved: {rollout_file}")
    print(f"  [{datetime.now()}] Exp B1 complete — EXIT")


if __name__ == "__main__":
    main()

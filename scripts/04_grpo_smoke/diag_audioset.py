#!/usr/bin/env python3
"""Diagnose: which AudioSet samples trigger CUDA device-side assert?"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.interleaved_infer import run_interleaved, load_model_and_processor

model_path = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
adapter_path = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

# Load up to N AudioSet samples
with open("dataJson/NAQA/EAQA_RL.jsonl") as f:
    audioset_samples = []
    for line in f:
        s = json.loads(line.strip())
        if "multi_choice" in s and "choices" not in s:
            s["choices"] = s["multi_choice"]
        if "AudioSet" in s.get("id", ""):
            audioset_samples.append(s)
        if len(audioset_samples) >= 20:
            break

print(f"Testing {len(audioset_samples)} AudioSet samples")
print("Loading model...")
model, processor = load_model_and_processor(model_path, adapter_path)
model.eval()

good, bad = [], []
for s in audioset_samples:
    sid = s["id"]
    fname = s["audio_path"].split("/")[-1]
    try:
        with torch.no_grad():
            result = run_interleaved(model, processor,
                audio_path=s["audio_path"], question=s["question"],
                choices=s["choices"], gold_answer=s["answer"],
                max_rounds=2, max_new_tokens_per_round=64, temperature=0.9,
                on_duplicate_seg="stop", finalize_on_stop=True,
                finalize_max_new_tokens=32)
        print(f"  OK: {sid} ({fname})")
        good.append(sid)
    except Exception as e:
        print(f"  BAD: {sid} ({fname}) — {str(e)[:80]}")
        bad.append(sid)
        # After a CUDA error, context is toast — stop
        if "device-side assert" in str(e) or "CUDA error" in str(e):
            print("  → CUDA context corrupted, stopping scan")
            break

print(f"\nGood: {good}")
print(f"Bad: {bad}")

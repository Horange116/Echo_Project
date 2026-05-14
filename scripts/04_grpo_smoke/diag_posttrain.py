#!/usr/bin/env python3
"""Diagnose: does one training step corrupt generation?"""
import sys, os, json, time, gc, torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from scripts.interleaved_infer import run_interleaved, load_model_and_processor
from grpo_utils import get_per_token_logps, compute_grpo_loss, compute_advantages, build_text_prompt, build_rollout_metadata
from echo_rl.rollout_rewards import rollout_reward

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["QWEN_OMNI_SKIP_SPK"] = "1"

model_path = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B"
adapter_path = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078"

# Load model
print("Loading model (fresh)...")
model, processor = load_model_and_processor(model_path, adapter_path)
model.eval()

# Test sample (known good)
with open("dataJson/NAQA/EAQA_RL.jsonl") as f:
    for line in f:
        s = json.loads(line.strip())
        if "multi_choice" in s and "choices" not in s:
            s["choices"] = s["multi_choice"]
        if s["id"] == "AudioSet_0":
            test_sample = s
            break

print(f"Test sample: {test_sample['id']}")

# --- Pre-training generation test ---
print("\n--- Pre-training generation test ---")
try:
    with torch.no_grad():
        result = run_interleaved(model, processor,
            audio_path=test_sample["audio_path"], question=test_sample["question"],
            choices=test_sample["choices"], gold_answer=test_sample["answer"],
            max_rounds=2, max_new_tokens_per_round=64, temperature=0.9,
            on_duplicate_seg="stop", finalize_on_stop=True, finalize_max_new_tokens=32)
    print(f"  PRE-TRAIN: OK, pred={result.get('pred_answer','?')[:60]}")
except Exception as e:
    print(f"  PRE-TRAIN: FAIL - {e}")
    sys.exit(1)

# --- One GRPO training step ---
print("\n--- Running 1 GRPO training step ---")

# Load a batch of 4 samples
dataset = []
with open("dataJson/NAQA/EAQA_RL.jsonl") as f:
    for line in f:
        s = json.loads(line.strip())
        if "multi_choice" in s and "choices" not in s:
            s["choices"] = s["multi_choice"]
        if os.path.exists(s.get("audio_path", "")):
            dataset.append(s)
        if len(dataset) >= 4:
            break

num_rollouts = 4
max_rounds = 5
temperature = 0.9
max_new_tokens = 128

print("Phase 1: Rollouts...")
all_results = []
model.eval()
for sample in dataset:
    for r_idx in range(num_rollouts):
        try:
            with torch.no_grad():
                result = run_interleaved(model, processor,
                    audio_path=sample["audio_path"], question=sample["question"],
                    choices=sample["choices"], gold_answer=sample["answer"],
                    max_rounds=max_rounds, max_new_tokens_per_round=max_new_tokens,
                    temperature=temperature, on_duplicate_seg="stop",
                    finalize_on_stop=True, finalize_max_new_tokens=64)
            all_results.append((result, sample))
        except Exception as e:
            print(f"  Rollout error: {e}")
            all_results.append(({"final_response": "", "pred_answer": "",
                "total_rounds": 0, "used_segments": [], "stop_reason": "error"}, sample))

torch.cuda.empty_cache()
gc.collect()

# Phase 2: Rewards
print("Phase 2: Rewards...")
all_metrics = []
for result, sample in all_results:
    meta = build_rollout_metadata(result)
    rew = rollout_reward(result.get("final_response", ""), sample.get("answer", ""), meta)
    all_metrics.append(rew)

# Phase 3: Encode
print("Phase 3: Encode...")
tokenizer = processor.tokenizer
encoded = []
for result, sample in all_results:
    prompt = build_text_prompt(sample["question"], sample.get("choices", []))
    completion = result.get("final_response", "")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
    if len(full_ids) > 2048:
        full_ids = full_ids[:2048]
    encoded.append((full_ids, len(full_ids), len(prompt_ids)))

B_total = len(encoded)
total_masked_tokens = sum(max(0, (sl - 1) - (pl - 1)) for _, sl, pl in encoded)

rollout_totals = [m["rollout_total"] for m in all_metrics]
group_ids = [i // num_rollouts for i in range(len(all_results))]
advantages = compute_advantages(rollout_totals, group_ids).cuda()

# Phase 4-5: Micro-batch forward + backward
print("Phase 4-5: Forward/backward...")
model.train()

# Freeze non-LoRA
for n, p in model.named_parameters():
    if "lora" in n:
        p.requires_grad_(True)
    else:
        p.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

optimizer.zero_grad()
micro_bs = 4

for mb_idx in range(0, B_total, micro_bs):
    mb_encoded = encoded[mb_idx:mb_idx + micro_bs]
    mb_adv = advantages[mb_idx:mb_idx + micro_bs]
    mb_bs = len(mb_encoded)
    mb_max_len = max(sl for _, sl, _ in mb_encoded)
    device = torch.device("cuda")
    mb_ids = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
    mb_attn = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
    mb_cmask = torch.zeros((mb_bs, mb_max_len - 1), dtype=torch.float, device=device)
    for j, (full_ids, seq_len, prompt_len) in enumerate(mb_encoded):
        mb_ids[j, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
        mb_attn[j, :seq_len] = 1
        if seq_len - 1 > prompt_len - 1:
            mb_cmask[j, prompt_len - 1:seq_len - 1] = 1.0

    mb_policy_logps = get_per_token_logps(model, mb_ids, mb_attn)
    mb_old_logps = mb_policy_logps.detach()
    with torch.no_grad():
        mb_ref_logps = get_per_token_logps(model, mb_ids, mb_attn)
    loss_dict = compute_grpo_loss(mb_policy_logps, mb_old_logps, mb_adv,
        ref_logps=mb_ref_logps, beta=0.04, epsilon=0.2, mask=mb_cmask)
    n_tokens = mb_cmask.sum()
    scale = n_tokens / total_masked_tokens if total_masked_tokens > 0 else 1.0 / (B_total // micro_bs)
    (loss_dict["loss"] * scale).backward()
    del mb_policy_logps, mb_old_logps, mb_ref_logps, loss_dict

grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"  grad_norm={grad_norm.item():.4f}")

# Check for NaN grads
has_nan = False
for n, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"  NaN grad in {n}")
        has_nan = True
if has_nan:
    print("  WARNING: NaN gradients detected!")

optimizer.step()
model.eval()
torch.cuda.empty_cache()
gc.collect()
print("  Training step complete")

# --- Post-training generation test ---
print("\n--- Post-training generation test (same sample) ---")
try:
    with torch.no_grad():
        result = run_interleaved(model, processor,
            audio_path=test_sample["audio_path"], question=test_sample["question"],
            choices=test_sample["choices"], gold_answer=test_sample["answer"],
            max_rounds=2, max_new_tokens_per_round=64, temperature=0.9,
            on_duplicate_seg="stop", finalize_on_stop=True, finalize_max_new_tokens=32)
    print(f"  POST-TRAIN: OK, pred={result.get('pred_answer','?')[:60]}")
except Exception as e:
    print(f"  POST-TRAIN: FAIL - {e}")
    sys.exit(1)

# Also test AudioSet_53 (previously bad)
print("\n--- Post-training: AudioSet_53 (previously bad) ---")
with open("dataJson/NAQA/EAQA_RL.jsonl") as f:
    for line in f:
        s = json.loads(line.strip())
        if "multi_choice" in s and "choices" not in s:
            s["choices"] = s["multi_choice"]
        if s["id"] == "AudioSet_53":
            bad_sample = s
            break
try:
    with torch.no_grad():
        result = run_interleaved(model, processor,
            audio_path=bad_sample["audio_path"], question=bad_sample["question"],
            choices=bad_sample["choices"], gold_answer=bad_sample["answer"],
            max_rounds=2, max_new_tokens_per_round=64, temperature=0.9,
            on_duplicate_seg="stop", finalize_on_stop=True, finalize_max_new_tokens=32)
    print(f"  AudioSet_53 POST-TRAIN: OK, pred={result.get('pred_answer','?')[:60]}")
except Exception as e:
    print(f"  AudioSet_53 POST-TRAIN: FAIL - {e}")

print("\n=== Done ===")

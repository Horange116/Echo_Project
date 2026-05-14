# VERL + vLLM Migration Design

## Overview

The official Echo paper implementation already uses **VERL** (GRPO training loop) + **vLLM** (rollout generation) with a custom multi-turn interleaved controller. This document explains how the pieces fit together and how to migrate from our current fcc5fdf isolated-rollout code without breaking the stable path.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  VERL trainer (verl.trainer.main_ppo)                │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │ Actor (FSDP)          │  │ Rollout (vLLM)        │ │
│  │ - GRPO loss           │  │ - generate_sequences() │ │
│  │ - KL penalty          │  │ - multi-turn loop      │ │
│  │ - optimizer            │  │ - seg detect + crop   │ │
│  └──────────────────────┘  └──────────────────────┘ │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │ Ref model (FSDP)      │  │ Reward model          │ │
│  │ - frozen, KL baseline │  │ - Rformat + Rconsist  │ │
│  │                        │  │ + Racc + Rseg         │ │
│  └──────────────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## 1. VERL's Role — GRPO Training Loop

VERL provides the production-grade training infrastructure:

- **FSDP**: Shards model across 8 GPUs for memory efficiency
- **GRPO**: Group-relative advantage normalisation, clipped surrogate loss
- **KL penalty**: Low-variance KL estimator (`kl_loss_type=low_var_kl`)
- **Checkpointing**: Automatic save/load with `save_freq`
- **Wandb logging**: Built-in experiment tracking

Our current `scripts/rl/rollout_smoke_test.py` replicates a simplified version of this loop on a single GPU. Migrating to VERL means we get multi-GPU FSDP, proper checkpoint management, and a battle-tested training loop for free.

### Key VERL config (from `script/stage2_multiturn_rl.sh`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `algorithm.adv_estimator` | `grpo` | Group-normalised advantages |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.04` | KL penalty weight |
| `actor_rollout_ref.rollout.name` | `vllm` | vLLM rollout engine |
| `actor_rollout_ref.rollout.temperature` | `1.0` | Rollout temperature |
| `actor_rollout_ref.rollout.n` | `8` | Rollouts per prompt |
| `actor_rollout_ref.model.freeze_audio_encoder` | `False` | Train audio encoder |
| `reward_model.reward_kwargs.id` | `multiturn_rl_6` | Custom reward plugin |

## 2. vLLM's Role — Rollout Generation

vLLM serves as the rollout engine, replacing our HF `model.generate()`:

- **File**: `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- **Class**: `vLLMRollout(BaseRollout)`
- **Key method**: `generate_sequences(prompts: DataProto) → DataProto`

### How it works:

1. VERL calls `vLLMRollout.generate_sequences()` with batched prompts
2. vLLM engine generates responses with `stop=["</seg>"]` to pause at each segment
3. Python code in `generate_sequences` detects `<seg>start,end</seg>`, crops audio, inserts `<|audio_bos|><|AUDIO|><|audio_eos|>` placeholder tokens
4. Continues generation with updated prompt + audio data (up to 8 turns)
5. Returns final token IDs and masks

vLLM uses a **merged model** (base + LoRA merged via `verl/scripts/model_merger.py`). Weights are synchronised from the FSDP training model to vLLM before each rollout batch.

### vLLM LoRA support:

vLLM supports LoRA via `enable_lora=True` and `LoRARequest`:
```python
lora_request = LoRARequest("adapter", 1, adapter_path)
output = engine.generate(prompts, sampling_params, lora_request=lora_request)
```

## 3. Custom Interleaved Controller — `<seg>` Detection and Audio Cropping

The official VERL modifications (relative to upstream) are concentrated in `generate_sequences()`:

1. **Stop on `<seg>`**: `SamplingParams(stop=["</seg>"])` halts generation at segment boundaries
2. **Parse timestamps**: `re.finditer(r'<seg>([\d\.]+,\s*[\d\.]+)</seg>', text)` extracts time ranges
3. **Crop audio**: `original_audio[start_sample:end_sample]` extracts the segment
4. **Insert audio tokens**: `processed_text = text[:match_end] + "<|audio_bos|><|AUDIO|><|audio_eos|>" + text[match_end:]`
5. **Update multi_modal_data**: Append cropped audio to `multi_modal_data["audio"]`
6. **Continue generation**: Call `engine.generate()` again with updated prompt + audios
7. **Max 8 turns**: `for turn_time in range(max_turns)` prevents infinite loops

Key difference from our current HF approach: vLLM uses **audio placeholder tokens** inserted as text, not actual audio feature tensors. The `Qwen2_5OmniProcessor` handles tokenising these placeholders into the correct token IDs (151645-151648).

## 4. Reward Integration

VERL's reward system uses a plugin architecture. The official Echo code registers reward ID `multiturn_rl_6`.

Our reward function (`echo_rl/rewards.py`) is already aligned with the paper:

```
R(τ) = Rformat(τ) + Rconsist(τ) + Racc(τ) + Rseg(τ)
```

| Component | Max Value | Condition |
|-----------|-----------|-----------|
| Rformat | +0.5 | Output contains `<think>...</think><answer>...</answer>` |
| Rconsist | 0 ~ -0.5 | Penalty proportional to hallucinated wrong-choice tokens in think |
| Racc | +0.5 | Extracted answer matches ground truth (exact match) |
| Rseg | +0.5 | Answer is correct AND at least one `<seg>` was used |

The `rollout_reward()` wrapper in `echo_rl/rollout_rewards.py` produces the flat dict that our current training loop expects. For VERL, the same logic is adapted into VERL's reward plugin format.

## 5. Audio Token Masking in Loss

In the strict interleaved forward pass, audio placeholder tokens (151645-151648) must be excluded from the policy loss. VERL does this at the response mask level:

```python
# verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py, lines 578-580
audio_token_mask = (response == 151646) | (response == 151647) | (response == 151648) | (response == 151645)
response_mask = response_attention_mask.clone()
response_mask = response_mask.masked_fill(audio_token_mask, 0)
```

This is equivalent to our `build_strict_interleaved_input()` in `scripts/rl/rollout_smoke_test.py` (lines 356-469), which builds a `loss_mask` tensor that zeroes out audio token positions.

With VERL, this masking happens automatically — no custom loss mask code needed.

## 6. Text-Only GRPO Fallback

Our `scripts/rl/rollout_smoke_test.py` with `--grpo_forward_mode text_only` remains as a lightweight alternative:

- **Use case**: Quick experiments, debugging reward functions, testing on login node
- **How it works**: Rollout workers generate full interleaved text; training uses text-only token IDs with completion mask → thinker forward → per-token logprobs
- **Advantage**: No audio tokens to mask; simpler loss computation; works on any GPU
- **Limitation**: No gradient signal through audio encoder; ~15% gradient gap vs strict mode

This fallback is preserved indefinitely. It's the fastest path from idea to result.

## 7. Migration Path

### Step 1: SFT (Stage 1) — already done
- Train v9b-diverse-cot-2epoch (or newer SFT checkpoint)
- Model learns to output `<seg>` tags and `<answer>` format

### Step 2: Merge checkpoint — `verl/scripts/model_merger.py`
```bash
python verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/sft_checkpoint \
    --target_dir /path/to/merged_model
```
Note: our current checkpoints are HF PEFT format (not FSDP). Need a separate merge step:
```bash
# Simple merge: load base + LoRA, save merged
python -c "
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration
base = Qwen2_5OmniForConditionalGeneration.from_pretrained('base_model')
model = PeftModel.from_pretrained(base, 'adapter_path')
model = model.merge_and_unload()
model.save_pretrained('merged_model')
"
```

### Step 3: Convert data to Parquet — VERL format
Convert `dataJson/NAQA/EAQA_RL.jsonl` to Parquet with train/val splits following [VERL data requirements](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).

### Step 4: Launch VERL training — `script/stage2_multiturn_rl.sh`
- Point `actor_rollout_ref.model.path` to merged SFT checkpoint
- Point `data.train_files` / `data.val_files` to Parquet datasets
- Run on 8× A800 GPUs

### Step 5: Inference — `inference/inference_multiturn.py`
- Load VERL checkpoint (merged from FSDP)
- Run multi-turn inference on MMAR/MMAU benchmarks

## Mapping: Our Code → VERL

| Our Component | VERL Equivalent | Notes |
|---------------|----------------|-------|
| `scripts/rl/rollout_smoke_test.py` | `verl.trainer.main_ppo` | Training orchestrator |
| `compute_advantages()` in `grpo_utils.py` | `algorithm.adv_estimator=grpo` | Group-normalised advantages |
| `compute_grpo_loss()` in `grpo_utils.py` | VERL's PPO clip + KL penalty | Same algorithm |
| `isolated_rollout_worker.py` | `vLLMRollout` with SPMD | Subprocess → vLLM engine |
| `build_strict_interleaved_input()` | `generate_sequences()` audio mask | Audio tokens excluded from loss |
| `run_interleaved()` in `interleaved_infer.py` | `generate_sequences()` multi-turn | seg→crop→continue loop |
| `load_training_models()` in rollout_smoke_test.py | FSDP actor + ref model | Multi-GPU sharded |
| `echo_rl/rewards.py` → `total_reward()` | `reward_model.reward_kwargs.id` plugin | Same R formula |
| `scripts/rl_backends/rollout_backend_vllm.py` | `vLLMRollout` (native VERL) | Our abstraction → native VERL class |

## Verification Checklist

- [ ] Phase 1 smoke test passes (all 6 vLLM tests)
- [ ] Phase 2 HF backend produces identical results to fcc5fdf `run_interleaved()`
- [ ] Phase 2 vLLM backend produces comparable results to HF backend
- [ ] Parquet dataset prepared for VERL
- [ ] Merged model loads in vLLM
- [ ] Single-GPU VERL dry run (1 step, 1 batch) succeeds
- [ ] 8-GPU VERL training runs without CUDA errors
- [ ] Reward values match between our loop and VERL
- [ ] Checkpoints can be merged and loaded for inference

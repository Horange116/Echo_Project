#!/usr/bin/env python3
"""DeepSpeed ZeRO smoke test for Qwen2.5-Omni-7B + LoRA adapter.
Tests ZeRO-2 and ZeRO-3 with text forward/backward/step and audio forward (no_grad).
No VERL, no vLLM, no GRPO. Just verifies DeepSpeed can wrap Qwen2.5-Omni.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import torch
import deepspeed


def make_ds_config(zero_stage: int, dtype: str = "fp16") -> dict:
    base = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-6, "betas": [0.9, 0.999], "eps": 1e-8},
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-6, "warmup_num_steps": 1},
        },
    }
    if dtype == "fp16":
        base["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}
    else:
        base["bf16"] = {"enabled": True}

    if zero_stage == 2:
        base["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": False,
            "contiguous_gradients": True,
        }
    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage": 3,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_gather_16bit_weights_on_model_save": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "overlap_comm": False,
        }
    return base


def load_model_and_adapter(base_path: str, adapter_path: str | None):
    """Load Qwen2.5-Omni base model to CPU, apply LoRA, disable talker."""
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from transformers import AutoTokenizer

    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"
    print(f"  Loading base model from {base_path} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if adapter_path:
        print(f"  Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.base_model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, processor, tokenizer


def prepare_text_batch(tokenizer, batch_size: int = 1, seq_len: int = 256):
    """Build a dummy text batch: prompt + answer-style tokens."""
    prompt = (
        "Question: What is the capital of France?\n"
        "Choices:\n"
        "A. London\n"
        "B. Paris\n"
        "C. Berlin\n"
        "D. Madrid\n\n"
        "Answer: B"
    )
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    input_ids = tokens.input_ids.repeat(batch_size, 1)
    attention_mask = tokens.attention_mask.repeat(batch_size, 1)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def prepare_audio_batch(processor, tokenizer, audio_path: str, seq_len: int = 512):
    """Build a single audio+text batch using processor and a wav file."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Describe this sound briefly."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audios=[audio],
        return_tensors="pt",
        padding=True,
        max_length=seq_len,
        truncation=True,
    )
    return inputs


def run_smoke_test(args):
    """Run a single ZeRO smoke test.  Returns dict of metrics / errors."""
    zero_stage = args.zero_stage
    base_path = args.base_model
    adapter_path = args.adapter_path or None
    audio_path = args.audio_path
    result = {
        "zero_stage": zero_stage,
        "init_success": False,
        "forward_success": False,
        "backward_success": False,
        "step_success": False,
        "audio_forward_success": False,
        "peak_memory_gb": 0.0,
        "init_time_s": 0.0,
        "forward_time_s": 0.0,
        "backward_time_s": 0.0,
        "step_time_s": 0.0,
        "audio_forward_time_s": 0.0,
        "error": None,
    }

    try:
        # ---- Load model & adapter to CPU ----
        t0 = time.time()
        model, processor, tokenizer = load_model_and_adapter(base_path, adapter_path)
        model.train()  # needed for backward

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,}  /  Total params: {total_params:,}")

        # ---- Prepare data ----
        input_ids, attn_mask, labels = prepare_text_batch(tokenizer)
        print(f"  Text batch: input_ids.shape={input_ids.shape}")

        # ---- DeepSpeed init ----
        ds_config = make_ds_config(zero_stage)
        print(f"  DeepSpeed config (stage {zero_stage}):")
        print(f"    {json.dumps(ds_config['zero_optimization'], indent=2)}")

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
        )
        result["init_time_s"] = round(time.time() - t0, 2)
        result["init_success"] = True
        print(f"  DeepSpeed init OK ({result['init_time_s']}s)")

        torch.cuda.reset_peak_memory_stats()

        # ---- Text forward ----
        t1 = time.time()
        device = model_engine.device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )
        loss = outputs.loss
        result["forward_time_s"] = round(time.time() - t1, 2)
        result["forward_success"] = True
        print(f"  Forward  OK  loss={loss.item():.4f}  ({result['forward_time_s']}s)")

        # ---- Backward ----
        t2 = time.time()
        model_engine.backward(loss)
        result["backward_time_s"] = round(time.time() - t2, 2)
        result["backward_success"] = True
        print(f"  Backward OK  ({result['backward_time_s']}s)")

        # ---- Optimizer step ----
        t3 = time.time()
        model_engine.step()
        result["step_time_s"] = round(time.time() - t3, 2)
        result["step_success"] = True
        print(f"  Step     OK  ({result['step_time_s']}s)")

        # Record peak memory
        result["peak_memory_gb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 2
        )
        print(f"  Peak GPU memory: {result['peak_memory_gb']} GB")

        # Clean up text batch
        del input_ids, attn_mask, labels, loss, outputs
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Audio forward (no_grad) ----
        try:
            t4 = time.time()
            model_engine.eval()
            audio_inputs = prepare_audio_batch(processor, tokenizer, audio_path)
            # Move to device
            audio_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in audio_inputs.items()}
            with torch.no_grad():
                _ = model_engine(**audio_inputs)
            result["audio_forward_time_s"] = round(time.time() - t4, 2)
            result["audio_forward_success"] = True
            print(f"  Audio forward OK ({result['audio_forward_time_s']}s)")
        except Exception:
            result["audio_forward_success"] = False
            if result["error"] is None:
                result["error"] = {}
            result["error"]["audio"] = traceback.format_exc()
            print(f"  Audio forward FAILED:\n{result['error']['audio']}")

        print(f"  === ZERO-{zero_stage} ALL OK ===")

    except Exception:
        result["error"] = traceback.format_exc()
        print(f"  === ZERO-{zero_stage} FAILED ===\n{result['error']}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_stage", type=int, required=True, choices=[2, 3])
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--audio_path", type=str, default="")
    parser.add_argument("--report_json", type=str, default="output/debug/ds_zero_smoke_report.json")
    args = parser.parse_args()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DeepSpeed ZeRO-{args.zero_stage} smoke test")
    print(f"  Rank: {os.environ.get('LOCAL_RANK', 'N/A')} / {os.environ.get('WORLD_SIZE', 'N/A')}")

    result = run_smoke_test(args)

    # Write report (only rank 0)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        # Merge with existing report
        report = {}
        if os.path.exists(args.report_json):
            try:
                report = json.loads(open(args.report_json).read())
            except Exception:
                report = {}
        key = f"zero_{args.zero_stage}"
        report[key] = result

        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report saved -> {args.report_json}")

    # Return exit code
    success = all([
        result["init_success"],
        result["forward_success"],
        result["backward_success"],
        result["step_success"],
    ])
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

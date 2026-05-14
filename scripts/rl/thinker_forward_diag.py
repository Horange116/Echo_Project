#!/usr/bin/env python3
"""
Minimal thinker forward diagnostic: base model vs LoRA policy vs LoRA ref.
Loads each model, runs one text-only thinker forward, checks logits and weights for nan.
"""
import argparse
import gc
import torch
import torch.nn.functional as F


def diag_logits(tag: str, logits: torch.Tensor) -> None:
    print(f"  [{tag}] logits: shape={tuple(logits.shape)} "
          f"min={logits.min().item():.6g} max={logits.max().item():.6g} "
          f"mean={logits.mean().item():.6g} "
          f"nan={bool(logits.isnan().any())} inf={bool(logits.isinf().any())}")


def diag_weights(tag: str, model: torch.nn.Module) -> None:
    """Scan all parameters for nan/inf."""
    total = n_nan = n_inf = 0
    first_nan_name = None
    first_inf_name = None
    for name, p in model.named_parameters():
        if p.numel() == 0:
            continue
        total += 1
        if p.isnan().any():
            n_nan += 1
            if first_nan_name is None:
                first_nan_name = name
        if p.isinf().any():
            n_inf += 1
            if first_inf_name is None:
                first_inf_name = name
    print(f"  [{tag}] weight scan: {total} params, {n_nan} with nan "
          f"{f'(first: {first_nan_name})' if first_nan_name else ''}, "
          f"{n_inf} with inf "
          f"{f'(first: {first_inf_name})' if first_inf_name else ''}")


def make_mini_input(tokenizer, device: torch.device) -> dict:
    """Create a tiny text-only input for thinker forward."""
    text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_thinker(model, kwargs: dict, tag: str) -> None:
    """Run thinker forward and print logits diagnostics."""
    if hasattr(model, "get_base_model"):
        thinker = model.get_base_model().thinker
    else:
        thinker = model.thinker
    print(f"\n  [{tag}] thinker forward ...")
    with torch.no_grad():
        outputs = thinker(**kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    diag_logits(tag, logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Model path: {args.model_path}")
    print(f"Adapter path: {args.adapter_path}")
    print()

    from transformers import Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    # Create a single mini input (same for all models)
    inp = make_mini_input(tokenizer, device)
    print(f"  input_ids: shape={tuple(inp['input_ids'].shape)}, "
          f"num_tokens={inp['input_ids'].shape[1]}")
    print()

    # ═══ Test A: Base model (no LoRA) ═══
    print("=" * 60)
    print("  A: Base model (no LoRA)")
    print("=" * 60)
    from transformers import Qwen2_5OmniForConditionalGeneration

    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu",
    )
    print(f"  class: {type(base).__name__}")
    print(f"  dtype: {base.dtype}")
    base = base.to(device, dtype=torch.float16)
    base.eval()
    diag_weights("base_weights", base)
    run_thinker(base, inp, "base_thinker")
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print()

    # ═══ Test B: Policy model (base + LoRA, is_trainable=True) ═══
    print("=" * 60)
    print("  B: Policy model (base + LoRA, is_trainable=True)")
    print("=" * 60)
    if args.adapter_path:
        from peft import PeftModel

        base2 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="cpu",
        )
        policy = PeftModel.from_pretrained(base2, args.adapter_path, is_trainable=True)
        policy.base_model.disable_talker()
        policy = policy.to(device, dtype=torch.float16)
        for n, p in policy.named_parameters():
            p.requires_grad_("lora" in n)
        policy.eval()
        print(f"  class: {type(policy).__name__}")
        print(f"  dtype: {policy.dtype}")
        diag_weights("policy_weights", policy)
        run_thinker(policy, inp, "policy_thinker")
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("  Skipped (no adapter_path)")
    print()

    # ═══ Test C: Ref model (base + LoRA, is_trainable=False) ═══
    print("=" * 60)
    print("  C: Ref model (base + LoRA, is_trainable=False)")
    print("=" * 60)
    if args.adapter_path:
        base3 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="cpu",
        )
        ref = PeftModel.from_pretrained(base3, args.adapter_path)
        ref.base_model.disable_talker()
        ref = ref.to(device, dtype=torch.float16)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(f"  class: {type(ref).__name__}")
        print(f"  dtype: {ref.dtype}")
        diag_weights("ref_weights", ref)
        run_thinker(ref, inp, "ref_thinker")
        del ref
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("  Skipped (no adapter_path)")
    print()

    print("=" * 60)
    print("  Diagnosis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

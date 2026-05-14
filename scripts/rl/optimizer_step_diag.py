#!/usr/bin/env python3
"""
Minimal optimizer step diagnosis: isolate why optimizer.step() corrupts LoRA weights.

Loads clean step_10 checkpoint, does one forward/backward with tiny dummy input,
then tests multiple optimizer configurations to find which factor causes NaN.

Usage:
    python scripts/rl/optimizer_step_diag.py \
        --model_path /path/to/Qwen2.5-Omni-7B \
        --adapter_path /path/to/step_10 \
        --gpu_id 0
"""
import argparse
import gc
import time

import torch
import torch.nn.functional as F


def scan_params(model, tag=""):
    """Scan trainable LoRA params for nan/inf. Returns (clean, first_bad_name)."""
    nan_count = inf_count = 0
    first_nan = first_inf = ""
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or "lora" not in name.lower():
            continue
        total += 1
        if p.isnan().any():
            nan_count += 1
            if not first_nan:
                first_nan = name
        if p.isinf().any():
            inf_count += 1
            if not first_inf:
                first_inf = name
    return {
        "tag": tag,
        "clean": nan_count == 0 and inf_count == 0,
        "total_lora": total,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "first_nan": first_nan,
        "first_inf": first_inf,
    }


def print_scan(s, prefix=""):
    """Pretty-print a scan result."""
    if s["clean"]:
        print(f"  {prefix}CLEAN: {s['total_lora']} LoRA params, 0 nan, 0 inf")
    else:
        print(f"  {prefix}CORRUPTED: {s['total_lora']} LoRA params, "
              f"{s['nan_count']} nan, {s['inf_count']} inf")
        if s["first_nan"]:
            print(f"    first_nan={s['first_nan']}")
        if s["first_inf"]:
            print(f"    first_inf={s['first_inf']}")


def print_grad_stats(model, tag=""):
    """Print grad stats of first trainable LoRA param with non-None grad."""
    for name, p in model.named_parameters():
        if not p.requires_grad or "lora" not in name.lower():
            continue
        if p.grad is None:
            continue
        g = p.grad
        print(f"  [grad {tag}] {name}: "
              f"dtype={g.dtype} "
              f"min={g.min().item():.8g} "
              f"max={g.max().item():.8g} "
              f"mean={g.mean().item():.8g} "
              f"nan={bool(g.isnan().any())} "
              f"inf={bool(g.isinf().any())} "
              f"has_finite={torch.isfinite(g).all().item()}")
        return
    print(f"  [grad {tag}] No LoRA params with non-None grad")


def print_param_stats(model, tag=""):
    """Print param stats of first trainable LoRA param."""
    for name, p in model.named_parameters():
        if not p.requires_grad or "lora" not in name.lower():
            continue
        print(f"  [param {tag}] {name}: "
              f"dtype={p.dtype} "
              f"min={p.min().item():.8g} "
              f"max={p.max().item():.8g} "
              f"mean={p.mean().item():.8g} "
              f"nan={bool(p.isnan().any())} "
              f"inf={bool(p.isinf().any())}")
        return


def inspect_optimizer_state(optimizer):
    """Print optimizer state dtype info."""
    for i, group in enumerate(optimizer.param_groups):
        print(f"  [opt] param_group {i}: lr={group['lr']}, "
              f"betas={group.get('betas')}, "
              f"eps={group.get('eps')}, "
              f"weight_decay={group.get('weight_decay')}")
        for p in group["params"]:
            if p.requires_grad and p.grad is not None:
                print(f"  [opt]   param dtype={p.dtype}, grad dtype={p.grad.dtype}")
                state = optimizer.state[p]
                if state:
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  [opt]   state['{k}'] dtype={v.dtype} "
                                  f"shape={tuple(v.shape)} "
                                  f"min={v.min().item():.8g} "
                                  f"max={v.max().item():.8g} "
                                  f"nan={bool(v.isnan().any())}")
                break
        break  # only first group, first param


def make_dummy_input(tokenizer, device):
    """Minimal text-only input for thinker forward."""
    text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(text, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def run_one_step(model, inp, optimizer):
    """One forward + backward + optimizer.step() cycle."""
    thinker = model.get_base_model().thinker
    outputs = thinker(**inp)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    log_probs = logits.log_softmax(dim=-1)
    # Use next-token prediction as loss (like language modeling)
    shift_logps = log_probs[:, :-1].gather(
        dim=-1, index=inp["input_ids"][:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    loss = -shift_logps.mean()  # negative log-likelihood
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test_config(model, inp, lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0.01, config_name=""):
    """Test one optimizer configuration. Returns (pre_scan, post_scan)."""
    print(f"\n  ── Config: {config_name} ──")
    print(f"     AdamW(lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})")

    # Snapshot fresh model weights
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, eps=eps,
        weight_decay=weight_decay,
    )

    # Check model state before step
    pre_scan = scan_params(model, f"pre_step/{config_name}")
    print_scan(pre_scan, "pre_step: ")
    print_param_stats(model, f"pre_step/{config_name}")

    # Inspect optimizer before step (state will be initialized on first step)
    print(f"  [opt] dtype inspection (state not yet initialized):")
    for n, p in model.named_parameters():
        if p.requires_grad and "lora" in n.lower() and p.grad is not None:
            print(f"    param '{n}': param.dtype={p.dtype}, grad.dtype={p.grad.dtype}")
            break

    # One forward + backward + step
    loss_val = run_one_step(model, inp, optimizer)
    print(f"  loss={loss_val:.8g}")

    # Check optimizer state AFTER step
    inspect_optimizer_state(optimizer)

    # Check model state after step
    post_scan = scan_params(model, f"post_step/{config_name}")
    print_scan(post_scan, "post_step: ")
    print_param_stats(model, f"post_step/{config_name}")

    # Restore model parameters for next test
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(state_dict[k])
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return pre_scan, post_scan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print()

    # ═══ 1. Load model (same as training) ═══
    print("=" * 70)
    print("  Loading model (same as load_training_models)")
    print("=" * 70)
    t0 = time.time()
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    model.base_model.disable_talker()
    model = model.to(device, dtype=torch.float16)
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    # ═══ 2. Verify model is clean ═══
    print("=" * 70)
    print("  Initial model state")
    print("=" * 70)
    initial = scan_params(model, "initial")
    print_scan(initial)
    print_param_stats(model, "initial")
    print()

    # ═══ 3. One forward/backward to get gradients ═══
    print("=" * 70)
    print("  Single forward/backward (dummy input)")
    print("=" * 70)
    inp = make_dummy_input(tokenizer, device)
    print(f"  input_ids: {tuple(inp['input_ids'].shape)}, "
          f"num_tokens={inp['input_ids'].shape[1]}")

    thinker = model.get_base_model().thinker
    outputs = thinker(**inp)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    log_probs = logits.log_softmax(dim=-1)
    shift_logps = log_probs[:, :-1].gather(
        dim=-1, index=inp["input_ids"][:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    loss = -shift_logps.mean()
    print(f"  logits: shape={tuple(logits.shape)} "
          f"min={logits.min().item():.6g} max={logits.max().item():.6g} "
          f"nan={bool(logits.isnan().any())}")
    print(f"  loss={loss.item():.8g} nan={bool(loss.isnan())}")

    # Backward
    model.zero_grad()
    loss.backward()
    print()

    # ═══ 4. Check gradients ═══
    print("=" * 70)
    print("  Gradient state after backward")
    print("=" * 70)
    print_grad_stats(model, "post_backward")
    scan_after_bw = scan_params(model, "after_backward")
    print_scan(scan_after_bw, "")
    print()

    # Print LoRA parameter dtype info
    print("=" * 70)
    print("  LoRA parameter dtype info")
    print("=" * 70)
    for name, p in model.named_parameters():
        if p.requires_grad and "lora" in name.lower():
            print(f"  {name}: param.dtype={p.dtype}, "
                  f"grad.dtype={p.grad.dtype if p.grad is not None else 'None'}")
            break
    print()

    # ═══ 5. Test optimizer configurations ═══
    print("=" * 70)
    print("  Optimizer config comparison")
    print("=" * 70)
    print()
    print(f"  {'Config':<35} {'pre_step':<10} {'post_step':<12} {'first_bad'}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*30}")

    # Save clean model state
    clean_state = {k: v.clone() for k, v in model.state_dict().items()}

    configs = [
        # (lr, betas, eps, weight_decay, name)
        (1e-6, (0.9, 0.999), 1e-8, 0.01,
         "A) current: AdamW(lr=1e-6, wd=0.01, eps=1e-8)"),
        (1e-6, (0.9, 0.999), 1e-8, 0.00,
         "B) wd=0"),
        (1e-6, (0.9, 0.999), 1e-4, 0.01,
         "C) eps=1e-4"),
        (1e-6, (0.9, 0.999), 1e-4, 0.00,
         "D) eps=1e-4, wd=0"),
        (1e-6, (0.9, 0.999), 1e-3, 0.01,
         "E) eps=1e-3"),
        (1e-6, (0.9, 0.999), 1e-8, 0.01,
         "F) repeat A (fresh optim)"),
        (1e-7, (0.9, 0.999), 1e-8, 0.01,
         "G) lr=1e-7"),
    ]

    results = []
    for lr, betas, eps, wd, name in configs:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(clean_state[k])

        pre, post = test_config(model, inp, lr=lr, betas=betas, eps=eps,
                                weight_decay=wd, config_name=name)
        results.append((name, pre, post))

        first_bad = post.get("first_nan") or post.get("first_inf") or "-"
        pre_clean = "CLEAN" if pre["clean"] else "BROKEN"
        post_clean = "CLEAN" if post["clean"] else "BROKEN"
        print(f"  {name:<35} {pre_clean:<10} {post_clean:<12} {first_bad}")

    print()

    # ═══ 6. Summary table ═══
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Config':<35} {'pre_step':<10} {'post_step':<12} {'first_bad'}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*30}")
    for name, pre, post in results:
        first_bad = post.get("first_nan") or post.get("first_inf") or "-"
        pre_clean = "CLEAN" if pre["clean"] else "BROKEN"
        post_clean = "CLEAN" if post["clean"] else "BROKEN"
        short_name = name.split(")")[0] + ")"
        print(f"  {short_name:<35} {pre_clean:<10} {post_clean:<12} {first_bad}")
    print()

    print("=" * 70)
    print("  Diagnosis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

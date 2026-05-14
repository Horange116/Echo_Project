#!/usr/bin/env python3
"""
Scan LoRA checkpoint directories for nan/inf in adapter weights.
Reports which checkpoint first introduced corruption.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def scan_safetensors(path: Path) -> dict:
    """Load a .safetensors file and scan all tensors for nan/inf."""
    tensors = load_file(str(path))
    result = {
        "path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "total_tensors": len(tensors),
        "nan_tensors": 0,
        "inf_tensors": 0,
        "nan_params": 0,
        "inf_params": 0,
        "total_params": 0,
        "first_nan_key": None,
        "first_inf_key": None,
        "clean": True,
    }
    for key, t in tensors.items():
        n = t.numel()
        result["total_params"] += n
        if t.isnan().any():
            result["nan_tensors"] += 1
            result["nan_params"] += t.isnan().sum().item()
            if result["first_nan_key"] is None:
                result["first_nan_key"] = key
            result["clean"] = False
        if t.isinf().any():
            result["inf_tensors"] += 1
            result["inf_params"] += t.isinf().sum().item()
            if result["first_inf_key"] is None:
                result["first_inf_key"] = key
            result["clean"] = False
    return result


def scan_checkpoint_dir(ckpt_dir: Path) -> dict:
    """Scan a single checkpoint directory for nan/inf in adapter weights."""
    result = {
        "checkpoint": ckpt_dir.name,
        "path": str(ckpt_dir),
        "has_adapter_config": (ckpt_dir / "adapter_config.json").exists(),
        "safetensors_files": [],
        "overall_clean": True,
        "total_params": 0,
        "total_nan_params": 0,
        "total_inf_params": 0,
        "errors": [],
    }

    # Load adapter_config to see LoRA rank etc.
    config_path = ckpt_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        result["lora_rank"] = cfg.get("r")
        result["lora_alpha"] = cfg.get("lora_alpha")

    # Scan all .safetensors files
    for fpath in sorted(ckpt_dir.glob("*.safetensors")):
        try:
            sr = scan_safetensors(fpath)
            result["safetensors_files"].append(sr)
            result["total_params"] += sr["total_params"]
            result["total_nan_params"] += sr["nan_params"]
            result["total_inf_params"] += sr["inf_params"]
            if not sr["clean"]:
                result["overall_clean"] = False
        except Exception as e:
            result["errors"].append(f"{fpath}: {e}")

    return result


def print_report(results: list[dict]) -> None:
    print()
    print("=" * 80)
    print("  LoRA Checkpoint Nan/Inf Scan Report")
    print("=" * 80)
    print()

    first_bad = None
    all_clean = True

    for r in results:
        tag = "  CLEAN  " if r["overall_clean"] else "  ** CORRUPTED **"
        print(f"  [{tag}] {r['checkpoint']}")
        print(f"         path: {r['path']}")
        if r.get("lora_rank"):
            print(f"         LoRA rank={r['lora_rank']}, alpha={r['lora_alpha']}")
        print(f"         total_params: {r['total_params']:,}")
        print(f"         nan params:   {r['total_nan_params']:,}")
        print(f"         inf params:   {r['total_inf_params']:,}")
        for sr in r["safetensors_files"]:
            status = "clean" if sr["clean"] else "NAN/INF"
            print(f"         file: {Path(sr['path']).name} "
                  f"({sr['file_size_mb']:.1f} MB, {sr['total_tensors']} tensors) [{status}]")
            if sr["first_nan_key"]:
                print(f"           first nan key: {sr['first_nan_key']}")
            if sr["first_inf_key"]:
                print(f"           first inf key: {sr['first_inf_key']}")
        if r["errors"]:
            for e in r["errors"]:
                print(f"         ERROR: {e}")
        print()

        if not r["overall_clean"] and first_bad is None:
            first_bad = r["checkpoint"]
        if not r["overall_clean"]:
            all_clean = False

    print("=" * 80)
    if all_clean:
        print("  RESULT: All checkpoints clean. No corruption found.")
    else:
        print(f"  RESULT: Corruption first detected in checkpoint: [{first_bad}]")
        print()
        print("  Chain of corruption:")
        for r in results:
            arrow = " --> " if not r["overall_clean"] else "     "
            tag = "CORRUPTED" if not r["overall_clean"] else "clean"
            print(f"    {arrow} {r['checkpoint']}: {tag} "
                  f"(nan={r['total_nan_params']:,}, inf={r['total_inf_params']:,})")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Scan LoRA checkpoint directories for nan/inf weights."
    )
    parser.add_argument(
        "checkpoint_dirs",
        nargs="+",
        help="One or more checkpoint directories to scan",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report instead of human-readable",
    )
    args = parser.parse_args()

    results = []
    for d in args.checkpoint_dirs:
        ckpt_dir = Path(d)
        if not ckpt_dir.is_dir():
            print(f"WARNING: not a directory, skipping: {d}", file=sys.stderr)
            continue
        r = scan_checkpoint_dir(ckpt_dir)
        results.append(r)

    # Sort by checkpoint name for consistent ordering
    results.sort(key=lambda r: r["checkpoint"])

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print_report(results)

    # Exit code: 0 if all clean, 1 if any corruption
    if any(not r["overall_clean"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

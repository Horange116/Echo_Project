#!/usr/bin/env python3
"""
Test30: VERL data format alignment.
Convert custom JSONL → VERL Parquet format and dry-load with RLHFDataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/s2025244189/s2025244265/Projects/Echo_Project",
)
sys.path.insert(0, PROJECT_ROOT)


def jsonl_to_verl_parquet(
    input_jsonl: str,
    output_parquet: str,
    max_samples: int = 4,
    prompt_max_length: int = 2048,
) -> int:
    """Convert custom JSONL to VERL-compatible Parquet format.

    VERL ``RLHFDataset`` expects a Parquet file with columns:
      - ``prompt``: list of chat messages (role/content dicts)
                   Content uses ``<audio>`` placeholder for audio paths
      - ``audios``: list of audio file paths (one per ``<audio>`` tag)
      - ``answer``: ground-truth answer string

    Our JSONL has: id, audio_path, question, choices, answer, type

    Returns number of samples written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    prompt_col = []
    audios_col = []
    answer_col = []
    extra_info_col = []
    data_source_col = []

    with open(input_jsonl) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            s = json.loads(line.strip())

            choices_str = ", ".join(s["choices"])
            prompt_text = (
                f"{s['question']} Choose the answer from {choices_str}. "
                f"Think step-by-step. Refer to the specific audio segments while thinking, "
                f"and indicate the corresponding timestamps with <seg>start, end</seg>. "
                f"Answer in the format of <think>...</think><answer>...</answer>."
            )
            messages = [
                {"role": "user", "content": f"<audio>{prompt_text}"}
            ]

            extra_info = {
                "id": s["id"],
                "type": s.get("type", "unknown"),
                "original_index": s.get("original_index", 0),
            }

            prompt_col.append(json.dumps(messages))
            audios_col.append(json.dumps([s["audio_path"]]))
            answer_col.append(s["answer"])
            extra_info_col.append(json.dumps(extra_info))
            data_source_col.append(s.get("type", "unknown"))

    table = pa.table({
        "prompt": pa.array(prompt_col, type=pa.string()),
        "audios": pa.array(audios_col, type=pa.string()),
        "answer": pa.array(answer_col, type=pa.string()),
        "extra_info": pa.array(extra_info_col, type=pa.string()),
        "data_source": pa.array(data_source_col, type=pa.string()),
    })
    pq.write_table(table, output_parquet)
    n = len(table)
    print(f"  Wrote {n} samples → {output_parquet}")
    return n


def dry_load_parquet(parquet_path: str, model_path: str) -> bool:
    """Dry-load Parquet with VERL's RLHFDataset.

    This tests:
    - Parquet format compatibility
    - Chat template application
    - Audio processing (process_mm_info)
    - Tokenization within max_prompt_length

    We run this inside the container where transformers+processor are available.
    """
    from omegaconf import DictConfig
    from transformers import AutoProcessor, AutoTokenizer

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Load tokenizer + processor (same as main_ppo does)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )

    # Minimal data config matching stage2_multiturn_rl.sh
    data_config = DictConfig({
        "prompt_key": "prompt",
        "image_key": "images",
        "video_key": "videos",
        "audio_key": "audios",
        "max_prompt_length": 2048,
        "truncation": "right",
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": 2,
        "return_raw_chat": True,
        "cache_dir": "~/.cache/verl/rlhf",
        "seed": 42,
        "shuffle": False,
    })

    print(f"\n  Loading Parquet: {parquet_path}")
    print(f"  Model: {model_path}")
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  Processor: {processor.__class__.__name__}")
    print()

    dataset = RLHFDataset(
        data_files=parquet_path,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    print(f"\n  Dataset size: {len(dataset)}")

    # Examine first sample
    sample = dataset[0]
    print(f"\n  --- Sample 0 keys ---")
    for k, v in sample.items():
        if isinstance(v, str):
            print(f"    {k}: str(len={len(v)}, first_80={v[:80]!r})")
        elif isinstance(v, list):
            print(f"    {k}: list(len={len(v)})")
        elif isinstance(v, dict):
            print(f"    {k}: dict(keys={list(v.keys())})")
        elif hasattr(v, "shape"):
            print(f"    {k}: tensor(shape={v.shape}, dtype={v.dtype})")
        else:
            print(f"    {k}: {type(v).__name__}({v})")

    # Verify key fields exist
    required_keys = {"input_ids", "attention_mask", "position_ids", "raw_prompt", "multi_modal_data", "multi_modal_inputs", "audios"}
    missing = required_keys - set(sample.keys())
    if missing:
        print(f"\n  ❌ Missing keys: {missing}")
        return False

    # Verify multi_modal_data structure
    mmd = sample.get("multi_modal_data", {})
    if "audio" not in mmd:
        print(f"\n  ❌ multi_modal_data missing 'audio' key")
        return False
    print(f"\n  ✅ multi_modal_data['audio']: {len(mmd['audio'])} segment(s)")
    for i, aud in enumerate(mmd["audio"]):
        print(f"       [{i}] waveform shape={aud[0].shape}, sr={aud[1]}")

    # Verify multi_modal_inputs structure (has input_features, feature_attention_mask)
    mmi = sample.get("multi_modal_inputs", {})
    print(f"  ✅ multi_modal_inputs keys: {list(mmi.keys())}")
    for k, v in mmi.items():
        if hasattr(v, "shape"):
            print(f"       {k}: shape={v.shape}, dtype={v.dtype}")

    # Verify token count
    print(f"\n  ✅ input_ids length: {sample['input_ids'].shape[0]}")
    print(f"  ✅ raw_prompt length: {len(sample['raw_prompt'])} chars")

    print(f"\n  ✅ RLHFDataset loading: PASS")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default=None,
                        help="Custom JSONL path")
    parser.add_argument("--output-parquet", default=None,
                        help="Output Parquet path")
    parser.add_argument("--model-path", required=True,
                        help="Qwen2.5-Omni-7B path")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip conversion, only load existing Parquet")
    args = parser.parse_args()

    print("=" * 60)
    print("Test30: VERL Data Format Alignment")
    print("=" * 60)

    # Default paths
    if args.input_jsonl is None:
        args.input_jsonl = os.path.join(
            PROJECT_ROOT, "output/GeneratedData/eval_manifest_500.jsonl"
        )
    if args.output_parquet is None:
        args.output_parquet = os.path.join(
            PROJECT_ROOT,
            f"output/verl_data_align_test30/test30_eval_manifest_{args.max_samples}.parquet",
        )

    print(f"\n  Input JSONL:  {args.input_jsonl}")
    print(f"  Output Parquet: {args.output_parquet}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Model path: {args.model_path}")

    # Step 1: Convert JSONL → Parquet
    print(f"\n{'='*60}")
    print("Step 1: Convert JSONL → VERL Parquet")
    print(f"{'='*60}")

    if not args.skip_convert:
        if not os.path.exists(args.input_jsonl):
            print(f"  ❌ Input not found: {args.input_jsonl}")
            return 1
        os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
        n = jsonl_to_verl_parquet(
            args.input_jsonl, args.output_parquet, args.max_samples
        )
    else:
        n = 0
        print(f"  Skipping conversion (--skip-convert)")

    # Step 2: Dry-load with RLHFDataset
    print(f"\n{'='*60}")
    print("Step 2: Dry-load with VERL RLHFDataset")
    print(f"{'='*60}")

    if not os.path.exists(args.output_parquet):
        print(f"  ❌ Parquet not found: {args.output_parquet}")
        return 1

    ok = dry_load_parquet(args.output_parquet, args.model_path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Conversion:  {n} samples written")
    print(f"  Dry-load:    {'✅ PASS' if ok else '❌ FAIL'}")
    print(f"  Output:      {args.output_parquet}")
    print()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

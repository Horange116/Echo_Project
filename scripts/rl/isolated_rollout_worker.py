#!/usr/bin/env python3
"""
Isolated rollout worker.

Run interleaved inference in a subprocess so that CUDA device-side asserts
do not contaminate the main training process.

Usage:
    python isolated_rollout_worker.py \
        --sample_json '{"id":"...","audio_path":"...","question":"...","choices":[...],"answer":"..."}' \
        --model_path /path/to/model \
        --adapter_path /path/to/adapter \
        --output /path/to/result.json \
        --max_rounds 2 --max_new_tokens 96 --num_generations 4 \
        --temperature 0.9 --timeout 600
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback

os.environ["QWEN_OMNI_SKIP_SPK"] = "1"

import torch

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.interleaved_infer import run_interleaved, load_model_and_processor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_json", required=True, help="JSON string of sample")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--max_rounds", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--num_generations", type=int, default=4,
                   help="Rollouts per sample")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout in seconds (handled by caller)")
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
    return p.parse_args()


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Worker timed out")


def main() -> None:
    args = parse_args()

    # Redirect all print output to stderr so stdout stays clean for JSON
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    sample = json.loads(args.sample_json)
    if "multi_choice" in sample and "choices" not in sample:
        sample["choices"] = sample["multi_choice"]

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.timeout)

    result = {
        "sample_id": sample.get("id", "?"),
        "rollouts": [],
        "worker_error": None,
        "worker_elapsed_s": 0.0,
    }
    t_start = time.time()

    try:
        # Load model (fresh CUDA context in subprocess)
        model, processor = load_model_and_processor(
            args.model_path, args.adapter_path,
        )
        model.eval()

        for r_idx in range(args.num_generations):
            try:
                with torch.no_grad():
                    rollout = run_interleaved(
                        model, processor,
                        audio_path=sample["audio_path"],
                        question=sample["question"],
                        choices=sample["choices"],
                        gold_answer=sample.get("answer", ""),
                        max_rounds=args.max_rounds,
                        max_new_tokens_per_round=args.max_new_tokens,
                        temperature=args.temperature,
                        on_duplicate_seg="stop",
                        finalize_on_stop=True,
                        finalize_max_new_tokens=args.finalize_max_new_tokens,
                    )
                result["rollouts"].append({
                    "final_response": rollout.get("final_response", ""),
                    "pred_answer": rollout.get("pred_answer", ""),
                    "total_rounds": rollout.get("total_rounds", 0),
                    "used_segments": rollout.get("used_segments", []),
                    "stop_reason": rollout.get("stop_reason", "unknown"),
                    "round_outputs": rollout.get("round_outputs", []),
                    "triggered_interleaved": rollout.get("triggered_interleaved", False),
                    "has_final_answer": rollout.get("has_final_answer", False),
                    "answer_correct": rollout.get("answer_correct", False),
                })
            except Exception as e:
                result["rollouts"].append({
                    "final_response": "",
                    "pred_answer": "",
                    "total_rounds": 0,
                    "used_segments": [],
                    "stop_reason": "error",
                    "round_outputs": [],
                    "triggered_interleaved": False,
                    "has_final_answer": False,
                    "answer_correct": False,
                    "error": str(e)[:200],
                })
                # If it's a CUDA error, stop — don't bother with remaining gens
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    break

    except TimeoutError:
        result["worker_error"] = "timeout"
    except Exception as e:
        result["worker_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["worker_traceback"] = traceback.format_exc()[-2000:]
    finally:
        signal.alarm(0)
        result["worker_elapsed_s"] = time.time() - t_start

    # Write result to stdout for main process to capture
    json.dump(result, _real_stdout, ensure_ascii=False)
    _real_stdout.flush()


if __name__ == "__main__":
    main()

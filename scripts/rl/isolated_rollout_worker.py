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
    p.add_argument("--sample_json", default=None,
                   help="JSON string of sample (one-shot mode)")
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
    p.add_argument("--persistent", action="store_true",
                   help="Run in persistent mode: read tasks from stdin, write results to stdout")
    return p.parse_args()


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Worker timed out")


def run_one_sample(model, processor, sample: dict, args: argparse.Namespace) -> dict:
    """Run rollouts for a single sample. Returns result dict."""
    signal.alarm(args.timeout)
    t_start = time.time()

    result = {
        "sample_id": sample.get("id", "?"),
        "rollouts": [],
        "worker_error": None,
        "worker_elapsed_s": 0.0,
    }

    try:
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
                    "used_segment_paths": rollout.get("used_segment_paths", []),
                    "triggered_interleaved": rollout.get("triggered_interleaved", False),
                    "has_final_answer": rollout.get("has_final_answer", False),
                    "answer_correct": rollout.get("answer_correct", False),
                })
            except Exception as e:
                result["rollouts"].append({
                    "final_response": "", "pred_answer": "",
                    "total_rounds": 0, "used_segments": [],
                    "stop_reason": "error", "round_outputs": [],
                    "used_segment_paths": [],
                    "triggered_interleaved": False,
                    "has_final_answer": False, "answer_correct": False,
                    "error": str(e)[:200],
                })
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

    return result


def main() -> None:
    args = parse_args()

    # Redirect all print output to stderr so stdout stays clean for JSON
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    if args.persistent:
        _main_persistent(args, _real_stdout)
    else:
        _main_one_shot(args, _real_stdout)


def _main_one_shot(args: argparse.Namespace, _real_stdout) -> None:
    sample = json.loads(args.sample_json)
    if "multi_choice" in sample and "choices" not in sample:
        sample["choices"] = sample["multi_choice"]

    signal.signal(signal.SIGALRM, _timeout_handler)

    model, processor = load_model_and_processor(
        args.model_path, args.adapter_path,
    )
    model.eval()

    result = run_one_sample(model, processor, sample, args)

    json.dump(result, _real_stdout, ensure_ascii=False)
    _real_stdout.flush()


def _main_persistent(args: argparse.Namespace, _real_stdout) -> None:
    """Persistent worker: load model once, process tasks from stdin."""
    print("[persistent_worker] Loading model ...", file=sys.stderr)
    model, processor = load_model_and_processor(
        args.model_path, args.adapter_path,
    )
    model.eval()
    print("[persistent_worker] Model loaded, waiting for tasks ...", file=sys.stderr)

    signal.signal(signal.SIGALRM, _timeout_handler)
    task_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError:
            print(f"[persistent_worker] Bad JSON input, skipping", file=sys.stderr)
            continue

        if task.get("action") == "shutdown":
            print(f"[persistent_worker] Shutdown after {task_count} tasks", file=sys.stderr)
            break

        sample = task.get("sample", task)
        if "multi_choice" in sample and "choices" not in sample:
            sample["choices"] = sample["multi_choice"]

        # Override per-task params if provided
        for key in ("num_generations", "max_rounds", "max_new_tokens",
                     "temperature", "finalize_max_new_tokens", "timeout"):
            if key in task:
                setattr(args, key, task[key])

        task_count += 1
        result = run_one_sample(model, processor, sample, args)

        # Write result line to real stdout
        json.dump(result, _real_stdout, ensure_ascii=False)
        _real_stdout.write("\n")
        _real_stdout.flush()


if __name__ == "__main__":
    main()

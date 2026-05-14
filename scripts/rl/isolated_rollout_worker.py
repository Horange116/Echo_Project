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
from typing import Any, Tuple

os.environ["QWEN_OMNI_SKIP_SPK"] = "1"

import torch

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.interleaved_infer import run_interleaved, load_model_and_processor
from scripts.rl_rollout.echo_interleaved_rollout_controller import (
    EchoVLLMBatchedRolloutController,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_json", default=None,
                   help="JSON string of sample (one-shot mode)")
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default="")
    p.add_argument("--rollout_backend", default="hf",
                   choices=["hf", "vllm_batched"])
    p.add_argument("--max_rounds", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--num_generations", type=int, default=4,
                   help="Rollouts per sample")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout in seconds (handled by caller)")
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--work_dir", default=None)
    p.add_argument("--persistent", action="store_true",
                   help="Run in persistent mode: read tasks from stdin, write results to stdout")
    return p.parse_args()


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Worker timed out")


def _load_runtime(args: argparse.Namespace) -> Tuple[Any, Any]:
    if args.rollout_backend == "hf":
        if not args.adapter_path:
            raise ValueError("--adapter_path is required for rollout_backend=hf")
        model, processor = load_model_and_processor(
            args.model_path, args.adapter_path or None,
        )
        # Disable talker for text-only inference (avoids speaker assignment error)
        model.base_model.disable_talker()
        model.eval()
        return model, processor

    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    controller = EchoVLLMBatchedRolloutController(
        model=llm,
        processor=None,
        work_dir=args.work_dir,
    )
    return llm, controller


def _convert_batched_rollout_to_legacy(rollout: dict, sample: dict) -> dict:
    unique_segments = rollout.get("unique_segments", []) or []
    pred_answer = rollout.get("model_prediction", "") or ""
    rounds = rollout.get("rounds", []) or []
    finish_reason = rollout.get("finish_reason", "unknown") or "unknown"
    used_segments = [
        {
            "round": seg.get("round"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "segment_path": seg.get("segment_path"),
        }
        for seg in unique_segments
    ]
    return {
        "request_id": rollout.get("request_id"),
        "sample_id": rollout.get("sample_id", sample.get("id", "?")),
        "rollout_id": rollout.get("rollout_id"),
        "final_response": rollout.get("final_response", "") or "",
        "pred_answer": pred_answer,
        "avg_logprob": rollout.get("avg_logprob"),
        "total_rounds": len(rounds),
        "used_segments": used_segments,
        "used_segment_paths": [
            seg.get("segment_path") for seg in unique_segments if seg.get("segment_path")
        ],
        "stop_reason": finish_reason,
        "round_outputs": rounds,
        "triggered_interleaved": len(used_segments) > 0,
        "has_final_answer": bool(pred_answer),
        "answer_correct": bool(pred_answer) and pred_answer == sample.get("answer", ""),
        "finalized": rollout.get("finalized", False),
        "error": rollout.get("error"),
        "reward_ready": rollout.get("reward_ready"),
    }


def run_one_sample(runtime_model, runtime_aux, sample: dict, args: argparse.Namespace) -> dict:
    """Run rollouts for a single sample. Returns result dict."""
    signal.alarm(args.timeout)
    t_start = time.time()

    result = {
        "sample_id": sample.get("id", "?"),
        "rollouts": [],
        "worker_error": None,
        "worker_elapsed_s": 0.0,
        "rollout_backend": args.rollout_backend,
    }

    try:
        if args.rollout_backend == "vllm_batched":
            controller = runtime_aux
            batched_results = controller.run_batch([
                {
                    "request_id": sample.get("id", "?"),
                    "sample_id": sample.get("id", "?"),
                    "audio_path": sample["audio_path"],
                    "question": sample["question"],
                    "choices": sample["choices"],
                    "num_rollouts": args.num_generations,
                    "temperature": args.temperature,
                    "max_rounds": args.max_rounds,
                    "max_tokens": args.max_new_tokens,
                    "stop_on_duplicate": True,
                    "finalize_on_stop": True,
                    "on_duplicate_seg": "stop",
                    "finalize_max_tokens": args.finalize_max_new_tokens,
                    "work_dir": args.work_dir,
                }
            ])
            for rollout in batched_results:
                result["rollouts"].append(_convert_batched_rollout_to_legacy(rollout, sample))
            return result

        for r_idx in range(args.num_generations):
            try:
                with torch.no_grad():
                    rollout = run_interleaved(
                        runtime_model, runtime_aux,
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

    runtime_model, runtime_aux = _load_runtime(args)
    if args.rollout_backend == "hf":
        runtime_model.eval()

    result = run_one_sample(runtime_model, runtime_aux, sample, args)

    json.dump(result, _real_stdout, ensure_ascii=False)
    _real_stdout.flush()


def _main_persistent(args: argparse.Namespace, _real_stdout) -> None:
    """Persistent worker: load model once, process tasks from stdin."""
    print("[persistent_worker] Loading model ...", file=sys.stderr)
    runtime_model, runtime_aux = _load_runtime(args)
    if args.rollout_backend == "hf":
        runtime_model.eval()
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
                     "temperature", "finalize_max_new_tokens", "timeout", "work_dir"):
            if key in task:
                setattr(args, key, task[key])

        task_count += 1
        result = run_one_sample(runtime_model, runtime_aux, sample, args)

        # Write result line to real stdout
        json.dump(result, _real_stdout, ensure_ascii=False)
        _real_stdout.write("\n")
        _real_stdout.flush()


if __name__ == "__main__":
    main()

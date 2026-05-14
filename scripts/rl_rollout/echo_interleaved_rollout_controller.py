#!/usr/bin/env python3
"""
Echo interleaved rollout controller built from existing project logic.

This controller does not re-invent Echo rollout. It stitches together:

- Author original vLLM multiturn prompt/audio-append pattern from
  ``inference/inference_multiturn.py``
- Duplicate guard, finalize-on-stop, and answer extraction behavior from
  ``scripts/interleaved_infer.py`` and ``scripts/03_interleaved``
- Reward-ready output fields aligned with ``echo_rl/rewards.py``

The main addition here is request-level batching for vLLM:

- batch multiple samples together
- expand per-sample rollout count into independent states
- allow each rollout to stop at different rounds
- isolate errors so one bad request does not kill the whole batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.interleaved_infer import (  # noqa: E402
    build_finalize_prompt,
    build_initial_prompt,
    clamp_seg,
    extract_answer,
    has_answer,
    is_duplicate_seg,
    parse_segments,
    save_segment_audio,
)
from scripts.rl_backends.rollout_backend_vllm import AUDIO_PLACEHOLDER  # noqa: E402

SAMPLE_RATE = 16000
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.8
DEFAULT_FINALIZE_MAX_TOKENS = 64


@dataclass
class EchoRolloutState:
    request_id: str
    sample_id: str
    rollout_id: int
    audio_path: str
    question: str
    choices: Optional[List[str]]
    prompt: str
    audio_inputs: List[Tuple[np.ndarray, int]]
    full_response: str = ""
    segments: List[dict] = field(default_factory=list)
    unique_segments: List[dict] = field(default_factory=list)
    rounds: List[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    finalized: bool = False
    error: Optional[str] = None
    base_audio: Optional[np.ndarray] = None
    sample_rate: int = SAMPLE_RATE
    duration: float = 0.0
    initial_prompt_text: str = ""
    max_rounds: int = 8
    max_tokens: int = 2048
    temperature: float = 1.0
    stop_on_duplicate: bool = True
    finalize_on_stop: bool = True
    on_duplicate_seg: str = "stop"
    duplicate_iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD
    finalize_max_tokens: int = DEFAULT_FINALIZE_MAX_TOKENS
    work_dir: Optional[str] = None
    # Confidence tracking: accumulated log-probs from vLLM rollout
    logprob_sum: float = 0.0
    logprob_count: int = 0
    phase: str = "interleaved"  # interleaved | finalize | done | error
    round_index: int = 0
    seen_seg_count: int = 0

    @property
    def active(self) -> bool:
        return self.phase in {"interleaved", "finalize"} and not self.error


def _json_choice_list(choices: Optional[List[str]]) -> List[str]:
    if not choices:
        return []
    return list(choices)


def _make_interleaved_prompt(initial_prompt_text: str) -> str:
    return (
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}\n{initial_prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _make_finalize_prompt(state: EchoRolloutState) -> str:
    prompt = (
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}\n"
        f"{state.initial_prompt_text}<|im_end|>\n"
    )
    if state.full_response.strip():
        prompt += f"<|im_start|>assistant\n{state.full_response.strip()}<|im_end|>\n"
    prompt += (
        f"<|im_start|>user\n{build_finalize_prompt()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def _append_stop_token(text: str, stop_reason: Any) -> str:
    stop_reason = str(stop_reason or "")
    if stop_reason == "</seg>":
        return text + "</seg>"
    if stop_reason == "</answer>":
        return text + "</answer>"
    return text


def _map_finish_reason(output: Any) -> str:
    stop_reason = str(getattr(output, "stop_reason", "") or "")
    finish_reason = str(getattr(output, "finish_reason", "") or "")
    if stop_reason == "</seg>":
        return "seg"
    if stop_reason == "</answer>":
        return "answer"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop":
        return "eos"
    return stop_reason or finish_reason or "unknown"


def _insert_audio_tags_after_unique_segments(
    response_text: str,
    raw_segments: List[Tuple[float, float]],
    accepted_indices: Iterable[int],
) -> str:
    accepted = set(accepted_indices)
    if not accepted:
        return response_text
    import re

    pattern = re.compile(r"<seg>\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*</seg>")
    pieces: List[str] = []
    last = 0
    seg_idx = 0
    for match in pattern.finditer(response_text):
        pieces.append(response_text[last:match.end()])
        if seg_idx in accepted:
            pieces.append(AUDIO_PLACEHOLDER)
        last = match.end()
        seg_idx += 1
    pieces.append(response_text[last:])
    return "".join(pieces)


class EchoVLLMBatchedRolloutController:
    """Request-level batched Echo rollout controller on top of vLLM."""

    def __init__(
        self,
        model: Any,
        processor: Any = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        duplicate_iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD,
        finalize_max_tokens: int = DEFAULT_FINALIZE_MAX_TOKENS,
        work_dir: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.sample_rate = sample_rate
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.finalize_max_tokens = finalize_max_tokens
        self.work_dir = work_dir or os.path.join(ROOT, "output", "interleaved_tmp", "batched_controller")

    def run_one(self, request: dict) -> dict:
        return self.run_batch([request])[0]

    def run_batch(self, requests: List[dict]) -> List[dict]:
        states = self._expand_requests(requests)

        while True:
            active = [s for s in states if s.active]
            if not active:
                break

            groups = self._group_active_states(active)
            for group_key, group_states in groups.items():
                self._run_group(group_key, group_states)

        return [self._serialize_state(state) for state in states]

    def _expand_requests(self, requests: List[dict]) -> List[EchoRolloutState]:
        states: List[EchoRolloutState] = []
        for req_idx, request in enumerate(requests):
            audio_path = request["audio_path"]
            question = request["question"]
            choices = _json_choice_list(request.get("choices"))
            sample_id = str(request.get("sample_id") or request.get("id") or f"sample_{req_idx}")
            request_base = str(request.get("request_id") or sample_id)
            num_rollouts = int(
                request.get("num_rollouts")
                or request.get("n_rollouts")
                or request.get("n")
                or 1
            )
            max_rounds = int(request.get("max_rounds", 8))
            max_tokens = int(request.get("max_tokens", 2048))
            temperature = float(request.get("temperature", 1.0))
            stop_on_duplicate = bool(request.get("stop_on_duplicate", True))
            finalize_on_stop = bool(request.get("finalize_on_stop", True))
            on_duplicate_seg = str(request.get("on_duplicate_seg", "stop"))
            duplicate_iou_threshold = float(
                request.get("duplicate_iou_threshold", self.duplicate_iou_threshold)
            )
            finalize_max_tokens = int(
                request.get("finalize_max_tokens", self.finalize_max_tokens)
            )
            work_dir = request.get("work_dir") or self.work_dir
            initial_prompt_text = build_initial_prompt(question, choices)

            for rollout_id in range(num_rollouts):
                request_id = f"{request_base}::r{rollout_id}"
                state = EchoRolloutState(
                    request_id=request_id,
                    sample_id=sample_id,
                    rollout_id=rollout_id,
                    audio_path=audio_path,
                    question=question,
                    choices=choices,
                    prompt=_make_interleaved_prompt(initial_prompt_text),
                    audio_inputs=[],
                    initial_prompt_text=initial_prompt_text,
                    max_rounds=max_rounds,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_on_duplicate=stop_on_duplicate,
                    finalize_on_stop=finalize_on_stop,
                    on_duplicate_seg=on_duplicate_seg,
                    duplicate_iou_threshold=duplicate_iou_threshold,
                    finalize_max_tokens=finalize_max_tokens,
                    work_dir=work_dir,
                )
                try:
                    full_audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                    duration = librosa.get_duration(path=audio_path)
                    state.base_audio = full_audio
                    state.sample_rate = sr
                    state.duration = duration
                    state.audio_inputs = [(full_audio, sr)]
                except Exception as exc:
                    state.error = f"audio_load_error: {exc}"
                    state.phase = "error"
                    state.finish_reason = "error"
                states.append(
                    state
                )
        return states

    def _group_active_states(self, states: List[EchoRolloutState]) -> dict:
        groups: Dict[Tuple[Any, ...], List[EchoRolloutState]] = {}
        for state in states:
            if state.phase == "finalize":
                key = (
                    "finalize",
                    state.temperature,
                    state.finalize_max_tokens,
                    tuple(["</answer>"]),
                )
            else:
                key = (
                    "interleaved",
                    state.temperature,
                    state.max_tokens,
                    tuple(["</seg>", "</answer>"]),
                )
            groups.setdefault(key, []).append(state)
        return groups

    def _run_group(self, group_key: Tuple[Any, ...], states: List[EchoRolloutState]) -> None:
        phase, temperature, max_tokens, stop_words = group_key
        if not states:
            return
        try:
            self._generate_and_dispatch(states, temperature, max_tokens, list(stop_words), phase)
        except Exception as batch_error:
            for state in states:
                try:
                    self._generate_and_dispatch([state], temperature, max_tokens, list(stop_words), phase)
                except Exception as single_error:
                    state.error = f"{phase}_generate_error: {single_error}"
                    state.phase = "error"
                    state.finish_reason = state.finish_reason or "error"
            if len(states) > 1:
                for state in states:
                    if state.phase == "error" and state.error and "batch_fallback" not in state.error:
                        state.error += f" | batch_fallback_from={batch_error}"

    def _generate_and_dispatch(
        self,
        states: List[EchoRolloutState],
        temperature: float,
        max_tokens: int,
        stop_words: List[str],
        phase: str,
    ) -> None:
        from vllm import SamplingParams

        prompts: List[Dict[str, Any]] = []
        for state in states:
            if phase == "finalize":
                prompt = _make_finalize_prompt(state)
                audios = [state.audio_inputs[0]]
            else:
                prompt = state.prompt
                audios = state.audio_inputs
            prompts.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "audio": [(audio_arr, sr) for audio_arr, sr in audios],
                    },
                }
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_words,
            skip_special_tokens=False,
            logprobs=1,
        )

        outputs = self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        for state, request_output in zip(states, outputs):
            out = request_output.outputs[0] if request_output.outputs else None
            if out is None:
                state.error = f"{phase}_empty_output"
                state.phase = "error"
                state.finish_reason = "error"
                continue

            # Capture token-level logprobs for confidence computation
            if out.logprobs:
                for step_lp in out.logprobs:
                    for token_id, lp_info in step_lp.items():
                        state.logprob_sum += lp_info.logprob
                        state.logprob_count += 1

            response = _append_stop_token(out.text or "", getattr(out, "stop_reason", None))
            stop_reason = _map_finish_reason(out)
            if phase == "finalize":
                self._handle_finalize_output(state, response, stop_reason)
            else:
                self._handle_interleaved_output(state, response, stop_reason)

    def _handle_interleaved_output(
        self,
        state: EchoRolloutState,
        response: str,
        stop_reason: str,
    ) -> None:
        state.round_index += 1
        raw_segs = parse_segments(response)
        round_info = {
            "round": state.round_index,
            "phase": "interleaved",
            "response": response,
            "stop_reason": stop_reason,
            "detected_segments": [],
            "inserted_segments": [],
            "duplicate_segments": [],
        }
        state.full_response += response

        accepted_seg_indices: List[int] = []
        duplicate_triggered = False

        for seg_idx, (start_raw, end_raw) in enumerate(raw_segs):
            clamped = clamp_seg(start_raw, end_raw, state.duration)
            if clamped is None:
                round_info["detected_segments"].append(
                    {"raw_start": start_raw, "raw_end": end_raw, "valid": False}
                )
                continue

            start, end = clamped
            dup_seg, iou = is_duplicate_seg(
                start,
                end,
                state.unique_segments,
                state.duplicate_iou_threshold,
            )
            seg_record = {
                "round": state.round_index,
                "start": start,
                "end": end,
                "raw_start": start_raw,
                "raw_end": end_raw,
                "is_duplicate": dup_seg is not None,
                "iou": round(iou, 4) if dup_seg is not None else None,
            }
            state.segments.append(seg_record)
            round_info["detected_segments"].append(seg_record)

            if dup_seg is not None:
                round_info["duplicate_segments"].append(
                    {
                        "start": start,
                        "end": end,
                        "iou": round(iou, 4),
                        "duplicate_of": {
                            "start": dup_seg["start"],
                            "end": dup_seg["end"],
                        },
                    }
                )
                if state.stop_on_duplicate and state.on_duplicate_seg == "stop":
                    duplicate_triggered = True
                    state.finish_reason = "duplicate_seg"
                    break
                if state.on_duplicate_seg in {"ignore_continue", "insert_once_continue"}:
                    continue

            segment = state.base_audio[int(start * state.sample_rate): int(end * state.sample_rate)]
            seg_path = save_segment_audio(
                state.base_audio,
                state.sample_rate,
                start,
                end,
                state.work_dir or self.work_dir,
                state.round_index,
                len(state.unique_segments) + 1,
            )
            state.unique_segments.append(
                {
                    "round": state.round_index,
                    "start": start,
                    "end": end,
                    "segment_path": seg_path,
                }
            )
            state.audio_inputs.append((segment, state.sample_rate))
            accepted_seg_indices.append(seg_idx)
            round_info["inserted_segments"].append(
                {"start": start, "end": end, "segment_path": seg_path}
            )

        prompt_append = _insert_audio_tags_after_unique_segments(
            response,
            raw_segs,
            accepted_seg_indices,
        )
        state.prompt += prompt_append
        state.rounds.append(round_info)

        if duplicate_triggered:
            if has_answer(response) or has_answer(state.full_response):
                state.finish_reason = "has_answer"
                state.phase = "done"
            elif state.finalize_on_stop:
                state.phase = "finalize"
            else:
                state.phase = "done"
            return

        if has_answer(response) or stop_reason == "answer":
            state.finish_reason = "has_answer"
            state.phase = "done"
            return

        if state.round_index >= state.max_rounds:
            state.finish_reason = "max_rounds"
            if state.finalize_on_stop and not has_answer(state.full_response):
                state.phase = "finalize"
            else:
                state.phase = "done"
            return

        if not response:
            state.finish_reason = "empty_response"
            state.phase = "done"
            return

        if raw_segs:
            state.finish_reason = "continue"
        else:
            state.finish_reason = "continue_no_seg"
        state.phase = "interleaved"

    def _handle_finalize_output(
        self,
        state: EchoRolloutState,
        response: str,
        stop_reason: str,
    ) -> None:
        state.round_index += 1
        state.full_response += response
        state.finalized = True
        state.finish_reason = state.finish_reason or f"finalize_{stop_reason}"
        state.rounds.append(
            {
                "round": state.round_index,
                "phase": "finalize",
                "response": response,
                "stop_reason": stop_reason,
                "detected_segments": [],
                "inserted_segments": [],
                "duplicate_segments": [],
            }
        )
        state.phase = "done"

    def _serialize_state(self, state: EchoRolloutState) -> dict:
        pred = extract_answer(state.full_response)
        avg_logprob = (state.logprob_sum / state.logprob_count) if state.logprob_count > 0 else None
        return {
            "request_id": state.request_id,
            "sample_id": state.sample_id,
            "rollout_id": state.rollout_id,
            "final_response": state.full_response,
            "model_prediction": pred,
            "avg_logprob": avg_logprob,
            "segments": state.segments,
            "unique_segments": [
                {
                    "round": seg["round"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "segment_path": seg["segment_path"],
                }
                for seg in state.unique_segments
            ],
            "rounds": state.rounds,
            "finish_reason": state.finish_reason,
            "finalized": state.finalized,
            "error": state.error,
            "reward_ready": {
                "response": state.full_response,
                "answer": pred,
                "segments": [
                    (seg["start"], seg["end"]) for seg in state.unique_segments
                ],
            },
        }


def run_echo_interleaved_rollout(
    model,
    processor,
    audio_path,
    question,
    choices=None,
    temperature=1.0,
    max_rounds=8,
    max_tokens=2048,
    stop_on_duplicate=True,
    finalize_on_stop=True,
    on_duplicate_seg="stop",
    work_dir=None,
) -> dict:
    controller = EchoVLLMBatchedRolloutController(
        model=model,
        processor=processor,
        work_dir=work_dir,
    )
    return controller.run_one(
        {
            "request_id": "single_request",
            "sample_id": os.path.splitext(os.path.basename(audio_path))[0],
            "audio_path": audio_path,
            "question": question,
            "choices": choices or [],
            "temperature": temperature,
            "max_rounds": max_rounds,
            "max_tokens": max_tokens,
            "stop_on_duplicate": stop_on_duplicate,
            "finalize_on_stop": finalize_on_stop,
            "on_duplicate_seg": on_duplicate_seg,
            "work_dir": work_dir,
        }
    )


def _pick_manifest_samples(manifest_path: str, num_samples: int) -> List[dict]:
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if os.path.exists(row.get("audio_path", "")):
                samples.append(row)
            if len(samples) >= num_samples:
                break
    return samples


def _build_requests_from_manifest(
    manifest_path: str,
    num_samples: int,
    num_rollouts_per_sample: int,
    temperature: float,
    max_rounds: int,
    max_tokens: int,
    work_dir: Optional[str],
) -> List[dict]:
    rows = _pick_manifest_samples(manifest_path, num_samples)
    requests = []
    for row in rows:
        requests.append(
            {
                "request_id": row.get("id") or str(uuid.uuid4()),
                "sample_id": row.get("id") or os.path.splitext(os.path.basename(row["audio_path"]))[0],
                "audio_path": row["audio_path"],
                "question": row["question"],
                "choices": row.get("choices", []),
                "num_rollouts": num_rollouts_per_sample,
                "temperature": temperature,
                "max_rounds": max_rounds,
                "max_tokens": max_tokens,
                "stop_on_duplicate": True,
                "finalize_on_stop": True,
                "on_duplicate_seg": "stop",
                "work_dir": work_dir,
            }
        )
    return requests


def _load_vllm_model(args):
    from vllm import LLM

    return LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


def main():
    parser = argparse.ArgumentParser(description="Batched Echo interleaved rollout on vLLM")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio_path")
    parser.add_argument("--question")
    parser.add_argument("--choices", default="[]")
    parser.add_argument("--manifest_path")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_rollouts_per_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--work_dir", default=os.path.join(ROOT, "output", "interleaved_tmp", "batched_controller"))
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    model = _load_vllm_model(args)
    controller = EchoVLLMBatchedRolloutController(model=model, processor=None, work_dir=args.work_dir)

    if args.manifest_path:
        requests = _build_requests_from_manifest(
            manifest_path=args.manifest_path,
            num_samples=args.num_samples,
            num_rollouts_per_sample=args.num_rollouts_per_sample,
            temperature=args.temperature,
            max_rounds=args.max_rounds,
            max_tokens=args.max_tokens,
            work_dir=args.work_dir,
        )
    else:
        if not args.audio_path or not args.question:
            raise ValueError("Either --manifest_path or both --audio_path and --question are required.")
        requests = [
            {
                "request_id": "single_request",
                "sample_id": os.path.splitext(os.path.basename(args.audio_path))[0],
                "audio_path": args.audio_path,
                "question": args.question,
                "choices": json.loads(args.choices),
                "num_rollouts": 1,
                "temperature": args.temperature,
                "max_rounds": args.max_rounds,
                "max_tokens": args.max_tokens,
                "stop_on_duplicate": True,
                "finalize_on_stop": True,
                "on_duplicate_seg": "stop",
                "work_dir": args.work_dir,
            }
        ]

    t0 = time.time()
    results = controller.run_batch(requests)
    elapsed = time.time() - t0

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "num_requests": len(requests),
        "num_rollout_results": len(results),
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved rollout results to {args.output_json}")
    print(f"Requests: {len(requests)} | Rollout results: {len(results)} | Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

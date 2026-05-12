#!/usr/bin/env python3
"""
GRPO training with subprocess-isolated rollouts.

Features (controlled by CLI switches):
  --rollout_worker_mode:
      per_task       Current stable: one subprocess per sample (default)
      persistent     Single long-lived worker, stdin/stdout JSON line protocol
      pool           Multiple persistent workers, round-robin, optional multi-GPU

  --grpo_forward_mode:
      text_only            Current stable: text-only thinker forward (default)
      strict_interleaved   Experimental: full multimodal forward with audio context,
                           audio tokens masked from loss

Design:
  - Workers run FIRST (main process has NO models on GPU)
  - After workers complete, main process loads models -> trains -> unloads
  - This ensures complete CUDA context isolation
"""
from __future__ import annotations

import argparse
import json
import os
import gc
import random
import subprocess
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_grpo_smoke"))
from echo_rl.rollout_rewards import rollout_reward
from grpo_utils import (
    build_rollout_metadata,
    build_text_inputs,
    build_text_prompt,
    compute_advantages,
    compute_grpo_loss,
    get_per_token_logps,
)

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "isolated_rollout_worker.py")

# Audio token IDs for Qwen2.5-Omni
AUDIO_BOS_ID = 151647
AUDIO_EOS_ID = 151648
AUDIO_TOKEN_ID = 151646


# ── args ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--data_path", default="dataJson/NAQA/EAQA_RL.jsonl")
    p.add_argument("--output_dir", default="output/grpo_isolated_smoke")
    p.add_argument("--max_samples", type=int, default=20)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--kl_coef", type=float, default=0.04)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rounds", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--finalize_max_new_tokens", type=int, default=64)
    p.add_argument("--worker_timeout", type=int, default=600)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--policy_forward_micro_batch_size", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--checkpoint_every", type=int, default=30)
    # New: worker mode
    p.add_argument("--rollout_worker_mode", default="per_task",
                   choices=["per_task", "persistent", "pool"])
    p.add_argument("--num_rollout_workers", type=int, default=1)
    p.add_argument("--worker_devices", default="",
                   help="Comma-separated GPU IDs for pool workers, e.g. '0,1,2,3'")
    # New: GRPO forward mode
    p.add_argument("--grpo_forward_mode", default="text_only",
                   choices=["text_only", "strict_interleaved"])
    return p.parse_args()


# ── data ──

def load_dataset(path: str, max_samples: int) -> List[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if "multi_choice" in s and "choices" not in s:
                s["choices"] = s["multi_choice"]
            if os.path.exists(s.get("audio_path", "")):
                samples.append(s)
            if len(samples) >= max_samples:
                break
    return samples


# ── per-task worker (original) ──

def run_worker(
    sample: dict, model_path: str, adapter_path: str,
    gpu_id: int,
    max_rounds: int, max_new_tokens: int, num_rollouts: int,
    temperature: float, finalize_max_new_tokens: int, timeout: int,
) -> dict:
    sample_id = sample.get("id", "?")

    cmd = [
        sys.executable, "-u", WORKER_SCRIPT,
        "--sample_json", json.dumps(sample, ensure_ascii=False),
        "--model_path", model_path,
        "--adapter_path", adapter_path,
        "--max_rounds", str(max_rounds),
        "--max_new_tokens", str(max_new_tokens),
        "--num_generations", str(num_rollouts),
        "--temperature", str(temperature),
        "--finalize_max_new_tokens", str(finalize_max_new_tokens),
        "--timeout", str(timeout - 10),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["QWEN_OMNI_SKIP_SPK"] = "1"

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        else:
            return {
                "sample_id": sample_id, "rollouts": [],
                "worker_error": f"no stdout; stderr={proc.stderr[-300:]}",
            }
    except subprocess.TimeoutExpired:
        return {"sample_id": sample_id, "rollouts": [], "worker_error": "timeout"}
    except json.JSONDecodeError:
        return {"sample_id": sample_id, "rollouts": [],
                "worker_error": f"bad json; stdout={proc.stdout[-300:]} stderr={proc.stderr[-300:]}"}
    except Exception as e:
        return {"sample_id": sample_id, "rollouts": [], "worker_error": str(e)[:300]}


# ── persistent worker handle ──

class PersistentWorkerHandle:
    """Manages a single persistent rollout worker subprocess."""

    def __init__(self, model_path: str, adapter_path: str, gpu_id: int,
                 max_rounds: int, max_new_tokens: int, num_rollouts: int,
                 temperature: float, finalize_max_new_tokens: int, timeout: int):
        self.gpu_id = gpu_id
        self.num_rollouts = num_rollouts
        self.restart_count = 0

        cmd = [
            sys.executable, "-u", WORKER_SCRIPT,
            "--model_path", model_path,
            "--adapter_path", adapter_path,
            "--max_rounds", str(max_rounds),
            "--max_new_tokens", str(max_new_tokens),
            "--num_generations", str(num_rollouts),
            "--temperature", str(temperature),
            "--finalize_max_new_tokens", str(finalize_max_new_tokens),
            "--timeout", str(timeout - 10),
            "--persistent",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["QWEN_OMNI_SKIP_SPK"] = "1"

        self.proc = subprocess.Popen(
            cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._drain_stderr, daemon=True,
        )
        self._stderr_reader.start()

    def _drain_stderr(self):
        for _line in self.proc.stderr:
            pass

    def send(self, sample: dict) -> None:
        task = {"sample": sample}
        line = json.dumps(task, ensure_ascii=False) + "\n"
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError(f"Worker stdin closed (worker likely crashed)")

    def recv(self, timeout: float = 600.0) -> dict:
        import select
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(f"Worker recv timed out after {timeout}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Worker stdout closed (worker likely crashed)")
        return json.loads(line.strip())

    def restart(self, model_path: str, adapter_path: str,
                max_rounds: int, max_new_tokens: int, num_rollouts: int,
                temperature: float, finalize_max_new_tokens: int, timeout: int) -> None:
        """Kill and restart the worker."""
        self.shutdown()
        self.__init__(model_path, adapter_path, self.gpu_id,
                      max_rounds, max_new_tokens, num_rollouts,
                      temperature, finalize_max_new_tokens, timeout)
        self.restart_count += 1

    def shutdown(self) -> None:
        try:
            self.proc.stdin.write('{"action":"shutdown"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# ── worker pool ──

class WorkerPool:
    """Pool of persistent rollout workers, optionally multi-GPU."""

    def __init__(self, model_path: str, adapter_path: str,
                 num_workers: int, devices: List[int],
                 max_rounds: int, max_new_tokens: int, num_rollouts: int,
                 temperature: float, finalize_max_new_tokens: int, timeout: int):
        self.num_workers = num_workers
        self.devices = devices
        self.workers: List[PersistentWorkerHandle] = []
        self.restart_count = 0
        self.failed_sample_ids: List[str] = []

        for i in range(num_workers):
            gpu = devices[i] if i < len(devices) else devices[-1]
            wh = PersistentWorkerHandle(
                model_path, adapter_path, gpu,
                max_rounds, max_new_tokens, num_rollouts,
                temperature, finalize_max_new_tokens, timeout,
            )
            self.workers.append(wh)

        self._worker_args = (model_path, adapter_path,
                             max_rounds, max_new_tokens, num_rollouts,
                             temperature, finalize_max_new_tokens, timeout)

    def map(self, samples: List[dict]) -> List[dict]:
        """Distribute samples round-robin to workers, collect results."""
        results: List[Optional[dict]] = [None] * len(samples)
        # Dispatch all tasks
        for i, sample in enumerate(samples):
            worker_idx = i % self.num_workers
            wh = self.workers[worker_idx]
            wh.send(sample)

        # Collect all results
        for i, sample in enumerate(samples):
            worker_idx = i % self.num_workers
            wh = self.workers[worker_idx]
            try:
                results[i] = wh.recv(timeout=600)
            except (TimeoutError, RuntimeError, Exception) as e:
                print(f"    [pool] Worker {worker_idx} error on sample "
                      f"{sample.get('id','?')}: {e}")
                results[i] = {
                    "sample_id": sample.get("id", "?"), "rollouts": [],
                    "worker_error": str(e)[:300],
                }
                # Restart worker
                try:
                    wh.restart(*self._worker_args)
                    self.restart_count += 1
                except Exception as re:
                    print(f"    [pool] Failed to restart worker {worker_idx}: {re}")

        return results

    def shutdown(self) -> None:
        for wh in self.workers:
            wh.shutdown()


# ── model load/unload for training ──

def load_training_models(model_path: str, adapter_path: str, device: torch.device,
                        disable_talker: bool = True):
    """Load policy + reference models for training.

    Args:
        disable_talker: If True, disable talker (saves VRAM, used in text_only mode).
                        If False, keep talker (needed for strict_interleaved audio forward).
    """
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )
    policy_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    if disable_talker:
        policy_model.base_model.disable_talker()
    policy_model = policy_model.to(device, dtype=torch.float16)
    for n, p in policy_model.named_parameters():
        p.requires_grad_("lora" in n)
    policy_model.eval()

    base2 = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )
    ref_model = PeftModel.from_pretrained(base2, adapter_path)
    if disable_talker:
        ref_model.base_model.disable_talker()
    ref_model = ref_model.to(device, dtype=torch.float16)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return policy_model, ref_model, processor


def unload_training_models(policy_model, ref_model):
    """Release all GPU memory held by training models."""
    del policy_model
    del ref_model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


# ── strict interleaved forward ──

def build_strict_interleaved_input(
    processor, sample: dict, rollout_data: dict, device: torch.device,
    max_length: int = 2048,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                    Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    Reconstruct interleaved multimodal sequence.

    Sequence: [full_audio][prompt][R1_text][seg1_audio][R2_text]...

    Returns:
        input_ids: (1, T)
        attention_mask: (1, T)
        loss_mask: (1, T-1)  — 1 for model-generated text tokens, 0 elsewhere
        input_features: audio features tensor, or None
        feature_attention_mask: audio feature attention mask, or None
    """
    try:
        import librosa
        from scripts.interleaved_infer import build_conversation

        round_outputs = rollout_data.get("round_outputs", [])
        all_round_texts = []
        segs_by_round = {}
        used_segments = rollout_data.get("used_segments", [])

        for ro in round_outputs:
            if isinstance(ro.get("round"), int):
                all_round_texts.append(ro.get("text", ""))
        for seg in used_segments:
            r = seg.get("round", 0)
            segs_by_round.setdefault(r, []).append(seg)

        # Build final conversation (for finalization)
        conversation = build_conversation(
            audio_path=sample["audio_path"],
            prompt=build_text_prompt(sample["question"], sample.get("choices", [])),
            all_round_texts=all_round_texts,
            used_segments=used_segments,
            is_finalize=False,
            sample_rate=16000,
        )

        full_audio, sr = librosa.load(sample["audio_path"], sr=16000)

        # Build content: [full_audio][prompt] + interleaved rounds
        content = [
            {"type": "audio", "audio": full_audio},
            {"type": "text", "text": build_text_prompt(
                sample["question"], sample.get("choices", []))},
        ]

        for i, round_text in enumerate(all_round_texts):
            content.append({"type": "text", "text": round_text.strip()})
            for seg_info in segs_by_round.get(i + 1, []):
                seg_audio, _ = librosa.load(seg_info["segment_path"], sr=16000)
                content.append({"type": "audio", "audio": seg_audio})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, add_generation_prompt=False,
                                              tokenize=False)

        # Gather all audios
        all_audios = [full_audio]
        for seg in used_segments:
            seg_audio, _ = librosa.load(seg["segment_path"], sr=16000)
            all_audios.append(seg_audio)

        if len(all_audios) > 1:
            inputs = processor(text=text, audio=all_audios, return_tensors="pt",
                               padding=True, sampling_rate=16000)
        else:
            inputs = processor(text=text, audio=all_audios[0], return_tensors="pt",
                               padding=True, sampling_rate=16000)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask",
                                     torch.ones_like(input_ids)).to(device)
        input_features = inputs.get("input_features")
        feature_attention_mask = inputs.get("feature_attention_mask")
        if input_features is not None:
            input_features = input_features.to(device)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device)

        # Build loss mask: find audio token spans and prompt span
        T = input_ids.shape[1]
        loss_mask = torch.ones(T - 1, dtype=torch.float, device=device)

        # Mask audio token spans
        ids = input_ids[0].tolist()
        in_audio = False
        for i, tid in enumerate(ids):
            if tid == AUDIO_BOS_ID:
                in_audio = True
            if in_audio and i < T - 1:
                loss_mask[i] = 0.0
            if tid == AUDIO_EOS_ID:
                in_audio = False
                if i < T - 1:
                    loss_mask[i] = 0.0  # eos token also masked

        # Mask prompt tokens (everything up to the first model-generated text)
        # Find prompt boundary: tokenize just the prompt part
        prompt_text = build_text_prompt(sample["question"],
                                         sample.get("choices", []))
        prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)
        # Find where prompt_ids appears in input_ids
        prompt_end = len(prompt_ids)
        # Account for audio tokens before prompt
        audio_prefix_len = _count_audio_tokens(ids[:prompt_end + 100])
        mask_start = prompt_end + audio_prefix_len
        for i in range(min(mask_start, T - 1)):
            loss_mask[i] = 0.0

        n_masked_text = loss_mask.sum().item()
        n_audio_masked = (T - 1) - loss_mask.sum().item()
        print(f"    [strict] input_ids={input_ids.shape}, "
              f"masked_text_tokens={int(n_masked_text)}, "
              f"masked_audio+prompt={int(n_audio_masked)}",
              file=sys.stderr)

        return (input_ids, attention_mask, loss_mask,
                input_features, feature_attention_mask)

    except Exception as e:
        print(f"    [strict] build_strict_interleaved_input failed: {e}",
              file=sys.stderr)
        return None


def _count_audio_tokens(ids: List[int]) -> int:
    """Count tokens in audio spans (bos, N x audio_token, eos)."""
    count = 0
    in_audio = False
    for tid in ids:
        if tid == AUDIO_BOS_ID:
            in_audio = True
        if in_audio:
            count += 1
        if tid == AUDIO_EOS_ID:
            in_audio = False
    return count


def get_per_token_logps_multimodal(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    input_features: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get per-token log-probs using thinker with optional audio features.
    Falls back to text-only if input_features is None.
    """
    thinker = model.get_base_model().thinker

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if input_features is not None:
        kwargs["input_features"] = input_features
    if feature_attention_mask is not None:
        kwargs["feature_attention_mask"] = feature_attention_mask

    outputs = thinker(**kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    log_probs = logits.log_softmax(dim=-1)
    per_token_logps = log_probs[:, :-1].gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    return per_token_logps  # (B, T-1)


# ── report ──

class RunReport:
    def __init__(self):
        self.rollout_success_count = 0
        self.rollout_failed_count = 0
        self.worker_restart_count = 0
        self.rollout_times: List[float] = []
        self.batch_times: List[float] = []
        self.strict_forward_success = 0
        self.strict_forward_failed = 0
        self.peak_memory_mb: List[int] = []

    def print(self):
        print(f"\n  ── Run Report ──")
        print(f"  rollout_success_count:  {self.rollout_success_count}")
        print(f"  rollout_failed_count:   {self.rollout_failed_count}")
        print(f"  worker_restart_count:   {self.worker_restart_count}")
        if self.rollout_times:
            avg = sum(self.rollout_times) / len(self.rollout_times)
            print(f"  avg_rollout_time:       {avg:.1f}s")
        if self.batch_times:
            total = sum(self.batch_times)
            print(f"  total_wall_time:        {total:.1f}s")
        if self.strict_forward_success + self.strict_forward_failed > 0:
            print(f"  strict_forward_success: {self.strict_forward_success}")
            print(f"  strict_forward_failed:  {self.strict_forward_failed}")
        if self.peak_memory_mb:
            print(f"  peak_memory_mb:         {max(self.peak_memory_mb)}")


# ── main ──

def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Parse worker devices
    worker_devices = (
        [int(x.strip()) for x in args.worker_devices.split(",") if x.strip()]
        if args.worker_devices else [args.gpu_id]
    )

    print(f"  Device: {device}")
    print(f"  Config: {vars(args)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.data_path, args.max_samples)
    print(f"  Dataset: {len(dataset)} samples")

    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    report = RunReport()
    global_step = 0
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    print(f"\n{'='*60}")
    print(f"  GRPO Training: rollout_mode={args.rollout_worker_mode}, "
          f"forward_mode={args.grpo_forward_mode}")
    print(f"  {len(dataset)} samples, {args.num_epochs} epoch(s), "
          f"{total_batches} batches")
    print(f"  Workers per batch: {args.batch_size}")
    print(f"  Rollouts per sample: {args.num_rollouts}")
    print(f"{'='*60}\n")

    # ── Worker pool (created once if pool mode) ──
    pool: Optional[WorkerPool] = None
    persistent_worker: Optional[PersistentWorkerHandle] = None

    if args.rollout_worker_mode == "pool":
        if args.num_rollout_workers > len(worker_devices):
            print(f"  WARNING: num_rollout_workers({args.num_rollout_workers}) > "
                  f"available devices({len(worker_devices)}), reducing")
            args.num_rollout_workers = len(worker_devices)
        pool = WorkerPool(
            args.model_path, args.adapter_path,
            args.num_rollout_workers, worker_devices,
            args.max_rounds, args.max_new_tokens, args.num_rollouts,
            args.temperature, args.finalize_max_new_tokens,
            args.worker_timeout,
        )
        print(f"  [pool] {args.num_rollout_workers} workers on devices "
              f"{worker_devices[:args.num_rollout_workers]}")

    if args.rollout_worker_mode == "persistent":
        persistent_worker = PersistentWorkerHandle(
            args.model_path, args.adapter_path,
            worker_devices[0],
            args.max_rounds, args.max_new_tokens, args.num_rollouts,
            args.temperature, args.finalize_max_new_tokens,
            args.worker_timeout,
        )
        print(f"  [persistent] worker on GPU {worker_devices[0]}")

    try:
        for epoch in range(args.num_epochs):
            random.shuffle(dataset)
            epoch_start = time.time()

            for batch_idx in range(0, len(dataset), args.batch_size):
                batch_start = time.time()
                batch = dataset[batch_idx:batch_idx + args.batch_size]
                current_batch_size = len(batch)

                # ═══ Phase 1: Run workers ═══
                print(f"\n  ── Batch {batch_idx // args.batch_size}: "
                      f"spawning {current_batch_size} workers ──")
                worker_results: List[dict] = []

                if args.rollout_worker_mode == "pool":
                    t_w = time.time()
                    worker_results = pool.map(batch)
                    w_total = time.time() - t_w
                    for i, (wres, sample) in enumerate(zip(worker_results, batch)):
                        sid = sample.get("id", "?")
                        n_ok = sum(1 for r in wres.get("rollouts", [])
                                  if r.get("stop_reason") != "error")
                        err = wres.get("worker_error", "")
                        status = "OK" if not err else f"ERR: {err[:60]}"
                        print(f"    {sid[:40]}: {n_ok}/{args.num_rollouts} ok  "
                              f"{status}")
                    print(f"  [pool] {current_batch_size} tasks in {w_total:.1f}s "
                          f"(~{w_total/current_batch_size:.1f}s per sample)")

                elif args.rollout_worker_mode == "persistent":
                    for sample in batch:
                        sid = sample.get("id", "?")
                        t_w = time.time()
                        try:
                            persistent_worker.send(sample)
                            wres = persistent_worker.recv(timeout=600)
                        except Exception as e:
                            print(f"    {sid[:40]}: ERR {e}")
                            persistent_worker.restart(
                                args.model_path, args.adapter_path,
                                args.max_rounds, args.max_new_tokens,
                                args.num_rollouts, args.temperature,
                                args.finalize_max_new_tokens,
                                args.worker_timeout,
                            )
                            report.worker_restart_count += 1
                            wres = {"sample_id": sid, "rollouts": [],
                                    "worker_error": str(e)[:300]}
                        w_elapsed = time.time() - t_w
                        report.rollout_times.append(w_elapsed)
                        n_ok = sum(1 for r in wres.get("rollouts", [])
                                  if r.get("stop_reason") != "error")
                        err = wres.get("worker_error", "")
                        status = "OK" if not err else f"ERR: {err[:60]}"
                        print(f"    {sid[:40]}: {n_ok}/{args.num_rollouts} ok  "
                              f"{w_elapsed:.1f}s  {status}")
                        worker_results.append(wres)

                else:  # per_task (original)
                    for sample in batch:
                        sid = sample.get("id", "?")
                        t_w = time.time()
                        wres = run_worker(
                            sample, args.model_path, args.adapter_path,
                            args.gpu_id,
                            args.max_rounds, args.max_new_tokens,
                            args.num_rollouts,
                            args.temperature, args.finalize_max_new_tokens,
                            args.worker_timeout,
                        )
                        w_elapsed = time.time() - t_w
                        report.rollout_times.append(w_elapsed)
                        n_ok = sum(1 for r in wres.get("rollouts", [])
                                  if r.get("stop_reason") != "error")
                        err = wres.get("worker_error", "")
                        status = "OK" if not err else f"ERR: {err[:60]}"
                        print(f"    {sid[:40]}: {n_ok}/{args.num_rollouts} ok  "
                              f"{w_elapsed:.1f}s  {status}")
                        worker_results.append(wres)

                # Count successes
                total_worker_ok = sum(
                    1 for wr in worker_results if not wr.get("worker_error")
                )
                n_ok = sum(
                    1 for wr in worker_results
                    for r in wr.get("rollouts", [])
                    if r.get("stop_reason") != "error"
                )
                n_fail = sum(
                    1 for wr in worker_results
                    for r in wr.get("rollouts", [])
                    if r.get("stop_reason") == "error"
                )
                report.rollout_success_count += n_ok
                report.rollout_failed_count += n_fail
                print(f"  Workers done: {total_worker_ok}/{current_batch_size} "
                      f"succeeded ({n_ok} rollouts ok, {n_fail} failed)")

                # ═══ Phase 2: Flatten results ═══
                all_results: List[Tuple[dict, dict]] = []
                for wres, sample in zip(worker_results, batch):
                    rollouts = wres.get("rollouts", [])
                    while len(rollouts) < args.num_rollouts:
                        rollouts.append({
                            "final_response": "", "pred_answer": "",
                            "total_rounds": 0, "used_segments": [],
                            "stop_reason": "error",
                        })
                    for r in rollouts:
                        all_results.append((r, sample))

                # ═══ Phase 3: Load models, rewards, training ═══
                print(f"\n  [{datetime.now()}] Loading training models ...")
                t_load = time.time()
                policy_model, ref_model, processor = load_training_models(
                    args.model_path, args.adapter_path, device,
                    disable_talker=(args.grpo_forward_mode == "text_only"),
                )
                tokenizer = processor.tokenizer
                optimizer = torch.optim.AdamW(
                    policy_model.parameters(), lr=args.learning_rate,
                )
                trainable = sum(
                    p.numel() for p in policy_model.parameters() if p.requires_grad
                )
                print(f"  Models loaded in {time.time() - t_load:.1f}s "
                      f"(trainable: {trainable:,})")

                try:
                    # Rewards
                    all_metrics: List[Dict[str, Any]] = []
                    for rollout_data, sample in all_results:
                        resp = rollout_data.get("final_response", "")
                        meta = build_rollout_metadata(rollout_data)
                        rew = rollout_reward(resp, sample.get("answer", ""), meta)
                        all_metrics.append({
                            "step": global_step,
                            "sample_id": sample.get("id", "?"),
                            "question": sample.get("question", "")[:200],
                            "gold_answer": sample.get("answer", ""),
                            "rollout_total": rew.get("rollout_total", 0.0),
                            "format": rew.get("format", 0.0),
                            "consistency": rew.get("consistency", 0.0),
                            "accuracy": rew.get("accuracy", 0.0),
                            "segment": rew.get("segment", 0.0),
                            "pred_answer": rollout_data.get("pred_answer", ""),
                            "has_answer": int(bool(rollout_data.get("pred_answer", ""))),
                            "is_correct": int(rollout_data.get("pred_answer", "") == sample.get("answer", "")),
                            "round_count": rollout_data.get("total_rounds", 0),
                            "stop_reason": rollout_data.get("stop_reason", "error"),
                            "triggered_interleaved": rollout_data.get("triggered_interleaved", False),
                            "has_final_answer": rollout_data.get("has_final_answer", False),
                            "answer_correct": rollout_data.get("answer_correct", False),
                            "final_response": resp,
                            "used_segments": rollout_data.get("used_segments", []),
                            "round_outputs": rollout_data.get("round_outputs", []),
                        })

                    # Advantages
                    rollout_totals = [m["rollout_total"] for m in all_metrics]
                    group_ids = [i // args.num_rollouts for i in range(len(all_results))]
                    advantages = compute_advantages(rollout_totals, group_ids).to(device)

                    # ═══ Forward/Backward ═══
                    if args.grpo_forward_mode == "strict_interleaved":
                        # ── Strict interleaved multimodal forward ──
                        policy_model.train()
                        micro_bs = args.policy_forward_micro_batch_size
                        if micro_bs > 1:
                            print(f"  [strict] forcing micro_batch_size=1")
                            micro_bs = 1
                        B_total = len(all_results)
                        optimizer.zero_grad()

                        log_loss_sum = log_kl_sum = log_ratio_sum = 0.0
                        log_tokens = 0
                        total_masked_tokens = 0

                        # Pre-compute interleaved inputs
                        strict_inputs = []
                        for rollout_data, sample in all_results:
                            inp = build_strict_interleaved_input(
                                processor, sample, rollout_data, device,
                            )
                            strict_inputs.append(inp)

                        for i, (inp, (rollout_data, sample)) in enumerate(
                            zip(strict_inputs, all_results)
                        ):
                            if inp is None:
                                report.strict_forward_failed += 1
                                continue
                            input_ids, attn_mask, loss_mask, input_features, feat_attn = inp
                            total_masked_tokens += loss_mask.sum().item()
                            report.strict_forward_success += 1

                            try:
                                policy_logps = get_per_token_logps_multimodal(
                                    policy_model, input_ids, attn_mask,
                                    input_features=input_features,
                                    feature_attention_mask=feat_attn,
                                )
                                old_logps = policy_logps.detach()
                                with torch.no_grad():
                                    ref_logps = get_per_token_logps_multimodal(
                                        ref_model, input_ids, attn_mask,
                                        input_features=input_features,
                                        feature_attention_mask=feat_attn,
                                    )

                                adv_i = advantages[i:i+1]
                                loss_dict = compute_grpo_loss(
                                    policy_logps, old_logps, adv_i,
                                    ref_logps=ref_logps, beta=args.kl_coef,
                                    epsilon=0.2, mask=loss_mask.unsqueeze(0),
                                )
                                n_tokens = loss_mask.sum()
                                if n_tokens > 0 and total_masked_tokens > 0:
                                    scale = n_tokens / total_masked_tokens
                                else:
                                    scale = 1.0 / max(B_total, 1)
                                (loss_dict["loss"] * scale).backward()

                                nt = max(n_tokens.item(), 1)
                                log_loss_sum += loss_dict["loss"].detach().item() * nt
                                log_kl_sum += loss_dict["kl"].detach().item() * nt
                                log_ratio_sum += loss_dict["ratio"].detach().item() * nt
                                log_tokens += nt

                                if hasattr(torch.cuda, 'max_memory_allocated'):
                                    report.peak_memory_mb.append(
                                        torch.cuda.max_memory_allocated(device) // 1024 // 1024
                                    )

                                del policy_logps, old_logps, ref_logps, loss_dict

                            except RuntimeError as e:
                                if "CUDA" in str(e) or "device-side assert" in str(e):
                                    print(f"  [strict] CUDA error on rollout {i}: "
                                          f"{str(e)[:100]}")
                                    report.strict_forward_failed += 1
                                    torch.cuda.empty_cache()
                                    continue
                                raise

                        if log_tokens > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                policy_model.parameters(), args.max_grad_norm,
                            )
                            optimizer.step()
                            agg_loss = log_loss_sum / log_tokens
                            agg_kl = log_kl_sum / log_tokens
                        else:
                            grad_norm = torch.tensor(0.0)
                            agg_loss = 0.0
                            agg_kl = 0.0

                    else:
                        # ── Text-only forward (stable path) ──
                        encoded = []
                        for rollout_data, sample in all_results:
                            prompt = build_text_prompt(
                                sample["question"], sample.get("choices", []),
                            )
                            completion = rollout_data.get("final_response", "")
                            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                            full_ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
                            full_ids = full_ids[:2048]
                            encoded.append((full_ids, len(full_ids), len(prompt_ids)))

                        B_total = len(encoded)
                        total_masked_tokens = sum(
                            max(0, (sl - 1) - (pl - 1)) for _, sl, pl in encoded
                        )

                        policy_model.train()
                        micro_bs = args.policy_forward_micro_batch_size
                        num_micro = (B_total + micro_bs - 1) // micro_bs
                        optimizer.zero_grad()

                        log_loss_sum = log_kl_sum = log_ratio_sum = 0.0
                        log_tokens = 0

                        for mb_idx in range(0, B_total, micro_bs):
                            mb_encoded = encoded[mb_idx:mb_idx + micro_bs]
                            mb_adv = advantages[mb_idx:mb_idx + micro_bs]
                            mb_bs = len(mb_encoded)
                            mb_max_len = max(sl for _, sl, _ in mb_encoded)

                            mb_ids = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
                            mb_attn = torch.zeros((mb_bs, mb_max_len), dtype=torch.long, device=device)
                            mb_cmask = torch.zeros((mb_bs, mb_max_len - 1), dtype=torch.float, device=device)
                            for j, (full_ids, seq_len, prompt_len) in enumerate(mb_encoded):
                                mb_ids[j, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=device)
                                mb_attn[j, :seq_len] = 1
                                if seq_len - 1 > prompt_len - 1:
                                    mb_cmask[j, prompt_len - 1:seq_len - 1] = 1.0

                            mb_policy_logps = get_per_token_logps(policy_model, mb_ids, mb_attn)
                            mb_old_logps = mb_policy_logps.detach()
                            with torch.no_grad():
                                mb_ref_logps = get_per_token_logps(ref_model, mb_ids, mb_attn)

                            loss_dict = compute_grpo_loss(
                                mb_policy_logps, mb_old_logps, mb_adv,
                                ref_logps=mb_ref_logps, beta=args.kl_coef, epsilon=0.2,
                                mask=mb_cmask,
                            )
                            n_tokens = mb_cmask.sum()
                            if n_tokens > 0 and total_masked_tokens > 0:
                                scale = n_tokens / total_masked_tokens
                            else:
                                scale = 1.0 / max(num_micro, 1)
                            (loss_dict["loss"] * scale).backward()

                            nt = max(n_tokens.item(), 1)
                            log_loss_sum += loss_dict["loss"].detach().item() * nt
                            log_kl_sum += loss_dict["kl"].detach().item() * nt
                            log_ratio_sum += loss_dict["ratio"].detach().item() * nt
                            log_tokens += nt

                            del mb_policy_logps, mb_old_logps, mb_ref_logps, loss_dict

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            policy_model.parameters(), args.max_grad_norm,
                        )
                        optimizer.step()
                        agg_loss = log_loss_sum / max(log_tokens, 1)
                        agg_kl = log_kl_sum / max(log_tokens, 1)

                    # ── Log ──
                    n = len(all_metrics)
                    avg_rt = sum(m["rollout_total"] for m in all_metrics) / max(n, 1)
                    avg_fmt = sum(m["format"] for m in all_metrics) / max(n, 1)
                    avg_cst = sum(m["consistency"] for m in all_metrics) / max(n, 1)
                    avg_acc = sum(m["accuracy"] for m in all_metrics) / max(n, 1)
                    avg_seg = sum(m["segment"] for m in all_metrics) / max(n, 1)
                    n_correct = sum(m["is_correct"] for m in all_metrics)

                    batch_time = time.time() - batch_start
                    report.batch_times.append(batch_time)

                    mode_tag = (
                        f"fw={args.grpo_forward_mode[:4]} "
                        f"wk={args.rollout_worker_mode[:4]}"
                    )
                    print(f"  step {global_step:3d} | loss {agg_loss:.4f} | "
                          f"R {avg_rt:+.3f} (fmt {avg_fmt:+.2f} cst {avg_cst:+.2f} "
                          f"acc {avg_acc:.2f} seg {avg_seg:.2f}) | "
                          f"correct {n_correct}/{n} | KL {agg_kl:.4f} | "
                          f"{batch_time:.1f}s | {mode_tag}")

                    writer.add_scalar("train/loss", agg_loss, global_step)
                    writer.add_scalar("train/approx_kl", agg_kl, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm.item() if hasattr(grad_norm, 'item') else 0.0, global_step)
                    for key in ("rollout_total", "format", "consistency", "accuracy", "segment"):
                        vals = [m[key] for m in all_metrics]
                        writer.add_scalar(f"reward/{key}", sum(vals) / max(len(vals), 1), global_step)

                    if global_step > 0 and global_step % args.checkpoint_every == 0:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
                        policy_model.save_pretrained(ckpt_dir)
                        print(f"  -> Checkpoint: {ckpt_dir}")

                    log_path = os.path.join(args.output_dir, "logs", "rollouts.jsonl")
                    with open(log_path, "a") as f:
                        for m in all_metrics:
                            f.write(json.dumps(m, ensure_ascii=False) + "\n")

                finally:
                    unload_training_models(policy_model, ref_model)
                    torch.cuda.empty_cache()
                    gc.collect()

                global_step += 1
                if 0 < args.max_steps <= global_step:
                    print(f"\n  Reached --max_steps {args.max_steps}, stopping.")
                    break

            if 0 < args.max_steps <= global_step:
                break

            print(f"\n  Epoch {epoch + 1} done in {time.time() - epoch_start:.0f}s\n")

    finally:
        if pool:
            pool.shutdown()
        if persistent_worker:
            persistent_worker.shutdown()

    report.print()
    print(f"  Done. Logs: {os.path.join(args.output_dir, 'logs')}")


if __name__ == "__main__":
    main()

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

Role boundaries (3 core GRPO roles):
  1. collect_rollouts()    — rollout worker dispatch + result flattening
  2. score_ref_logprobs()  — reference model KL scoring (inside update_actor_*)
  3. update_actor_*()      — policy model GRPO gradient update
  (All three imported from engine_roles.py)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shlex
import shutil
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
from batch_schema import TrainingBatch
from engine_roles import (
    build_advantages_from_metrics,
    check_model_for_nan_inf,
    compute_rewards,
    encode_text_rollouts,
    load_training_models,
    unload_training_models,
    RunReport,
    update_actor_strict,
    update_actor_text,
)

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "isolated_rollout_worker.py")


# ── args ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default="")
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
    p.add_argument("--gpu_id", type=int, default=0,
                   help="[DEPRECATED: use --train_device] GPU for training + fallback for workers")
    p.add_argument("--train_device", type=int, default=None,
                   help="GPU for training (actor/ref). Falls back to --gpu_id")
    p.add_argument("--policy_forward_micro_batch_size", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--checkpoint_every", type=int, default=30)
    # New: worker mode
    p.add_argument("--rollout_worker_mode", default="per_task",
                   choices=["per_task", "persistent", "pool"])
    p.add_argument("--num_rollout_workers", type=int, default=1)
    p.add_argument("--worker_devices", default="",
                   help="[DEPRECATED: use --rollout_worker_devices] Comma-separated GPU IDs for pool workers")
    p.add_argument("--rollout_worker_devices", default=None,
                   help="Comma-separated GPU IDs for rollout workers, e.g. '1,2'. Falls back to --worker_devices")
    p.add_argument("--rollout_backend", default="hf",
                   choices=["hf", "vllm_batched"])
    p.add_argument("--worker_use_singularity", action="store_true")
    p.add_argument("--worker_sif_path", default="")
    p.add_argument("--worker_container_root", default="")
    p.add_argument("--worker_gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--worker_max_model_len", type=int, default=32768)
    p.add_argument("--worker_work_dir", default="")
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


def _find_container_cmd() -> str:
    for cmd in ("singularity", "apptainer"):
        if shutil.which(cmd):
            return cmd
    raise RuntimeError("Neither singularity nor apptainer found")


def _resolve_worker_visible_device(gpu_id: int, parent_visible: str) -> str:
    """Resolve one worker's CUDA_VISIBLE_DEVICES from the parent allocation.

    Cases handled:
    - parent unset: use gpu_id directly
    - parent set to a single device token: reuse that token
    - gpu_id already matches one parent token (physical-id style): keep it
    - gpu_id is a logical index into parent tokens (0/1 inside a 2-GPU srun):
      map it to the corresponding parent token
    """
    if not parent_visible:
        return str(gpu_id)

    tokens = [tok.strip() for tok in parent_visible.split(",") if tok.strip()]
    if not tokens:
        return str(gpu_id)
    if len(tokens) == 1:
        return tokens[0]

    gpu_str = str(gpu_id)
    if gpu_str in tokens:
        return gpu_str

    if 0 <= gpu_id < len(tokens):
        return tokens[gpu_id]

    return gpu_str


def _build_worker_env(gpu_id: int, *, pin_single_visible: bool) -> Dict[str, str]:
    env = os.environ.copy()
    parent_visible = env.get("CUDA_VISIBLE_DEVICES", "")

    if pin_single_visible:
        env["CUDA_VISIBLE_DEVICES"] = _resolve_worker_visible_device(gpu_id, parent_visible)
    elif not parent_visible:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    env["QWEN_OMNI_SKIP_SPK"] = "1"
    return env


def _build_worker_cmd(
    sample: Optional[dict],
    model_path: str,
    adapter_path: str,
    max_rounds: int,
    max_new_tokens: int,
    num_rollouts: int,
    temperature: float,
    finalize_max_new_tokens: int,
    timeout: int,
    rollout_backend: str,
    worker_use_singularity: bool,
    worker_sif_path: str,
    worker_container_root: str,
    worker_gpu_memory_utilization: float,
    worker_max_model_len: int,
    worker_work_dir: str,
    persistent: bool = False,
) -> List[str]:
    worker_args = [
        os.path.abspath(WORKER_SCRIPT),
        "--model_path", model_path,
        "--adapter_path", adapter_path,
        "--rollout_backend", rollout_backend,
        "--max_rounds", str(max_rounds),
        "--max_new_tokens", str(max_new_tokens),
        "--num_generations", str(num_rollouts),
        "--temperature", str(temperature),
        "--finalize_max_new_tokens", str(finalize_max_new_tokens),
        "--timeout", str(timeout - 10),
        "--gpu_memory_utilization", str(worker_gpu_memory_utilization),
        "--max_model_len", str(worker_max_model_len),
    ]
    if worker_work_dir:
        worker_args.extend(["--work_dir", worker_work_dir])
    if persistent:
        worker_args.append("--persistent")
    if sample is not None:
        worker_args.extend(["--sample_json", json.dumps(sample, ensure_ascii=False)])

    if not worker_use_singularity:
        return [sys.executable, "-u", *worker_args]

    if not worker_sif_path or not worker_container_root:
        raise ValueError("worker_use_singularity requires worker_sif_path and worker_container_root")
    container_cmd = _find_container_cmd()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    inner_cmd = " ".join(
        [shlex.quote(os.path.join(worker_container_root, "bin", "python")), "-u"]
        + [shlex.quote(arg) for arg in worker_args]
    )
    return [
        container_cmd, "exec", "--nv",
        "--bind", "/hpai:/hpai",
        "--bind", "/home:/home",
        "--bind", f"{project_root}:{project_root}",
        "--bind", f"{model_path}:{model_path}",
        "--bind", f"{worker_container_root}:{worker_container_root}",
        worker_sif_path,
        "bash", "-lc",
        (
            f"export PATH={shlex.quote(os.path.join(worker_container_root, 'bin'))}:$PATH; "
            f"export PYTHONNOUSERSITE=1; "
            f"export HF_HOME={shlex.quote(os.path.join(project_root, 'output', 'singularity', 'hf_cache'))}; "
            f"export TRANSFORMERS_CACHE={shlex.quote(os.path.join(project_root, 'output', 'singularity', 'hf_cache'))}; "
            f"cd {shlex.quote(project_root)}; "
            f"{inner_cmd}"
        ),
    ]


# ── per-task worker (original) ──

def run_worker(
    sample: dict, model_path: str, adapter_path: str,
    gpu_id: int,
    max_rounds: int, max_new_tokens: int, num_rollouts: int,
    temperature: float, finalize_max_new_tokens: int, timeout: int,
    rollout_backend: str = "hf",
    worker_use_singularity: bool = False,
    worker_sif_path: str = "",
    worker_container_root: str = "",
    worker_gpu_memory_utilization: float = 0.85,
    worker_max_model_len: int = 32768,
    worker_work_dir: str = "",
) -> dict:
    sample_id = sample.get("id", "?")
    cmd = _build_worker_cmd(
        sample=sample,
        model_path=model_path,
        adapter_path=adapter_path,
        max_rounds=max_rounds,
        max_new_tokens=max_new_tokens,
        num_rollouts=num_rollouts,
        temperature=temperature,
        finalize_max_new_tokens=finalize_max_new_tokens,
        timeout=timeout,
        rollout_backend=rollout_backend,
        worker_use_singularity=worker_use_singularity,
        worker_sif_path=worker_sif_path,
        worker_container_root=worker_container_root,
        worker_gpu_memory_utilization=worker_gpu_memory_utilization,
        worker_max_model_len=worker_max_model_len,
        worker_work_dir=worker_work_dir,
        persistent=False,
    )

    # Per-task mode inherits the parent's CUDA_VISIBLE_DEVICES when present.
    env = _build_worker_env(gpu_id, pin_single_visible=False)

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
                 temperature: float, finalize_max_new_tokens: int, timeout: int,
                 rollout_backend: str = "hf",
                 worker_use_singularity: bool = False,
                 worker_sif_path: str = "",
                 worker_container_root: str = "",
                 worker_gpu_memory_utilization: float = 0.85,
                 worker_max_model_len: int = 32768,
                 worker_work_dir: str = ""):
        self.gpu_id = gpu_id
        self.num_rollouts = num_rollouts
        self.restart_count = 0
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.finalize_max_new_tokens = finalize_max_new_tokens
        self.timeout = timeout
        self.rollout_backend = rollout_backend
        self.worker_use_singularity = worker_use_singularity
        self.worker_sif_path = worker_sif_path
        self.worker_container_root = worker_container_root
        self.worker_gpu_memory_utilization = worker_gpu_memory_utilization
        self.worker_max_model_len = worker_max_model_len
        self.worker_work_dir = worker_work_dir

        cmd = _build_worker_cmd(
            sample=None,
            model_path=model_path,
            adapter_path=adapter_path,
            max_rounds=max_rounds,
            max_new_tokens=max_new_tokens,
            num_rollouts=num_rollouts,
            temperature=temperature,
            finalize_max_new_tokens=finalize_max_new_tokens,
            timeout=timeout,
            rollout_backend=rollout_backend,
            worker_use_singularity=worker_use_singularity,
            worker_sif_path=worker_sif_path,
            worker_container_root=worker_container_root,
            worker_gpu_memory_utilization=worker_gpu_memory_utilization,
            worker_max_model_len=worker_max_model_len,
            worker_work_dir=worker_work_dir,
            persistent=True,
        )

        # Persistent/pool mode must pin each worker to one GPU inside the
        # parent SLURM allocation instead of exposing the full device list.
        env = _build_worker_env(gpu_id, pin_single_visible=True)

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
                temperature: float, finalize_max_new_tokens: int, timeout: int,
                **kwargs) -> None:
        """Kill and restart the worker."""
        self.shutdown()
        self.__init__(
            model_path, adapter_path, self.gpu_id,
            max_rounds, max_new_tokens, num_rollouts,
            temperature, finalize_max_new_tokens, timeout,
            rollout_backend=kwargs.get("rollout_backend", self.rollout_backend),
            worker_use_singularity=kwargs.get("worker_use_singularity", self.worker_use_singularity),
            worker_sif_path=kwargs.get("worker_sif_path", self.worker_sif_path),
            worker_container_root=kwargs.get("worker_container_root", self.worker_container_root),
            worker_gpu_memory_utilization=kwargs.get("worker_gpu_memory_utilization", self.worker_gpu_memory_utilization),
            worker_max_model_len=kwargs.get("worker_max_model_len", self.worker_max_model_len),
            worker_work_dir=kwargs.get("worker_work_dir", self.worker_work_dir),
        )
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
                 temperature: float, finalize_max_new_tokens: int, timeout: int,
                 rollout_backend: str = "hf",
                 worker_use_singularity: bool = False,
                 worker_sif_path: str = "",
                 worker_container_root: str = "",
                 worker_gpu_memory_utilization: float = 0.85,
                 worker_max_model_len: int = 32768,
                 worker_work_dir: str = ""):
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
                rollout_backend=rollout_backend,
                worker_use_singularity=worker_use_singularity,
                worker_sif_path=worker_sif_path,
                worker_container_root=worker_container_root,
                worker_gpu_memory_utilization=worker_gpu_memory_utilization,
                worker_max_model_len=worker_max_model_len,
                worker_work_dir=worker_work_dir,
            )
            self.workers.append(wh)

        self._worker_args = (model_path, adapter_path,
                             max_rounds, max_new_tokens, num_rollouts,
                             temperature, finalize_max_new_tokens, timeout)
        self._worker_kwargs = {
            "rollout_backend": rollout_backend,
            "worker_use_singularity": worker_use_singularity,
            "worker_sif_path": worker_sif_path,
            "worker_container_root": worker_container_root,
            "worker_gpu_memory_utilization": worker_gpu_memory_utilization,
            "worker_max_model_len": worker_max_model_len,
            "worker_work_dir": worker_work_dir,
        }

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
                    wh.restart(*self._worker_args, **self._worker_kwargs)
                    self.restart_count += 1
                except Exception as re:
                    print(f"    [pool] Failed to restart worker {worker_idx}: {re}")

        return results

    def shutdown(self) -> None:
        for wh in self.workers:
            wh.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# Role: Collect Rollouts (Phase 1: worker dispatch + Phase 2: flatten)
# ═══════════════════════════════════════════════════════════════════════════════


def collect_rollouts(
    batch: List[dict],
    args: argparse.Namespace,
    worker_devices: List[int],
    report: RunReport,
    batch_idx: int = 0,
    pool: Optional[WorkerPool] = None,
    persistent_worker: Optional[PersistentWorkerHandle] = None,
) -> Tuple[TrainingBatch, RunReport]:
    """Dispatch rollout workers, collect results, flatten into TrainingBatch.

    Handles all three worker modes (per_task / persistent / pool).
    Mutates ``report`` in place with timing and success/failure counters.

    Returns:
        training_batch: TrainingBatch with rollout_data + samples populated
        report:         updated RunReport
    """
    current_batch_size = len(batch)
    print(f"\n  ── Batch {batch_idx}: "
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
            # Debug: print rollout errors when all fail
            if n_ok == 0:
                for r in wres.get("rollouts", []):
                    r_err = r.get("error") or ""
                    if r_err:
                        print(f"    [pool_error] {r_err[:200]}")
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
                    rollout_backend=args.rollout_backend,
                    worker_use_singularity=args.worker_use_singularity,
                    worker_sif_path=args.worker_sif_path,
                    worker_container_root=args.worker_container_root,
                    worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
                    worker_max_model_len=args.worker_max_model_len,
                    worker_work_dir=args.worker_work_dir,
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
                worker_devices[0] if worker_devices else args.gpu_id,
                args.max_rounds, args.max_new_tokens,
                args.num_rollouts,
                args.temperature, args.finalize_max_new_tokens,
                args.worker_timeout,
                rollout_backend=args.rollout_backend,
                worker_use_singularity=args.worker_use_singularity,
                worker_sif_path=args.worker_sif_path,
                worker_container_root=args.worker_container_root,
                worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
                worker_max_model_len=args.worker_max_model_len,
                worker_work_dir=args.worker_work_dir,
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
    # Debug: print first rollout error when all fail
    if n_ok == 0 and n_fail > 0:
        for wr in worker_results:
            for r in wr.get("rollouts", []):
                r_err = r.get("error") or ""
                if r_err:
                    print(f"    [rollout_error] {r_err[:300]}")
                    break
            else:
                continue
            break
    print(f"  Workers done: {total_worker_ok}/{current_batch_size} "
          f"succeeded ({n_ok} rollouts ok, {n_fail} failed)")

    # Flatten results: rollout_data + samples into TrainingBatch
    rollout_data_list: List[dict] = []
    samples_list: List[dict] = []
    for wres, sample in zip(worker_results, batch):
        rollouts = wres.get("rollouts", [])
        while len(rollouts) < args.num_rollouts:
            rollouts.append({
                "final_response": "", "pred_answer": "",
                "total_rounds": 0, "used_segments": [],
                "stop_reason": "error",
            })
        for r in rollouts:
            rollout_data_list.append(r)
            samples_list.append(sample)

    training_batch = TrainingBatch(
        rollout_data=rollout_data_list,
        samples=samples_list,
        num_rollouts=args.num_rollouts,
    )
    return training_batch, report


# ═══════════════════════════════════════════════════════════════════════════════
# Main: GRPO training loop with role-boundary function calls
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    if args.rollout_backend == "hf" and not args.adapter_path:
        raise ValueError("--adapter_path is required for rollout_backend=hf")
    if args.worker_use_singularity:
        if not args.worker_sif_path or not args.worker_container_root:
            raise ValueError(
                "--worker_use_singularity requires --worker_sif_path and --worker_container_root"
            )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # ── Device role resolution ──
    # Priority: --train_device > --gpu_id
    train_device_id = args.train_device if args.train_device is not None else args.gpu_id
    device = torch.device(f"cuda:{train_device_id}" if torch.cuda.is_available() else "cpu")

    # Priority: --rollout_worker_devices > --worker_devices > [train_device_id]
    rollout_dev_str = (
        args.rollout_worker_devices
        if args.rollout_worker_devices is not None
        else args.worker_devices
    )
    worker_devices = (
        [int(x.strip()) for x in rollout_dev_str.split(",") if x.strip()]
        if rollout_dev_str
        else [train_device_id]
    )

    print(f"\n{'='*60}")
    print(f"  Device Roles:")
    print(f"    Training (actor/ref):  GPU {train_device_id}")
    print(f"    Rollout workers:       GPUs {worker_devices}")
    print(f"    Worker mode:           {args.rollout_worker_mode}")
    if args.rollout_worker_mode in ("pool", "persistent"):
        print(f"    Num workers:           {args.num_rollout_workers}")
    print(f"{'='*60}\n")

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
            rollout_backend=args.rollout_backend,
            worker_use_singularity=args.worker_use_singularity,
            worker_sif_path=args.worker_sif_path,
            worker_container_root=args.worker_container_root,
            worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
            worker_max_model_len=args.worker_max_model_len,
            worker_work_dir=args.worker_work_dir,
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
            rollout_backend=args.rollout_backend,
            worker_use_singularity=args.worker_use_singularity,
            worker_sif_path=args.worker_sif_path,
            worker_container_root=args.worker_container_root,
            worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
            worker_max_model_len=args.worker_max_model_len,
            worker_work_dir=args.worker_work_dir,
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

                # ═══════════════════════════════════════════════════════════
                # Phase 1+2: Collect Rollouts → TrainingBatch
                # ═══════════════════════════════════════════════════════════
                batch, report = collect_rollouts(
                    batch, args, worker_devices, report,
                    batch_idx=batch_idx // args.batch_size,
                    pool=pool, persistent_worker=persistent_worker,
                )

                # ═══════════════════════════════════════════════════════════
                # Phase 2.5: Free GPU memory for training
                # ═══════════════════════════════════════════════════════════
                had_workers = bool(persistent_worker or pool)
                if persistent_worker:
                    print(f"  [persistent] shutting down worker before training ...")
                    persistent_worker.shutdown()
                    persistent_worker = None
                if pool:
                    print(f"  [pool] shutting down workers before training ...")
                    pool.shutdown()
                    pool = None
                if had_workers:
                    import time as _time
                    _time.sleep(3)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # ═══════════════════════════════════════════════════════════
                # Phase 3: Load Training Models
                # ═══════════════════════════════════════════════════════════
                print(f"\n  [{datetime.now()}] Loading training models ...")
                t_load = time.time()
                policy_model, ref_model, processor = load_training_models(
                    args.model_path, args.adapter_path, device,
                    disable_talker=True,
                )
                tokenizer = processor.tokenizer
                optimizer = torch.optim.AdamW(
                    policy_model.parameters(), lr=args.learning_rate,
                    eps=1e-4,
                )
                trainable = sum(
                    p.numel() for p in policy_model.parameters() if p.requires_grad
                )
                print(f"  Models loaded in {time.time() - t_load:.1f}s "
                      f"(trainable: {trainable:,})")

                try:
                    # ═══════════════════════════════════════════════════════
                    # Phase 4a: Compute Rewards → batch.metrics
                    # ═══════════════════════════════════════════════════════
                    batch = compute_rewards(batch, global_step)

                    # ═══════════════════════════════════════════════════════
                    # Phase 4b: Compute Advantages → batch.advantages
                    # ═══════════════════════════════════════════════════════
                    batch = build_advantages_from_metrics(batch, device)

                    # ═══════════════════════════════════════════════════════
                    # Phase 5: GRPO Policy Update
                    # ═══════════════════════════════════════════════════════
                    weights_healthy = True

                    if args.grpo_forward_mode == "strict_interleaved":
                        # ── Strict Interleaved Forward (multimodal) ──
                        agg_loss, agg_kl, grad_norm, weights_healthy = (
                            update_actor_strict(
                                policy_model, ref_model, processor,
                                optimizer, batch, args, report, device,
                            )
                        )
                    else:
                        # ── Text-only Forward ──
                        batch = encode_text_rollouts(batch, tokenizer)
                        agg_loss, agg_kl, grad_norm, weights_healthy = (
                            update_actor_text(
                                policy_model, ref_model, optimizer,
                                batch, args, device,
                            )
                        )

                    # ═══════════════════════════════════════════════════════
                    # Phase 6: Logging & Checkpoint
                    # ═══════════════════════════════════════════════════════
                    n = len(batch.metrics)
                    avg_rt = sum(m["rollout_total"] for m in batch.metrics) / max(n, 1)
                    avg_fmt = sum(m["format"] for m in batch.metrics) / max(n, 1)
                    avg_cst = sum(m["consistency"] for m in batch.metrics) / max(n, 1)
                    avg_acc = sum(m["accuracy"] for m in batch.metrics) / max(n, 1)
                    avg_seg = sum(m["segment"] for m in batch.metrics) / max(n, 1)
                    n_correct = sum(m["is_correct"] for m in batch.metrics)

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
                    writer.add_scalar(
                        "train/grad_norm",
                        grad_norm.item() if hasattr(grad_norm, 'item') else 0.0,
                        global_step,
                    )
                    for key in ("rollout_total", "format", "consistency",
                                "accuracy", "segment"):
                        vals = [m[key] for m in batch.metrics]
                        writer.add_scalar(
                            f"reward/{key}",
                            sum(vals) / max(len(vals), 1), global_step,
                        )

                    if global_step > 0 and global_step % args.checkpoint_every == 0:
                        if not weights_healthy:
                            print(f"  -> WARNING: nan/inf detected in weights, "
                                  f"SKIPPING checkpoint save")
                        elif not check_model_for_nan_inf(
                            policy_model, "checkpoint_pre_save",
                        ):
                            print(f"  -> WARNING: nan/inf detected in weights, "
                                  f"SKIPPING checkpoint save")
                        else:
                            ckpt_dir = os.path.join(
                                args.output_dir, "checkpoints",
                                f"step_{global_step}",
                            )
                            policy_model.save_pretrained(ckpt_dir)
                            print(f"  -> Checkpoint: {ckpt_dir}")

                    log_path = os.path.join(
                        args.output_dir, "logs", "rollouts.jsonl",
                    )
                    with open(log_path, "a") as f:
                        for m in batch.metrics:
                            f.write(json.dumps(m, ensure_ascii=False) + "\n")

                finally:
                    unload_training_models(policy_model, ref_model)
                    torch.cuda.empty_cache()
                    gc.collect()

                # ═══════════════════════════════════════════════════════════
                # Phase 7: Restart Workers for Next Batch
                # ═══════════════════════════════════════════════════════════
                if args.rollout_worker_mode == "persistent":
                    print(f"  [persistent] restarting worker for next batch ...")
                    persistent_worker = PersistentWorkerHandle(
                        args.model_path, args.adapter_path,
                        worker_devices[0],
                        args.max_rounds, args.max_new_tokens, args.num_rollouts,
                        args.temperature, args.finalize_max_new_tokens,
                        args.worker_timeout,
                        rollout_backend=args.rollout_backend,
                        worker_use_singularity=args.worker_use_singularity,
                        worker_sif_path=args.worker_sif_path,
                        worker_container_root=args.worker_container_root,
                        worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
                        worker_max_model_len=args.worker_max_model_len,
                        worker_work_dir=args.worker_work_dir,
                    )
                elif args.rollout_worker_mode == "pool":
                    print(f"  [pool] restarting workers for next batch ...")
                    pool = WorkerPool(
                        args.model_path, args.adapter_path,
                        args.num_rollout_workers, worker_devices,
                        args.max_rounds, args.max_new_tokens, args.num_rollouts,
                        args.temperature, args.finalize_max_new_tokens,
                        args.worker_timeout,
                        rollout_backend=args.rollout_backend,
                        worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
                        worker_max_model_len=args.worker_max_model_len,
                        worker_work_dir=args.worker_work_dir,
                    )
                    print(f"    [pool] {args.num_rollout_workers} workers restarted "
                          f"on devices {worker_devices}")

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

# Echo Rollout Controller Integration

日期: 2026-05-14

## 目标

把仓库里已经存在的 Echo interleaved 推理实现整合成一个统一的 rollout controller，直接服务后续 rollout server / custom GRPO 接入，而不是重写一套新的 Echo 逻辑。

新增入口：

- `scripts/rl_rollout/echo_interleaved_rollout_controller.py`
- `script/test7_vllm_interleaved_controller_from_existing.sh`
- `script/test8_vllm_batched_interleaved_rollout.sh`

## 复用来源

### 1. 来自作者 `inference/inference_multiturn.py`

复用的核心思路：

- vLLM `LLM.generate(...)` 多轮推理
- 每轮检测 `</seg>`
- 从原始 full audio 裁剪新片段
- 把新片段 append 到 `multi_modal_data["audio"]`
- 把对应 audio placeholder append 回 prompt

这部分决定了新 controller 的 **prompt/audio 状态更新方式**：

- 不重新设计一套 conversation schema
- 直接沿用作者的“prompt 逐轮追加、audio 逐轮追加”的思路
- 在拿到第一个 `seg` 之后，**继续在同一条 prompt 上追加文本与 audio placeholder**
- 不走“新开一个 user message / assistant message 再继续推理”的旧实验路径

### 2. 来自 `scripts/interleaved_infer.py` 与 `scripts/03_interleaved`

复用的逻辑：

- `build_initial_prompt`
- `build_finalize_prompt`
- `parse_segments`
- `clamp_seg`
- `save_segment_audio`
- `extract_answer`
- `has_answer`
- `is_duplicate_seg`

策略层直接沿用之前验证过的行为：

- duplicate guard
- `on_duplicate_seg`
- `finalize_on_stop`
- `stop + finalize` 仍是当前默认主线

对应结论见：

- `tasks/05_interleaved_eval_results.md`
- `tasks/07_strategy_comparison.md`

### 3. 来自 `echo_rl/rewards.py` / `echo_rl/rollout_rewards.py`

reward 需要的稳定字段是：

- 原始 `response`
- `<answer>` 抽取结果
- `<seg>` 抽取结果

因此新 controller 的每条 rollout 返回里保留：

- `final_response`
- `model_prediction`
- `segments`
- `unique_segments`

并附加：

- `reward_ready.response`
- `reward_ready.answer`
- `reward_ready.segments`

这样后续 GRPO 不需要重新解析另一套自定义格式。

## 新 controller 结构

### `EchoRolloutState`

每条 rollout 一个独立 state，字段包括：

- `request_id`
- `sample_id`
- `rollout_id`
- `audio_path`
- `question`
- `choices`
- `prompt`
- `audio_inputs`
- `full_response`
- `segments`
- `unique_segments`
- `rounds`
- `finish_reason`
- `finalized`
- `error`

内部还维护：

- full audio
- sample rate
- duration
- 当前 phase（`interleaved` / `finalize` / `done` / `error`）
- 当前 round index

### `EchoVLLMBatchedRolloutController`

提供：

- `run_batch(requests: list[dict]) -> list[dict]`
- `run_one(request) -> dict`

并发方式：

- 每个 request 可展开成多个 rollout state
- 每轮收集全部 active states
- 按 phase / sampling 参数分组
- 对每组一次 batched `vllm.generate`
- 再把结果分发回各自 state

这比把 `n=8` 丢给 `SamplingParams(n=8)` 更适合 rollout/reward：

- 每条 rollout 都有独立 `request_id`
- segment 和 reward 字段天然一一对应
- 某条 rollout 提前结束不会阻塞其他 rollout

## 推理 continuation 逻辑

这里再强调一次当前主线的推理逻辑来源：

- 对齐的是 `inference/inference_multiturn.py`
- 不是早期某些 HF 实验里“新开 user/assistant 轮次”的写法

当前 controller 的 interleaved 主链路是：

1. 建立初始 prompt
2. 生成到 `</seg>`
3. 裁剪音频片段
4. 把新的 audio placeholder 直接 append 回当前 prompt
5. 把对应裁剪音频 append 到 `multi_modal_data["audio"]`
6. 在同一条生成链上继续 `vllm.generate`

只有在 `duplicate_seg` / `max_rounds` 且还没 answer 时，才会走额外的 finalize 分支。

因此：

- interleaved continuation 本体：**同一条 prompt 原地续写**
- finalize：**兜底收尾**

## 返回 schema

每条 rollout 返回一个 dict：

```json
{
  "request_id": "...",
  "sample_id": "...",
  "rollout_id": 0,
  "final_response": "...",
  "model_prediction": "...",
  "segments": [...],
  "unique_segments": [...],
  "rounds": [...],
  "finish_reason": "...",
  "finalized": true,
  "error": null,
  "reward_ready": {
    "response": "...",
    "answer": "...",
    "segments": [[0.1, 0.8]]
  }
}
```

## 当前边界

当前这一步只做：

- batched vLLM interleaved rollout controller
- 单样本 smoke
- 2 样本 × 2 rollout batched smoke

当前还没有做：

- VERL 接入
- 训练
- server 化请求协议
- custom GRPO 采样编排

## 实测结果

### Test 7: 单样本 controller smoke

环境：

- Singularity `nvidia_cuda_12.4.sif`
- Ubuntu 22.04 / `glibc 2.35`
- `torch 2.6.0+cu124`
- `vllm 0.8.5`

结果：

- 脚本 `script/test7_vllm_interleaved_controller_from_existing.sh` 跑通
- controller 在容器内成功 `from vllm import LLM` 并完成模型初始化
- 单条 request → 单条 rollout 正常执行
- 输出 JSON 包含：
  - `final_response`
  - `model_prediction`
  - `segments`
  - `unique_segments`
  - `rounds`
  - `finish_reason`
  - `finalized`
  - `error`
  - `reward_ready`

结论：

- 单样本 controller 在已打通的 Singularity + `vllm 0.8.5` 路线上可用

### Test 8: batched controller smoke

配置：

- `2` 个样本
- 每样本 `2` 个 rollout
- 总计 `4` 个 rollout result

结果：

- 脚本 `script/test8_vllm_batched_interleaved_rollout.sh` 跑通
- controller 能从 `eval_manifest_500.jsonl` 读取样本并展开 request
- `results` 中返回 `4` 个 dict
- 每个 dict 对应一条独立 rollout
- `request_id` 使用 `::r0 / ::r1` 正确区分 rollout
- `error: null × 4`
- `finish_reason` 均为有效结束状态

结论：

- batched controller 在当前容器 vLLM 路线上可用
- batch 内 rollout 可以独立输出，不会互相阻塞

### Test 9: isolated worker + vllm_batched

入口：

- `script/test9_worker_vllm_batched_smoke.sh`
- `scripts/rl/isolated_rollout_worker.py`

环境：

- Singularity `nvidia_cuda_12.4.sif`
- Ubuntu 22.04 / `glibc 2.35`
- `vllm 0.8.5`

结果：

- worker 在容器内成功启动
- `vllm_batched` backend 成功加载 `vllm.LLM` / Qwen2.5-Omni
- 单样本 worker JSON 成功返回
- 产物：
  - `output/rl_rollout/test9_worker_vllm_batched.json`

返回的兼容字段已经齐全：

- `final_response`
- `pred_answer`
- `total_rounds`
- `used_segments`
- `used_segment_paths`
- `stop_reason`
- `round_outputs`
- `triggered_interleaved`
- `has_final_answer`
- `answer_correct`

同时也保留了新链路字段：

- `request_id`
- `sample_id`
- `rollout_id`
- `finalized`
- `error`
- `reward_ready`

结论：

- `isolated_rollout_worker.py + vllm_batched + Singularity` 可用

### Test 10: rollout_smoke_test + vllm_batched worker

入口：

- `script/test10_smoke_test_vllm_batched.sh`
- `scripts/rl/rollout_smoke_test.py`

结果：

- `rollout_smoke_test.py` 成功调起新 worker
- 成功完成：
  - rollout
  - reward
  - GRPO forward smoke
- 说明新 worker 不只是能返回 JSON，而且已经能被现有 smoke 训练脚手架消费

关键观测：

- `rollout_success_count: 1`
- `rollout_failed_count: 0`
- reward 阶段正常执行

结论：

- `rollout_smoke_test.py + vllm_batched + Singularity` 可用

### Test 10 过程中暴露并修复的小 bug

初版失败点：

- `RuntimeError: CUDA driver error: invalid argument`
- 深层表现为 `vLLM V1 engine` 初始化失败

根因：

- `scripts/rl/rollout_smoke_test.py` 中 `run_worker()` 原本强制设置
  `env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)`
- 当主进程已经在 SLURM 环境下初始化过 CUDA（例如已 import `torch`）时，
  再对子进程强行覆盖 `CUDA_VISIBLE_DEVICES`，会让 vLLM V1 engine fork 出来的 core process 遇到 CUDA 参数冲突

修复：

- 改为优先继承 SLURM / 父进程已有的 `CUDA_VISIBLE_DEVICES`
- 只有当父进程未设置时，才回退到 `gpu_id`

这条修复已经在当前代码中落地。

## 错误隔离说明

当前代码里，`_expand_requests()` 已经对单条 request 的音频加载做了 `try/except`：

- 如果某条 request 的音频加载失败，该条 state 会被标记为：
  - `error = "audio_load_error: ..."`
  - `phase = "error"`
  - `finish_reason = "error"`
- 其他 request 仍可继续进入后续 batch generate

因此按当前实现，**单个坏音频不会在 expand 阶段直接拖死整个 batch**。

## 下一步

下一步更自然的两条路：

1. 基于这份 batched controller 做 server 化包装
2. 直接把 controller 接到 custom GRPO rollout worker

## 已完成的 worker 接入

为了尽量少改上层训练脚手架，这次优先接的是现有 subprocess rollout worker：

- `scripts/rl/isolated_rollout_worker.py`
- `scripts/rl/rollout_smoke_test.py`

做法：

- 保留原有 `hf` rollout backend
- 新增 `vllm_batched` rollout backend
- 当 worker 使用 `vllm_batched` 时：
  - 内部加载 `vllm.LLM`
  - 调用 `EchoVLLMBatchedRolloutController`
  - 将新 controller 的输出映射回旧 worker 期望的字段名

这意味着当前代码状态是：

- rollout controller：已完成并 smoke 通过
- worker 兼容层：已完成并通过静态检查
- worker + Singularity + `vllm_batched` 的完整端到端链路：**已完成最小实跑并通过**

兼容回写的关键字段：

- `final_response`
- `pred_answer`
- `total_rounds`
- `used_segments`
- `used_segment_paths`
- `stop_reason`
- `round_outputs`
- `triggered_interleaved`
- `has_final_answer`
- `answer_correct`

这样 `build_rollout_metadata()` 和现有 reward / smoke 训练脚本不需要立即重写。

### Singularity worker 启动支持

`scripts/rl/rollout_smoke_test.py` 新增了 worker 启动参数：

- `--rollout_backend {hf,vllm_batched}`
- `--worker_use_singularity`
- `--worker_sif_path`
- `--worker_container_root`
- `--worker_gpu_memory_utilization`
- `--worker_max_model_len`
- `--worker_work_dir`

用途：

- 允许现有 worker 模式（`per_task` / `persistent` / `pool`）在 Singularity 容器里拉起 `vllm_batched` worker
- 保持主训练脚本对 worker 结果结构的预期不变

注意：

- 这一步是 **代码接入完成**
- `isolated_rollout_worker.py + vllm_batched + Singularity` 已通过
- `rollout_smoke_test.py + vllm_batched + Singularity` 已通过
- 下一步不再是“验证 worker 能不能接通”，而是决定：
  - 先接更完整的 custom GRPO 运行
  - 还是先做 server 化包装

容器环境建议沿用已经打通的：

- Singularity / Ubuntu 22.04 / glibc 2.35
- `torch 2.6.0+cu124`
- `vllm 0.8.5`
- `gpu_memory_utilization=0.85`

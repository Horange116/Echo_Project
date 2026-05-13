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
- custom rollout worker / GRPO smoke 接线

当前还没有做：

- VERL 接入
- 训练
- server 化请求协议
- 更大规模训练放大

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

## Test 11: scale20 custom GRPO smoke

入口：

- `script/test11_vllm_batched_grpo_scale20.sh`

默认配置：

- `MAX_SAMPLES=20`
- `NUM_ROLLOUTS=2`
- `MAX_STEPS=3`
- `MAX_ROUNDS=2`
- `MAX_TOKENS=96`
- `TEMPERATURE=1.0`
- `FINALIZE_MAX_TOKENS=64`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MAX_MODEL_LEN=32768`

输出目录：

- `output/rl_rollout/test11_scale20/`
- `output/grpo_vllm_batched_scale20/`

结果：

- `test11` 全部通过
- 在 A800Z 单卡 + Singularity 路线上稳定跑完
- 成功进入 `rollout_smoke_test.py`
- 成功启动 `vllm_batched` worker
- 完成 `step 0 / 1 / 2`，共 `3` 个 step
- `3 steps × 2 rollouts = 6` 条 rollout 全部成功
- `rollout_success_count=6`
- `rollout_failed_count=0`
- reward 统计正常：
  - `mean=0.583`
  - `min=0.5`
  - `max=1.0`
- 总耗时 `584` 秒，约 `9.7` 分钟
- 未见 CUDA 异常、worker 启动异常、reward 崩溃、GRPO text-only forward 异常

关键产物：

- `output/rl_rollout/test11_scale20/test11_stdout.log`
- `output/rl_rollout/test11_scale20/test11_stderr.log`
- `output/rl_rollout/test11_scale20/test11_summary.json`
- `output/grpo_vllm_batched_scale20/logs/rollouts.jsonl`

结论：

- `vllm_batched` rollout 已经不只是“单点 smoke 可用”
- 在小规模放大版 custom GRPO smoke 下也已验证稳定
- 当前阶段已经从“链路打通”进入“可小规模稳定运行”

## 当前阶段结论

截至 `test11`，当前主线已经完成：

- controller 单样本 smoke
- controller batched smoke
- isolated worker 接通
- rollout + reward + GRPO forward 最小 smoke
- scale20 小规模放大 smoke

因此当前更合理的下一步不再是证明“vLLM 能不能推理”，而是：

1. 继续做更大规模稳定性 / 吞吐验证
2. 评估 `pool` / 多 GPU worker 分摊
3. 再往后一层推进更完整的 custom GRPO 训练放大

## 当前瓶颈判断

基于 `test11` 的运行行为，当前实现里最明显的固定成本来自：

- `per_task` 模式下反复拉起 worker 子进程
- 每次 worker 都重新初始化 `vllm.LLM`
- vLLM engine / compile / warmup 的固定开销较大

因此：

- `persistent` 模式的主要价值是**摊薄单卡反复加载成本**
- 这条路线大概率能进一步缩短单卡总时长

但当前优先级不应先转去打磨 `persistent`，原因是：

- `persistent` 更偏向**单卡优化**
- 当前更关键的问题是：**多卡多 worker rollout 的并行吞吐形态是否稳定**

所以顺序上应当是：

1. 先验证 `pool + worker_devices=0,1,...` 的多卡多 worker 方案
2. 如果 `pool` 稳定，再回头比较：
   - `per_task`
   - `persistent`
   - `pool`
   三种模式的总耗时 / 吞吐 / 稳定性

一句话：

- `per_task` 的瓶颈判断基本成立
- 但 `persistent` 当前属于**后续优化项**
- `pool` 才是更贴近论文多卡 rollout 形态的优先验证方向

## Test 12: 2-card pool multi-worker rollout smoke

入口：

- `script/test12_vllm_batched_pool2_scale20.sh`

目标：

- 验证 `rollout_worker_mode=pool`
- 验证 `2` 张 GPU + `2` 个 persistent workers
- 验证多卡 worker 分摊下的 rollout / reward / text-only GRPO 训练链路

实际运行：

- `srun -p A800Z --gres=gpu:2 --time=10:00 bash script/test12_vllm_batched_pool2_scale20.sh`

结果：

- pool 模式已成功启用
- 日志确认：
  - `[pool] 2 workers on devices [0, 1]`
  - Batch 0 / Batch 1 的 rollout 都成功完成
  - shutdown-before-training / restart-after-training 逻辑已正常工作
- 已完成：
  - `step 0` 完整训练
  - `step 1` 的 rollout 阶段
- 关键统计：
  - `rollout_success_count=8`
  - `rollout_failed_count=0`
  - Step 0 reward: `mean=0.600`
  - Step 1 rollout 也已成功写入 `rollouts.jsonl`
- batch rollout 耗时：
  - Batch 0: `232.8s`
  - Batch 1: `218.3s`

阻塞点：

- 当前失败不是 pool 代码逻辑失败
- 真正阻塞是 `srun --time=10:00` 被 SLURM QOS 强制终止
- 终止时机发生在：
  - Batch 1 rollout 完成之后
  - 进入 Batch 1 training model load 阶段时

补充修复：

- `scripts/rl/rollout_smoke_test.py`
  - 修复了 pool shutdown-before-training 的语法/流程问题
  - 新增了 pool restart-after-training 逻辑

结论：

- `test12` 可以视为**功能链路部分通过**
- 当前已经证明：
  - `2` 卡 pool 多 worker rollout 形态可运行
  - persistent worker 的 shutdown / restart 逻辑可运行
  - 多卡 worker 分摊下没有出现 CUDA 异常 / worker 通信崩溃
- 但还**没有在足够长时限下完整跑完 3 steps**
- 下一步应优先：
  - 在更长 `srun --time` 下重跑 `test12`
  - 先拿到完整 3-step 通过结论，再进入下一阶段

## Test 13: strict_interleaved minimum smoke

入口：

- `script/test13_vllm_batched_strict_min.sh`

目标：

- 沿用已打通的 `Singularity + vllm_batched` rollout 路线
- 不改 controller 主逻辑
- 只把训练 forward 从 `text_only` 切到 `strict_interleaved`
- 做最小 smoke：`4` 样本、`2` rollout、`1` step

正确运行方式：

- `srun -p A800Z --gres=gpu:1 --time=10:00 bash script/test13_vllm_batched_strict_min.sh`

结果：

- `test13` 全部通过
- 成功进入 `rollout_smoke_test.py`
- 成功进入 `strict_interleaved` forward
- 成功完成 `step 0`
- 关键统计：
  - `strict_forward_success=4`
  - `strict_forward_failed=0`
  - `rollout_success_count=4`
  - `rollout_failed_count=0`
  - `peak_memory_mb=37198`
- step 日志：
  - `step   0 | loss nan | R +0.438 | correct 0/4 | 338.1s | fw=stri wk=per_`

关键产物：

- `output/rl_rollout/test13_strict_min/test13_stdout.log`
- `output/rl_rollout/test13_strict_min/test13_stderr.log`
- `output/rl_rollout/test13_strict_min/test13_summary.json`
- `output/grpo_vllm_batched_strict_min/logs/rollouts.jsonl`

结论：

- `strict_interleaved` 已不再只是“代码里存在的 experimental 分支”
- 至少在最小 smoke 条件下，已验证：
  - rollout 正常
  - multimodal strict forward 正常
  - strict 输入重建没有失败
  - strict logprob 路径没有直接崩溃

注意：

- 当前 `loss nan` 仍然值得后续继续盯
- 但就“strict_interleaved 最小链路能否跑通”这个问题，当前答案已经是 **能跑通**

## Test 14: strict_interleaved small-scale smoke

入口：

- `script/test14_vllm_batched_strict_scale.sh`

目标：

- 在 `test13` 通过的基础上，小幅放大 strict 路径
- 验证 `loss nan` 是否稳定复现
- 判断问题更像出在 strict 输入重建，还是 GRPO 数值链路

实际运行：

- `srun -p A800Z --gres=gpu:1 --time=10:00 bash script/test14_vllm_batched_strict_scale.sh`

结果：

- 成功进入 `rollout_smoke_test.py`
- 成功进入 `strict_interleaved` forward
- Batch 0 的 strict forward 全部成功：
  - `strict_forward_success=4`
  - `strict_forward_failed=0`
- rollout 成功：
  - `rollout_success_count=8`
  - `rollout_failed_count=0`
- Step 0 reward 正常：
  - `mean=0.812`
  - `min=0.500`
  - `max=1.000`
  - `correct=3/4`
- 但 step 0 仍出现：
  - `loss nan`
  - `KL nan`
- Batch 1 rollout 完成后，作业在第二步 training model load 阶段被 `srun` 时间限制终止

关键观测：

- stderr 中 strict input 重建正常：
  - `input_ids=[1, 349~354]`
  - `masked_text_tokens=82~87`
- 无 strict input 重建失败
- 无 CUDA 错误
- 无 Python traceback

结论：

- `loss nan` 在 strict 路径下稳定复现
- 而且它与 `text_only` 路径中的现象一致：`loss nan + KL nan`
- 因此当前问题**不像是 strict_interleaved 特有 bug**
- 更像是当前 GRPO 实现中的：
  - `policy/ref logprob`
  - `KL`
  - 或其上游数值稳定性
  出现了 `inf/-inf/nan`

当前判断优先级：

1. `multimodal logprob / KL` 数值问题
2. `loss_mask` / token 边界问题（次优先）
3. `reward / advantage` 本身（优先级更低，因为 reward 值正常）

## 更新后的下一步

当前顺序调整为：

1. `pool` 多卡 worker 形态已经完成核心验证，但完整 3-step 结论仍缺
2. `strict_interleaved` 最小 smoke 与小规模放大 smoke 都已通过 rollout/forward 层
3. 下一步不再继续单纯放大，而应转为：
   - 数值诊断 `loss nan`
   - 重点检查 `policy_logps / ref_logps / KL / loss_mask / advantages`
4. 在数值问题定位清楚后，再决定是否继续放大 strict 规模

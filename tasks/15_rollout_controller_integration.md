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

## Test 16: thinker forward nan diagnosis

入口：

- `script/test16_thinker_nan_diag.sh`

目标：

- 继续追踪 `loss nan`
- 不再停留在 `policy_logps / ref_logps / KL`
- 直接检查 strict forward 的输入和 `thinker(...)` 前向输出
- 判断问题是否真的来自多模态音频特征分支

结果：

- `test16` 成功定位了更早的 nan 起点
- `input_features` 完全正常：
  - `shape=(1, 128, 30000)`
  - `dtype=torch.float32`
  - `min=-0.49947`
  - `max=1.50053`
  - `mean=-0.458107`
  - `nan=False`
  - `inf=False`
- `feature_attention_mask` 也正常：
  - `shape=(1, 30000)`
  - `sum=1000`
  - `min=0`
  - `max=1`
  - `nan=False`
- 但 `thinker(...)` 前向输出的 `logits` 已经全为 `nan`：
  - `shape=(1, 348, 152064)`
  - `min=nan`
  - `max=nan`
  - `mean=nan`
  - `nan=True`
- `log_softmax` 的 `nan` 只是进一步传播

最关键对照：

- 对同一组 `input_ids / attention_mask`
- 去掉 `input_features / feature_attention_mask`
- text-only control forward 结果仍然全为 `nan`

结论：

- 当前问题**不在多模态音频分支**
- 也不在 strict 输入重建
- `nan` 最早出现在 `thinker(**kwargs)` 的前向输出中
- 并且即使去掉音频特征，text-only 对照仍然 `nan`
- 所以当前最像的问题是：
  - 模型本体（+LoRA adapter）下的 `thinker` 前向本身异常

## 最新判断

到 `test16` 为止，问题已经从：

- `GRPO loss / KL / advantage`

进一步收敛成：

- **`thinker` 前向输出 logits 已经为 `nan`**

因此下一步不该再优先查 reward / advantage / mask，而应转为：

1. 检查 `policy_model / ref_model` 加载后的 `thinker` 权重是否正常
2. 检查是否是 `LoRA adapter` 合入后导致 `thinker` 前向失稳
3. 做最小 text-only thinker 单步前向对照（不经过 rollout / GRPO）

## Test 17: base vs LoRA thinker forward diagnosis

入口：

- `script/test17_thinker_base_vs_lora.sh`
- `scripts/rl/thinker_forward_diag.py`

目标：

- 不再经过 rollout / reward / GRPO
- 直接切开 `base model`、`policy LoRA`、`ref LoRA`
- 判断到底是谁先让 `thinker` 前向输出 `nan`

结果：

- `base model` 的 text-only thinker forward 完全正常：
  - `logits shape=(1, 9, 152064)`
  - `min=-15.95`
  - `max=29.14`
  - `mean=-2.14`
  - `nan=False`
  - `inf=False`
- `policy model`（加载当前训练 LoRA）前向立即异常：
  - `logits nan=True`
- `ref model`（加载同一 checkpoint 的 LoRA）同样立即异常：
  - `logits nan=True`

权重扫描：

- `base model` 权重：
  - `0` 个参数含 `nan`
  - `0` 个参数含 `inf`
- `policy model` 权重：
  - `390` 个参数含 `nan`
  - `392` 个参数含 `inf`
- `ref model` 权重：
  - `390` 个参数含 `nan`
  - `392` 个参数含 `inf`
- 第一处异常位置：
  - `base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.default.weight`

结论：

- 当前问题**不是训练代码把前向算坏**
- 也不是 strict 多模态分支把前向算坏
- 而是：
  - **LoRA adapter checkpoint `output/grpo_smoke/checkpoints/final` 本身已经损坏**
  - checkpoint 中已经保存了 `nan/inf` 权重
  - 因此一旦加载 policy/ref adapter，任何 thinker forward 都会直接输出 `nan logits`

## 最新判断

到 `test17` 为止，责任边界已经切清楚：

1. `base model` 本体正常
2. `LoRA adapter checkpoint` 已损坏
3. `policy/ref thinker forward nan` 是损坏权重的直接结果

因此下一步不该继续排查：

- reward
- advantage
- KL
- strict input
- audio features

而应改成：

1. 回溯并确认**第一个写坏 adapter 的训练 checkpoint**
2. 停止复用 `output/grpo_smoke/checkpoints/final`
3. 改为：
   - 从干净 adapter 重新开始
   - 或在训练保存前加入 `nan/inf` 权重保护
4. 再重新跑最小 thinker forward / strict smoke 验证

## Test 18: bad checkpoint backtrace and save guard

目标：

- 回溯 `grpo_smoke/checkpoints` 中哪个保存点第一次写坏了 LoRA
- 在当前 custom GRPO 训练路径里加入最小 `nan/inf` 防护

回溯结果：

对 `output/grpo_smoke/checkpoints/` 的可用保存点做了全量扫描：

| Checkpoint | Size | nan params | inf params | Status |
|---|---:|---:|---:|---|
| `step_5` | 77.1 MB | 0 | 0 | CLEAN |
| `step_10` | 77.1 MB | 0 | 0 | CLEAN |
| `final` | 38.6 MB | 1,871,846 | 18,312,005 | CORRUPTED |

结论：

- 损坏首次出现在 `final`
- `step_5` 与 `step_10` 都是干净的
- 说明在 `step_10 -> final` 之间的训练步骤中：
  - LoRA 权重被 `nan/inf` 梯度或更新污染
  - 且当时没有保存前防护
  - 最终把坏权重写进了 `final`

额外结论：

- 扫描覆盖了 `v0-v10` 的早期与最终 checkpoint
- 只有 `output/grpo_smoke/checkpoints/final` 这个 checkpoint 损坏

新增防护：

在 `scripts/rl/rollout_smoke_test.py` 中加入了最小保护：

1. `check_model_for_nan_inf()` 工具函数
   - 扫描模型可训练参数
   - 检测 `nan/inf`
   - 支持 `fatal=True` 时直接抛异常

2. strict 路径 `loss` 防护
   - 在 `.backward()` 前检查 `loss.isnan()/isinf()`
   - 若异常则打印：
     - `[nan-inf] strict loss=nan, skipping backward`
   - 并跳过该 rollout 的反向传播

3. `optimizer.step()` 后权重检查
   - 两条训练路径都在 step 后立刻扫描 LoRA 权重
   - 一旦发现 `nan/inf`，将 `weights_healthy = False`

4. 保存前防护
   - `save_pretrained()` 前双重检查：
     - `weights_healthy`
     - 再次重新扫描当前权重
   - 若发现损坏则：
     - 跳过保存
     - 打印：
       - `WARNING: nan/inf detected in weights, SKIPPING checkpoint save`

## 最新判断

到 `test18` 为止，问题链已经完整闭环：

1. `base model` 正常
2. `final` LoRA checkpoint 损坏
3. 损坏发生在 `step_10 -> final` 之间
4. 当前训练代码已加入“反向传播前 + step 后 + 保存前”的最小保护

因此下一步主线应改成：

1. 从干净 checkpoint（如 `step_10`）继续，而不是复用 `final`
2. 重新跑最小 thinker forward 验证 adapter 仍干净
3. 再重新跑 strict 最小 smoke
4. 若保护再次触发，则继续回溯具体是哪一个 step 先把权重写坏

## Test 19 / Test 20: step_10 recovery validation

入口：

- `script/test19_step10_recover_thinker.sh`
- `script/test20_step10_strict_recover.sh`

目标：

- 确认 `step_10` 是否真的是干净可继续的起点
- 验证 strict 最小 smoke 在干净 LoRA 下是否恢复正常
- 观察当前新增的训练保护会在什么层级触发

### Test 19: thinker recovery from step_10

结果：

| Component | nan params | inf params | logits nan | logits min / max / mean |
|---|---:|---:|---:|---|
| Base model | 0 | 0 | False | `-15.95 / 29.14 / -2.14` |
| Policy (`step_10` LoRA) | 0 | 0 | False | `-16.39 / 29.81 / -2.51` |
| Ref (`step_10` LoRA) | 0 | 0 | False | `-16.39 / 29.81 / -2.51` |

结论：

- `step_10` 作为恢复起点完全干净
- 不仅权重 `0 nan / 0 inf`
- `policy/ref` thinker logits 也都完全正常

### Test 20: strict smoke recovery from step_10

结果：

- `strict_forward_success = 4`
- `strict_forward_failed = 0`
- `rollout_success_count = 4`
- `rollout_failed_count = 0`
- `thinker logits nan = False`
- `loss nan = False`
- 关键统计：
  - `thinker_logits`: `min=-22.625 max=32.5 mean=-2.18164 nan=False inf=False`
  - `log_softmax`: `min=-41.6875 max=-4.17233e-06 mean=-17.4219 nan=False inf=False`
  - `text_control`: `min=-31.0781 max=-2.01464e-05 mean=-18.0625 nan=False inf=False`
  - `advantages`: `min=0 max=0 mean=0 nan=False inf=False`
  - `policy_logps`: `min=-27.7812 max=-5.57899e-05 mean=-14.5938 nan=False inf=False`
  - `kl_tensor`: `min=-0.000488281 max=0.0322266 mean=0.000365973 nan=False inf=False`
  - `loss_mask`: `min=0 max=1 mean=0.233429 nan=False inf=False`

最关键发现：

- forward 正常
- logits 正常
- `loss = 0.0000`，并且不是 `nan`
- 但一次 `optimizer.step()` 后，LoRA 权重立刻损坏

保护触发层级：

- `loss.isnan()/isinf()` 检查：**未触发**
- `optimizer.step()` 后权重扫描：**触发**
  - 打印：
    - `[nan-inf] strict_post_step: nan (...) inf (...)`
- `save_pretrained()` 前检查：**触发并阻止保存**
  - `checkpoints/` 目录保持为空，没有把坏权重写盘

第一处重新变坏的位置：

- `base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.default.weight`

## 最新判断

到 `test20` 为止，责任边界再次收紧：

1. `step_10` 本身是干净的
2. `strict` forward 本身是正常的
3. `loss` 本身也不是 `nan`
4. 真正把 LoRA 写坏的是：
   - **一次 `backward + optimizer.step()` 更新**

这意味着当前最像的问题不是：

- rollout
- strict input
- multimodal features
- thinker 前向
- checkpoint 保存逻辑

而是：

- **训练更新阶段本身**
- 更具体地说，是 LoRA 参数在一次优化器更新后立即出现 `nan/inf`

因此下一步主线应从“恢复验证”切换为：

1. 检查当前优化器 / 学习率 / 梯度缩放配置
2. 打印 `optimizer.step()` 前的 LoRA `grad` 是否已经含 `nan/inf`
3. 区分：
   - `grad` 先坏
   - 还是 `grad` 正常，但 optimizer update 把参数写坏

## Test 21: grad vs optimizer.step boundary

入口：

- `script/test21_grad_vs_step_diag.sh`

目标：

- 不再继续看 rollout / strict / logits
- 只切开：
  - `grad` 是否在 backward 后已经坏
  - 还是 `optimizer.step()` 本身把参数写坏

新增诊断：

在 `scripts/rl/rollout_smoke_test.py` 中加入：

- `check_trainable_grads_for_nan_inf()`
- `diagnose_lora_params()`
- strict 路径 `backward` 后、`step` 前的 grad/param 诊断
- `optimizer.step()` 后的 param 诊断

结果：

### backward 后、step 前

- `LoRA grad` 完全干净：
  - `min=-0.00175381`
  - `max=0.00229073`
  - `mean=-4.76837e-07`
  - `nan=False`
  - `inf=False`
- 同一参数的 `LoRA param` 此时也仍然干净：
  - `min=-0.0191345`
  - `max=0.0199738`
  - `nan=False`
  - `inf=False`

### optimizer.step() 后

- 第一处坏参数仍然是：
  - `base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.default.weight`
- 同一参数 step 后立刻变成：
  - `min=nan`
  - `max=nan`
  - `mean=nan`
  - `nan=True`
  - `inf=True`

关键统计：

| 时机 | 张量 | min | max | mean | nan | inf |
|---|---|---:|---:|---:|---|---|
| backward 后 | grad | -0.00175 | +0.00229 | -4.77e-07 | False | False |
| step 前 | param | -0.01913 | +0.01997 | +6.53e-05 | False | False |
| step 后 | param | nan | nan | nan | True | True |

结论：

- `grad` 不是问题源头
- `optimizer.step()` 才是直接把 LoRA 参数写坏的地方

## 最新判断

到 `test21` 为止，责任边界已经进一步切清：

1. 干净 checkpoint 起点正常
2. forward/logits/loss/grad 都正常
3. **问题出在 `optimizer.step()` 更新阶段本身**

当前最可疑方向应收敛为：

1. optimizer 类型与参数组配置
2. LoRA 参数的 dtype / master weight / state dtype 交互
3. AdamW / weight decay / eps / betas 与当前 LoRA 更新路径的组合
4. mixed precision / scaler / optimizer state 初始化方式

## Test 22: optimizer factor diagnosis

入口：

- `script/test22_optimizer_step_diag.sh`
- `scripts/rl/optimizer_step_diag.py`

目标：

- 不再停留在“optimizer.step 会写坏参数”
- 继续切开到底是：
  - `weight_decay`
  - `lr`
  - `eps`
  - 还是 `fp16` LoRA 参数与 AdamW state dtype 的数值交互

当前 optimizer 配置：

- `torch.optim.AdamW`
- `lr=1e-6`
- `betas=(0.9, 0.999)`
- `eps=1e-8`
- `weight_decay=0.01`

参数与 state 诊断：

- LoRA trainable `param.dtype = torch.float16`
- LoRA `grad.dtype = torch.float16`
- 只有一个 param group，配置与上面一致
- optimizer state：
  - `step`: `torch.float32`
  - `exp_avg`: `torch.float16`
  - `exp_avg_sq`: `torch.float16`
- 关键现象：
  - `exp_avg_sq` 大量元素下溢到 `0`
  - 默认 `eps=1e-8` 在 `float16` 下也下溢为 `0`

对照实验：

| Config | eps | wd | lr | pre_step | post_step | Result |
|---|---:|---:|---:|---|---|---|
| A current | `1e-8` | `0.01` | `1e-6` | CLEAN | BROKEN | baseline |
| B wd=0 | `1e-8` | `0.0` | `1e-6` | CLEAN | BROKEN | `wd` 无关 |
| C eps=1e-4 | `1e-4` | `0.01` | `1e-6` | CLEAN | CLEAN | fixed |
| D eps=1e-4, wd=0 | `1e-4` | `0.0` | `1e-6` | CLEAN | CLEAN | fixed |
| E eps=1e-3 | `1e-3` | `0.01` | `1e-6` | CLEAN | CLEAN | fixed |
| F repeat A | `1e-8` | `0.01` | `1e-6` | CLEAN | BROKEN | reproducible |
| G lr=1e-7 | `1e-8` | `0.01` | `1e-7` | CLEAN | BROKEN | `lr` 无关 |

第一处坏参数始终一致：

- `base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.default.weight`

根因：

- 问题不在 `weight_decay`
- 问题也不在 `lr`
- 根因是：
  - **AdamW 默认 `eps=1e-8` 在 `float16` LoRA 参数下下溢为 `0`**
  - 同时 `exp_avg_sq` 也因 `float16` 精度太低而大量下溢到 `0`
  - 导致：
    - `denom = sqrt(exp_avg_sq) + eps = 0 + 0 = 0`
    - 随后更新发生除零
    - 参数立刻写成 `nan/inf`

修复条件：

- `eps >= 1e-4` 即可稳定阻止本问题复现
- `1e-4` 在 `float16` 下可安全表示
- `1e-3` 同样有效

## 最新判断

到 `test22` 为止，根因已经定位完成：

1. 不是 rollout / strict / GRPO loss
2. 不是 logits / grad
3. 不是 checkpoint 起点
4. 不是 `weight_decay`
5. 不是 `lr`
6. 而是：
   - **`float16` LoRA 参数 + AdamW `eps=1e-8` 的数值下溢**

因此下一步应直接改训练配置主线：

1. 将 AdamW `eps` 提高到 `1e-4`
2. 重新从干净 checkpoint（如 `step_10`）启动最小 strict smoke
3. 验证 step 后 LoRA 参数不再损坏
4. 若稳定，再恢复到更大的 strict / rollout scale smoke

## Test 23: eps=1e-4 fix validation

入口：

- `script/test23_eps_fix_strict_recover.sh`

修复：

- 在 `scripts/rl/rollout_smoke_test.py` 中，将 AdamW 的
  - `eps: 1e-8`
  - 改为
  - `eps: 1e-4`
- 改动位置：
  - `rollout_smoke_test.py:1176`

目标：

- 基于干净的 `step_10`
- 验证把 AdamW `eps` 提高到 `1e-4` 后：
  - forward / logits / loss / grad 仍正常
  - `optimizer.step()` 后 LoRA 参数不再损坏
  - 保存前保护不再触发

结果：

- strict smoke 成功启动并正常退出
- `strict_forward_success = 4`
- `strict_forward_failed = 0`
- `rollout_success_count = 4`
- `rollout_failed_count = 0`

关键诊断：

- `thinker_logits`: `min=-22.625 max=32.5 nan=False inf=False`
- `log_softmax`: `min=-41.6875 max=-4.17e-06 nan=False inf=False`
- `text_control`: `min=-31.0781 nan=False inf=False`
- `loss = 0.0000`（非 `nan`）
- `grad`：
  - `min=-0.00191593`
  - `max=0.00162506`
  - `mean=3.57628e-07`
  - `nan=False`
  - `inf=False`

step 前后参数对比：

| 时机 | test21 (`eps=1e-8`) | test23 (`eps=1e-4`) |
|---|---|---|
| step 前 param | `min=-0.01913 max=0.01997 nan=False` | `min=-0.01913 max=0.01997 nan=False` |
| step 后 param | `min=nan max=nan mean=nan` | `min=-0.01913 max=0.01997 nan=False` |

结论：

- `eps=1e-4` 修复验证通过
- `optimizer.step()` 后 LoRA 参数保持干净
- 保存前保护未触发（`[nan-inf]` 零匹配）
- 当前虽然只跑了 `1 step`，未到 `checkpoint_every=30` 的真实保存触发点
- 但 `weights_healthy=True`，说明保护层不会阻止后续正常保存

## 最新判断

到 `test23` 为止，数值问题已经完成修复闭环：

1. 根因：`float16` LoRA + AdamW `eps=1e-8` 下溢
2. 修复：将 `eps` 提高到 `1e-4`
3. 验证：最小 strict smoke 全链路恢复正常

因此下一步主线已经可以从“根因定位/修复验证”切换回：

1. 恢复更大一点的 strict smoke
2. 再恢复更大一点的 rollout / GRPO scale smoke
3. 然后再继续多卡多 worker 放大

## Test 24: strict scale after eps fix

入口：

- `script/test24_strict_scale_after_eps_fix.sh`

目标：

- 在 `eps=1e-4` 修复已验证通过的基础上
- 将 strict 路径从最小 smoke 放大一档
- 验证修复后的训练链在 `2 steps`、`8 rollouts` 条件下是否仍稳定

结果：

- strict smoke 成功启动并正常结束
- `strict_interleaved` 模式工作正常
- `strict_forward_success = 8`
- `strict_forward_failed = 0`
- `rollout_success_count = 8`
- `rollout_failed_count = 0`
- `worker_restart_count = 0`
- 完成 `2 steps`
  - `step 0`: `258.5s`
  - `step 1`: `230.3s`

forward / logits / loss：

- `thinker_logits`: `min=-25.25 max=34.1875 nan=False inf=False`
- `log_softmax`: `min=-47.4062 nan=False inf=False`
- `text_control`: `min=-31.0781 nan=False inf=False`
- `step 0 | loss 0.0000 | KL 0.0002`
- `step 1 | loss 0.0000 | KL 0.0001`

backward 后 grad：

- `step 0`: `min=-0.000394 max=0.000481 mean=1.19e-07 nan=False inf=False`
- `step 1`: `min=-0.001970 max=0.001745 mean=-2.98e-07 nan=False inf=False`

optimizer.step() 后 LoRA 参数：

- 两个 step 前后参数统计完全一致
- `step 0 pre`: `min=-0.0191345 max=0.0199738 nan=False`
- `step 0 post`: `min=-0.0191345 max=0.0199738 nan=False`
- `step 1 pre`: `min=-0.0191345 max=0.0199738 nan=False`
- `step 1 post`: `min=-0.0191345 max=0.0199738 nan=False`

保护与保存：

- `[nan-inf]` / `skipping` / `SKIPPING` / `WARNING` 全部零匹配
- 训练保护未触发
- `checkpoint_every=30`
- 当前只到 `global_step=2`
- 因此未触发真实保存，但当前状态说明后续保存不会被健康检查阻止

## 最新判断

到 `test24` 为止，`eps=1e-4` 修复不仅通过了最小 strict smoke，也通过了放大一档验证：

1. strict forward 稳定
2. logits 稳定
3. grad 稳定
4. optimizer.step() 后 LoRA 参数保持干净
5. 训练保护完全不再触发

因此当前主线可以从“strict 修复验证”切换到：

1. 恢复更大一点的 rollout / GRPO scale smoke
2. 然后继续多卡多 worker 放大

## Test 25: rollout / GRPO scale after eps fix

入口：

- `script/test25_grpo_scale_after_eps_fix.sh`

目标：

- 在 `eps=1e-4` 修复与 `test24` strict 放大验证通过之后
- 继续恢复更大一点的 rollout / GRPO scale smoke
- 检查在 `20 samples × 3 steps` 条件下训练链是否仍稳定

结果：

- 任务完成状态：`COMPLETED`
- 总耗时：`9m 15s`
- `steps = 3 / 3`
- `rollout_success_count = 12`
- `rollout_failed_count = 0`
- `strict_forward_success = 12`
- `strict_forward_failed = 0`
- `worker_restart_count = 0`
- `nan/inf` 触发次数：`0`
- `weights_healthy` 报警：`0`
- 峰值显存：`68.5GB / 80GB`

关键说明：

- 配置中的 `checkpoint_every = 30` 只是参数命中
- 当前 `max_steps = 3`
- 因此 `checkpoints/` 目录为空是正常现象
- 这不是保存失败，而是**尚未达到真实保存步**

结论：

- `eps=1e-4` 修复在 `20 samples × 3 steps` 的 scale-up 下完全稳定
- LoRA 权重未再损坏
- strict 路径、rollout 路径、训练更新路径都保持健康

## 最新判断

到 `test25` 为止，修复后的 custom strict / rollout / GRPO 链已经完成：

1. 最小 strict smoke 验证
2. strict 放大一档验证
3. rollout / GRPO scale-up 验证

并且三者都稳定通过。

因此当前主线可以继续切到：

1. 多卡多 worker 放大
2. 再逐步朝更原生的 VERL 训练形态靠近

## Test 26: pool multi-worker after eps fix

入口：

- `script/test26_pool_after_eps_fix.sh`
- 提交方式：
  - `sbatch --qos=qmultiple9 --time=2:00:00 -p A800Z --gres=gpu:2 --ntasks=1 --cpus-per-task=8 --output=script/sbatch_test26_%j.out --wrap="GPU_ID=0 bash script/test26_pool_after_eps_fix.sh"`

目标：

- 在 `eps=1e-4` 修复已通过单卡 strict / scale-up 验证之后
- 恢复多卡多 worker 的 `pool` 路线
- 验证修复后的训练链在 `2 GPU + 2 workers + 3 steps` 条件下是否仍稳定

结果：

- job 成功完成：`42133 -> COMPLETED`
- `rollout_mode = pool`
- `[pool] 2 workers on devices [0, 1]`
- `strict_forward_success = 12`
- `strict_forward_failed = 0`
- `rollout_success_count = 12`
- `rollout_failed_count = 0`
- 完成 `3 steps`
- `worker_restart_count = 0`
- worker crash = `0`
- 峰值显存：`68,533 MB`（约 `68.5GB / 80GB`）

训练稳定性：

- 所有 `thinker_logits / log_softmax / text_control / strict_pre_loss` 均无 `nan/inf`
- 每个 step 的 `[diag-grad strict_pre_step]` 都是 `all clean`
- `strict_pre_step` 与 `strict_post_step` 参数统计一致
- `optimizer.step()` 后 LoRA 参数仍保持干净
- `weights_healthy / nan-inf / SKIPPING` 命中数均为 `0`

checkpoint：

- `max_steps = 3`
- `checkpoint_every = 30`
- 因此未到真实保存步，`checkpoints/` 目录为空是正常现象

结论：

- 修复后的多卡多 worker 放大验证通过
- `eps=1e-4` 修复不仅在单卡 strict/GRPO scale 下稳定
- 在 `pool + 2 GPU + 2 workers` 形态下也稳定

## 最新判断

到 `test26` 为止，当前 custom 路线已经完成：

1. 单卡 strict 修复验证
2. 单卡 strict 放大验证
3. 单卡 rollout / GRPO scale-up 验证
4. 多卡多 worker pool 放大验证

并且都稳定通过。

因此当前主线可以继续切到：

1. 更贴近论文的训练形态对齐
2. 评估并推进更原生的 VERL 训练入口

## Test 27: VERL alignment and reward compatibility

目标：

- 不直接重构到原生 VERL
- 先对齐分析“当前 custom 链路 vs 论文原生 VERL 路线”
- 再选一个最小、最有价值的论文对齐点做验证

关键对照文件：

- `script/stage2_multiturn_rl.sh`
- `verl/verl/trainer/main_ppo.py`
- `verl/verl/utils/reward_score/avqa.py`
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/verl/trainer/ppo/core_algos.py`
- `verl/verl/workers/fsdp_workers.py`
- `verl/verl/utils/dataset/rl_dataset.py`
- `echo_rl/rewards.py`
- `scripts/rl/rollout_smoke_test.py`
- `scripts/rl_rollout/echo_interleaved_rollout_controller.py`
- `scripts/04_grpo_smoke/grpo_utils.py`
- `scripts/rl/isolated_rollout_worker.py`

对齐清单摘要：

| 维度 | 论文（VERL） | 当前 custom | 差距判断 |
|---|---|---|---|
| 训练入口 | `verl.trainer.main_ppo` (`Hydra + Ray`) | `rollout_smoke_test.py` (`argparse`) | 入口差异大 |
| 进程模型 | Ray actor + FSDP 分布式 | 子进程隔离 + JSON 通信 | custom 更安全但更重 |
| 模型并行 | FSDP 跨多卡 | 单卡训练，policy+ref 常驻 | 训练形态差距大 |
| rollout 引擎 | native `vLLMRollout.generate_sequences()` | `EchoVLLMBatchedRolloutController` | 功能接近 |
| rollout.n | 论文默认 `n=8` | 当前稳定验证到 `n=2` | 未对齐 |
| interleaved continuation | 作者原生多轮续写 | 已对齐 `inference_multiturn.py` | 已基本对齐 |
| reward / GRPO | `multiturn_rl_6`，支持 confidence-weighted accuracy | binary exact match + custom GRPO | 算法层仍有差距 |
| strict multimodal training | 原生训练体系内处理 | custom strict path 已稳定 | 方向一致 |
| 数据格式 | Parquet / DataProto | JSONL / custom dict | 格式不同但可桥接 |
| actor / rollout / ref 解耦 | 原生分角色 | 仍偏 custom 单体脚手架 | 仍未对齐 |

当前最主要还未对齐的 3 个点：

1. reward 函数中的 **confidence-weighted accuracy**
2. `rollout.n=8` 的真实稳定性验证
3. FSDP / 更原生的多卡训练形态

选择的“最值得先对齐的最小点”：

- **先对齐 reward：将 VERL 的 confidence-weighted accuracy 引入 custom pipeline**

为什么选它：

1. reward 直接影响训练信号质量，属于论文算法差距
2. 改动范围小，主要集中在：
   - `echo_rl/rewards.py`
   - rollout 输出中增加 `old_log_probs` / logprob 信息
3. 不需要先做 FSDP 或大规模重构
4. 比 `rollout.n=8` 更像“论文算法对齐”，而不是纯工程扩容

最小验证：

- 新增：
  - `script/test27_verl_reward_compat.py`
  - `script/test27_verl_reward_compat.sh`
- 在无 GPU 条件下验证 custom rollout 文本输出能否直接喂给 VERL reward

验证结果：

- `VERL compute_score()` 可直接接受 current custom rollout 文本输出
- 在**无 `old_log_probs`** 的模式下：
  - `custom_total` 与 `verl_total` 在 4/4 样本上完全一致
- 这说明：
  - 格式兼容性已成立
  - 下一步真正需要补的是 **confidence-weighted accuracy 所需的 `old_log_probs` 传递**

## 最新判断

到 `test27` 为止，当前最值得先推进的“论文最小对齐点”已经明确：

1. 不是先做 FSDP 大重构
2. 不是先盲目把 `n=2` 提到 `n=8`
3. 而是：
   - **先把 VERL 的 confidence-weighted reward 引入 custom pipeline**

因此下一步主线应改成：

1. 在 rollout 输出中保存足够的 logprob / old_log_probs
2. 在 `echo_rl/rewards.py` 中补齐 confidence-weighted accuracy
3. 验证 custom reward 与 VERL reward 在带 logprobs 条件下继续对齐
4. 然后再进入 `rollout.n=8` 稳定性验证

## Test 28: confidence-weighted reward alignment

目标：

- 将 VERL `multiturn_rl_6` 中的 confidence-weighted accuracy 引入 custom pipeline
- 验证 custom reward 与 VERL reward 在带 confidence/logprob 输入时继续对齐

已完成的 pipeline：

- `echo_interleaved_rollout_controller.py`
- `isolated_rollout_worker.py`
- `grpo_utils.py` (`build_rollout_metadata`)
- `rollout_rewards.py`
- `rewards.py` (`r_acc`)
- `rollout_smoke_test.py` (`all_metrics` 中加入 `avg_logprob`)

本轮改动文件：

- `scripts/rl/rollout_smoke_test.py`
  - 在 `all_metrics` 中加入 `avg_logprob`
- `script/test28_reward_confidence_align.py`
- `script/test28_reward_confidence_align.sh`
- `tasks/08_grpo_smoke_plan.md`
  - 新增 Section 35 记录结果

验证结果：

1. 核心公式
   - confidence-weighted score 在 5 档 logprob 设置下全部正确

2. `r_acc` synthetic 场景
   - `8/8` 场景通过
   - 覆盖：
     - 高置信正确
     - 低置信正确
     - 高置信错误
     - 低置信错误

3. 与 VERL 公式对齐
   - 在 `6` 个测试 logprob 值下与 VERL 公式 exact match

4. backward compatibility
   - `avg_logprob=None` 时仍退回旧的 binary 逻辑
   - 保持 `0.5 / 0` 兼容行为

5. end-to-end 现象
   - confidence-weighted accuracy 小于 binary accuracy，符合预期

6. 真实 rollout 数据
   - 对旧数据无回归

结论：

- custom pipeline 中的 confidence-weighted reward 对齐已完成
- 这条论文算法层差距已经补上

## 最新判断

到 `test28` 为止，当前“最值得先对齐的论文最小点”已经完成：

1. reward 算法层已对齐到 VERL 的 confidence-weighted accuracy
2. custom pipeline 保持 backward compatibility
3. 现在线路可以继续推进到：
   - `rollout.n=8` 的稳定性验证

## Test 29: rollout.n=8 minimal stability

入口：

- `script/test29_rollout_n8_smoke.sh`
- `script/sbatch_test29.sh`

提交方式：

- `sbatch script/sbatch_test29.sh`

目标：

- 不做 FSDP，不回归原生 VERL
- 先做一个最小、可控的 `rollout.n=8` 稳定性验证
- 回答修复后的 custom strict / rollout / GRPO 链在更贴近论文默认 rollout 数的条件下是否仍稳定

结果：

- job 成功完成：`42137`
- exit code = `0`
- `NUM_ROLLOUTS = 8`
- `Rollouts per sample = 8`
- `strict_forward_success = 8`
- `strict_forward_failed = 0`
- `rollout_success_count = 8`
- `rollout_failed_count = 0`
- 完成 `1 step`

训练与数值稳定性：

- 所有 `nan-diag` 均为 `nan=False inf=False`
- `logits` 值域正常（`min≈-45~-19`, `max≈32`）
- `loss_mask` 正常（约 `23%` 文本 token，`77%` audio+prompt 被 mask）
- `grad` 正常：
  - `min≈-0.00072`
  - `max≈0.00167`
  - `mean≈6.5e-7`
- `optimizer.step()` 后 LoRA 参数仍然干净
  - pre/post 一致
  - 无 `nan/inf`
- step 0 中 `ratio=1`，`PG loss≈0`
  - 参数更新量约 `1e-12`
  - 低于 `float16` 显示精度，属于正常 step-0 行为

保护与 worker：

- `nan-inf / SKIPPING / weights_healthy` 触发次数均为 `0`
- `worker_restart_count = 0`
- worker crash = `0`

显存：

- 峰值显存：`37408 MB`（约 `37GB / 80GB`）
- 余量充足

保存：

- `checkpoint_every = 30`
- 当前只跑 `1 step`
- 因此未触发 checkpoint 保存
- 但 `logs/rollouts.jsonl` 已保存（`8 entries`）

结论：

- `rollout.n=8` 最小稳定性验证通过
- 当前 custom 链路已经在更贴近论文默认 rollout 数的条件下通过最小验证

## 最新判断

到 `test29` 为止，当前 custom 路线已经完成：

1. strict 修复与放大验证
2. rollout / GRPO scale-up 验证
3. pool 多卡多 worker 验证
4. reward 算法层对齐
5. `rollout.n=8` 最小稳定性验证

因此下一步可以继续切到：

1. 更原生 VERL 训练入口的最小可行性验证
2. 或进一步做更贴近论文形态的 actor / rollout / ref 解耦验证

## A1 / A2: custom system-shape refactor toward VERL

### A1: actor / rollout / ref role boundary cleanup

已完成：

- 新建 `scripts/rl/engine_roles.py`
- 将原先混在 `rollout_smoke_test.py` 中的角色逻辑拆清为显式函数
- 当前至少已明确：
  - `collect_rollouts(...)`
  - `score_ref_logprobs(...)`
  - `score_ref_logprobs_multimodal(...)`
  - `update_actor_text(...)`
  - `update_actor_strict(...)`

意义：

- 不改变训练数学
- 但让当前入口的系统形态更接近论文/VERL 的：
  - rollout phase
  - ref scoring phase
  - actor update phase

### A2: unify batch / data flow

已完成：

- 新建 `scripts/rl/batch_schema.py`
- 引入轻量 `TrainingBatch` dataclass

关键字段：

| 字段 | 类型 | 设置阶段 |
|---|---|---|
| `rollout_data` | `List[dict]` | `collect_rollouts` |
| `samples` | `List[dict]` | `collect_rollouts` |
| `num_rollouts` | `int` | `collect_rollouts` |
| `metrics` | `List[Dict]` | `compute_rewards` |
| `advantages` | `Optional[Tensor]` | `build_advantages_from_metrics` |
| `encoded` | `Optional[List[Tuple]]` | `encode_text_rollouts` |

便捷属性：

- `size`
- `rollout_rewards`
- `sample_ids`
- `completions`
- `predictions`
- `avg_logprobs`

当前已经围绕统一对象工作的阶段：

1. `collect_rollouts`
2. `compute_rewards`
3. `build_advantages_from_metrics`
4. `encode_text_rollouts`
5. `update_actor_text`
6. `update_actor_strict`
7. logging / jsonl 写入

结论：

- A2 已完成
- 当前 custom 链路的数据流已经明显更接近 VERL / DataProto 风格
- 但仍保持轻量 dataclass 方案，而不是强行重写成完整 DataProto

## 最新判断

到 A2 为止，阶段 A 的前两步已经完成：

1. A1：角色边界拆清
2. A2：batch/data flow 统一

因此下一步应进入：

3. A3：多卡资源分工的过渡性实现

## A3: transitional multi-GPU role split

目标：

- 在不进入原生 FSDP 形态的前提下
- 先把当前 custom 链路的资源角色分开
- 让 training（actor/ref）与 rollout worker 不再抢同一张 GPU

本轮结果（job `42152`）：

- 提交成功，job 状态：`COMPLETED`
- 节点：`node42`
- 总耗时：约 `79s`
- pipeline 本体耗时：约 `64.8s`

GPU 角色分工：

| 角色 | GPU |
|---|---|
| Training (`actor/ref`) | `GPU 0` |
| Rollout workers | `GPU 1` |

配置摘要：

- 申请：`--gres=gpu:2`
- 数据集：`2` 条样本
- `num_rollouts = 2`
- `batch_size = 1`
- `max_steps = 1`
- `grpo_forward_mode = text_only`
- `rollout_worker_mode = pool`
- worker 数量：`1`
- backend：`hf`
- worker 重启：`0`

功能结果：

- rollout 成功：`2 / 2`
- rollout 失败：`0`
- training step 完成：`1`
- `loss = 0.2410`
- `reward = +0.275`

资源隔离结果：

- training 在 `GPU 0`
- rollout worker 在 `GPU 1`
- 日志与运行结果都证明：
  - training/rollout 的 GPU 角色边界已分开
  - worker 通过 `CUDA_VISIBLE_DEVICES` 成功绑定到 rollout GPU
  - 训练阶段前 worker 已关闭并释放

重要注意：

- 本轮在首个训练步后，出现了 `nan/inf` 警告
- 第一处坏权重位于：
  - 第一个 ViT layer 的 LoRA 参数
- 该问题**没有导致本次 job 崩溃**
- 但说明：
  - A3 的**结构目标**已完成
  - A3 形态下的**数值稳定性**还需要继续回归验证

结论：

- A3 的“过渡性多卡资源分工”已经实现
- 但不能把它记成“完全稳定完成”
- 更准确的状态应为：
  - **A3 结构完成，数值稳定性仍需在该形态下继续验证**

## A3 regression: resource split + stability closed

后续 `test32` / `sbatch_test32.sh` 在同样的 A3 形态上完成了稳定性回归：

- training (`actor/ref`) 固定在 `GPU 0`
- rollout worker 固定在 `GPU 1`
- `pool` worker 模式继续正常工作
- `4/4` rollout 成功，`0` 失败
- `2` 个训练步全部完成
- `0` 次 `nan/inf`
- `0` 次 `SKIPPING`
- `0` 次 `weights_healthy` 告警
- `0` 次 worker restart

结论更新为：

- **A3 已完成**
- 不只是结构分工成立
- 在 `AdamW(eps=1e-4)` 修复后的训练主线上，A3 形态下的数值稳定性也已经收口

## A4: scale-up validation on A3 system shape

后续 `test33` / `sbatch_test33.sh` 对 A1+A2+A3 形成的过渡版系统形态做了放大验证：

- 资源分工保持不变：
  - training (`GPU 0`)
  - rollout worker (`GPU 1`)
- `pool` worker 保持常驻
- 配置：
  - `max_samples=8`
  - `num_rollouts=4`
  - `max_steps=3`
  - `batch_size=2`
  - `grpo_forward_mode=text_only`

实际执行结果：

- 由于 `max_steps=3`，本轮实际执行的是 `3` 个 batch
- 每个 batch 含 `2` 个 sample、每 sample `4` 个 rollout
- 因此实际执行 rollout 数为：
  - `3 * 2 * 4 = 24`
- `24/24` rollout 成功
- `0` rollout 失败
- `3` 个训练步全部完成
- `0` 次 `nan/inf`
- `0` 次 worker restart
- `0` 次 `SKIPPING`
- `0` 次 `weights_healthy` 告警

吞吐观察：

- Batch 0 rollout：`114.3s`
- Batch 1 rollout：`104.3s`
- Batch 2 rollout：`77.5s`
- `pool` worker 预热后，rollout 吞吐提升约 `32%`

结论：

- **A4 已通过**
- A1（角色边界）+ A2（TrainingBatch 数据流）+ A3（过渡性多卡资源分工）在放大后仍然稳定
- **阶段 A 可视为整体完成**

## Test 30: minimal native VERL entrypoint feasibility

目标：

- 不直接重构回完整 VERL
- 先找出“当前 custom 稳定链路回归原生 VERL”的最小可执行切入点
- 做一个最小验证，判断这条切入点是否真实可行

阅读与分析的关键文件：

- `script/stage2_multiturn_rl.sh`
- `verl/verl/trainer/main_ppo.py`
- `verl/verl/trainer/ppo/ray_trainer.py`
- `verl/verl/workers/fsdp_workers.py`
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/verl/utils/dataset/rl_dataset.py`
- `echo_rl/rewards.py`
- `echo_rl/rollout_rewards.py`
- `tasks/08_grpo_smoke_plan.md` §34

结论：最小可行 VERL 原生切入点

- **先对齐数据格式**
  - 将当前 custom JSONL 转为 VERL 可读的 Parquet
  - 核心列：
    - `prompt`
    - `audios`
    - `answer`

为什么选它：

1. 风险最低
   - 不改已有稳定训练链
   - 只新增格式转换与验证脚本
2. 可独立验证
   - 不依赖完整 FSDP / Ray / main_ppo 训练成功
3. 排除法上最合理
   - 当前真正阻塞原生 VERL 训练的不是数据格式
   - 而是 `torch 2.9.0 + FSDP + Qwen2.5-Omni` 的 `SIGSEGV`
4. 即使后续修复 FSDP，数据格式对齐仍然是进入 VERL 的第一道门槛

当前最主要还未回归原生 VERL 的 3 个点：

| 阻塞点 | 严重程度 | 说明 |
|---|---|---|
| `torch 2.9.0 + FSDP + Qwen2.5-Omni` `SIGSEGV` | 致命 | 单卡/多卡都会在 FSDP wrap 阶段崩溃 |
| Ray 顶层依赖 | 中 | `verl/__init__.py` 顶层 import `ray`，导致轻量导入也受阻 |
| DataProto 接口对齐 | 低 | 当前 custom 输出与 VERL DataProto 仍有形态差异 |

最小验证：

- 新增：
  - `script/test30_verl_data_align.sh`
  - `script/test30_verl_data_step2.py`

执行方式：

- `bash script/test30_verl_data_align.sh`

验证结果：

- `4/4` 样本全部通过
- Parquet 必需列齐全：
  - `prompt`
  - `audios`
  - `answer`
- message 构造正确：
  - `1` 条 message
  - `content_types=['audio', 'text']`
- 音频加载正确：
  - `shape=(160000,)`
  - `sr=16000`
  - `duration=10.0s`
- processor chat template 正常：
  - 含 `<|audio_bos|><|AUDIO|><|audio_eos|>`
- tokenization 正常：
  - `input_ids: shape=(1, 364)`
  - `attention_mask: shape=(1, 364)`
- 多模态特征正常：
  - `input_features: shape=(1, 128, 30000), dtype=float32`
  - `feature_attention_mask: shape=(1, 30000), sum=1000`
- audio token 边界正确：
  - `<|audio_bos|>=1`
  - `<|audio_eos|>=1`
- answer 字段保留正确

补充说明：

- `test30` 的 Step 2 没有直接 import `RLHFDataset`
- 而是复制其核心逻辑
- 原因是：
  - `verl/__init__.py` 顶层 import `ray`
  - 即使只做 CPU-only / dataset 验证，也会被 Ray 依赖阻塞

## 最新判断

到 `test30` 为止，当前“回归更原生 VERL”的最小可执行切入点已经明确：

1. 数据格式对齐这条路是可行的
2. 真正阻塞原生 `main_ppo` 的不是数据，而是：
   - `torch 2.9.0 + FSDP + Qwen2.5-Omni` 的 `SIGSEGV`
3. 因此下一步主线应改成：
   - 继续沿 native VERL 方向推进时，优先围绕 **FSDP / main_ppo 可行性** 找最小绕行或最小验证点
   - 数据格式问题已经不是当前主阻塞

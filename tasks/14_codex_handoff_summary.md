# Task 14: Codex Handoff Summary

日期: 2026-05-14

本文档用于让另一个 Codex 窗口快速接手 Echo_Project 当前的核心上下文，避免重复探索。

## 1. 项目当前主线

当前主线已经从“验证 interleaved inference 机制”转到“恢复论文路线中的 vLLM rollout / VERL 训练可行性”。

已知大方向：

- `torch 2.6.0 + VERL/FSDP + Qwen2.5-Omni` 可以加载模型、FSDP wrap、进入 `trainer.fit()`
- `vLLM 0.12.0` 强依赖 `torch 2.9.0`
- `torch 2.9.0 + Qwen2.5-Omni` 在当前环境会 `SIGSEGV`
- 因此探索路线先变成：
  - 先尝试“较老但仍可能支持 Qwen2.5-Omni 的 vLLM wheel”
  - 如果宿主机 wheel 不行，再转容器 / source build

## 2. Interleaved inference 现状

主线脚本：

- `scripts/interleaved_infer.py`

当前统一 rollout controller：

- `scripts/rl_rollout/echo_interleaved_rollout_controller.py`

实验增强版：

- `scripts/interleaved_infer_custom.py`

当前认知：

- `scripts/interleaved_infer.py` 已实现真正的多轮音频-文本交错推理
- 但它属于较早的 HF 路线，包含“新开 user/assistant 轮次”的组织方式
- **当前新的 vLLM rollout 主线，不是按这种 chat 轮次重建继续推理**
- 当前新的 controller 改为对齐作者原始 `inference/inference_multiturn.py`：
  1. 第 1 轮建立一条初始 prompt
  2. 检测 `<seg>start,end</seg>`
  3. 裁剪对应音频片段
  4. 将 audio placeholder 直接 append 回同一条 prompt
  5. 将新音频 append 到 `multi_modal_data["audio"]`
  6. 在**同一条生成链**上继续 `generate`
- 已实现：
  - `SegStoppingCriteria`
  - duplicate seg 检测
  - `finalize_on_stop`
  - batched request-level rollout state machine

一句话区分：

- 旧实验脚本：更像“新开轮次继续聊”
- 新 controller：更像“同一条 prompt 原地续写”

## 3. vLLM 降级测试文档与脚本

核心文档：

- `tasks/13_vllm_downgrade_matrix.md`

相关脚本：

- `script/create_vllm_torch26_env.sh`
- `script/test6_vllm_downgrade_smoke.sh`
- `script/run_vllm_downgrade_matrix.sh`
- `scripts/vllm_qwen_omni_downgrade_smoke.py`

输出目录：

- `output/vllm_downgrade_smoke/`

## 4. 已验证的 vLLM 结论

### 4.1 vLLM 0.8.5

环境：

- conda env: `echo_vllm_torch26_085`
- `torch==2.6.0+cu124`
- `transformers==4.57.6`
- `huggingface-hub==0.36.0`

结果：

- `import vllm`：通过
- registry check：失败，120 秒超时
- `LLM(model=Qwen2.5-Omni-7B)`：失败
- 单音频 inference：未成功

关键深层失败点：

- `Engine core initialization failed`
- Triton / engine 初始化阶段报：
  - `GLIBC_2.34 not found`

含义：

- `0.8.5` 是目前最接近成功的 wheel 版本
- 它已经比 `0.8.4` 更接近真实可用
- 但现成 wheel 被当前集群系统库 / Triton ABI 卡住

关键产物：

- `output/vllm_downgrade_smoke/report_0.8.5.json`
- `output/vllm_downgrade_smoke/smoke_0.8.5.log`

### 4.1b vLLM 0.8.5 + Singularity 容器

这是目前最新且最重要的成功结果。

容器环境：

- 镜像：`docker://nvidia/cuda:12.4.0-devel-ubuntu22.04`
- SIF：`output/singularity/nvidia_cuda_12.4.sif`
- `glibc==2.35`
- `torch==2.6.0+cu124`
- `transformers==4.57.6`
- `huggingface-hub==0.36.0`
- `vllm==0.8.5`

结果：

- `import vllm`：通过
- `LLM load Qwen2.5-Omni`：通过
- 单音频 inference：通过
- stop words：通过
- LoRA capability check：通过

补充：

- 默认 `gpu_memory_utilization=0.6` 会在 KV cache 分配阶段 OOM
- 提升到 `0.85` 后 smoke 全通过

关键产物：

- `output/vllm_downgrade_smoke/report_0.8.5_singularity.json`
- `output/vllm_downgrade_smoke/smoke_0.8.5_singularity.log`

### 4.2 vLLM 0.8.4

外包智能体回传的关键结论已经写入 `tasks/13_vllm_downgrade_matrix.md`。

当前应以文档结论为准：

- `import vllm`：通过
- 但 `LLM load` 没有成功
- 暴露出一个更关键的问题：
  - `Qwen2_5OmniConfig` 顶层没有 `num_attention_heads`
  - vLLM 0.8.4 某些逻辑直接按顶层 config 读取

含义：

- `0.8.4` 很可能发布时根本没有原生支持 `Qwen2.5-Omni`
- 也就是说，`0.8.4` 太旧，模型结构支持缺失

### 4.3 vLLM 0.6.4

当前只确认到：

- 经过补依赖后可以 `import transformers, vllm`

但没有拿到可靠的结构化 smoke 结果，因此：

- 不能说它支持
- 也不能拿它作为主线

## 5. 当前最重要判断

不要再机械地从 `0.8.3 -> 0.7.x -> 0.6.x` 全扫一遍。

原因：

- `0.8.4` 已经显露出“太旧，不认识 Qwen2.5-Omni config”
- `0.8.5` 已经显露出“版本更接近支持模型，但 wheel 被系统 glibc / Triton 卡住”

所以更合理的判断是：

1. 旧 wheel 版本：
   - 大概率模型支持不够
2. 新一点的 wheel 版本：
   - 大概率模型更接近支持，但受制于当前系统 ABI

## 6. 当前推荐路线

优先路线：

- **Singularity / Apptainer 容器化运行 `vllm==0.8.5`**

这条路线已经被验证成功，不再只是建议。

在 rollout 层面的最新状态：

- batched Echo rollout controller 已完成并通过单样本 / batched smoke
- controller 逻辑对齐作者 `inference_multiturn.py` 的“同一条 prompt 续写”
- controller 已接到现有 subprocess rollout worker
- worker 端到端最小链路已经实跑通过：
  - `isolated_rollout_worker.py + vllm_batched + Singularity` 通过
  - `rollout_smoke_test.py + vllm_batched + Singularity` 通过
- 小规模放大版 custom GRPO smoke 也已通过：
  - `script/test11_vllm_batched_grpo_scale20.sh`
  - `MAX_SAMPLES=20`
  - `NUM_ROLLOUTS=2`
  - `MAX_STEPS=3`
  - `rollout_success_count=6`
  - `rollout_failed_count=0`
  - reward `mean=0.583 / min=0.5 / max=1.0`
  - 总耗时约 `584` 秒
- 现有 worker / reward / smoke 训练脚本仍使用老字段名，因此新增的是兼容层，而不是整条训练脚手架重写

备选路线：

- `source build vLLM`，目标保持 `torch 2.6.0`

## 7. 已经准备好的外包提示词

当前已经给用户提供过两类外包提示词：

1. 继续测试下一个 vLLM 版本（例如 `0.8.4`）
2. 在 Singularity / Apptainer 容器中重测 `vllm==0.8.5`

因此另一个 Codex 窗口如果接手：

- 不需要重新设计外包协议
- 只需要根据用户新回传的实验结果继续判读

## 8. 另一个 Codex 窗口最应该做什么

如果用户带回新的 rollout / worker 实验结果，优先判断四件事：

1. 是否沿用了成功配置：
   - Ubuntu 22.04 / `glibc 2.35`
   - `torch 2.6.0+cu124`
   - `transformers 4.57.6`
   - `huggingface-hub 0.36.0`
   - `vllm 0.8.5`
2. 是否将 `gpu_memory_utilization` 提高到 `0.85`
3. interleaved 推理链是否仍然遵守“同一条 prompt 续写 + append audio inputs”，而不是退回“新开 user/assistant”
4. 如果新的容器实验失败，失败是否属于偏离上述成功配置，而不是路线本身失效

当前这四件事里，前 3 项已经被最新实验正向验证：

- 配置沿用了成功的 Singularity `vllm 0.8.5` 路线
- `gpu_memory_utilization=0.85`
- continuation 逻辑保持为“同一条 prompt 续写”
- `test11` 进一步证明这条链路在小规模多样本 / 多 rollout / 多 step 下稳定

## 6.1 当前性能判断

基于 `test11` 和现有 `rollout_smoke_test.py` 结构，当前最明显的瓶颈是：

- `per_task` 模式反复新起 worker
- 每次 worker 都重新加载并初始化 `vllm.LLM`

因此：

- `persistent` 很可能能改善单卡总耗时
- 但它更偏单卡优化，不是当前最高优先级

当前更合理的顺序是：

1. 先验证 `pool` / 多 GPU worker 分摊是否稳定
2. 再比较 `per_task` / `persistent` / `pool` 的吞吐差异

不要把“优化单卡 persistent”排在“验证多卡 pool”之前。

## 6.2 Test12 状态

`test12` 是当前第一步多卡 pool 验证：

- `script/test12_vllm_batched_pool2_scale20.sh`
- 目标：`2` 卡、`2` 个 persistent rollout workers、`pool` 模式

当前结论：

- pool 核心链路已经通了
- Batch 0 完整通过（rollout + training）
- Batch 1 rollout 也已通过
- pool 的 shutdown / restart 流程已修复并实跑成功
- 未见 CUDA 异常、worker persistent 通信崩溃、rollout JSON 结构错误

当前未完整通过的原因：

- `srun --time=10:00` 的 SLURM QOS 时限不够
- 作业在 Batch 1 training model load 阶段被强制 kill

所以：

- 这不是“pool 路线失败”
- 而是“pool 路线已基本验证通过，但需要更长时间配额完成完整 3-step 结论”

下一步应当先做：

- 用更长 `--time` 重跑 `test12`
- 拿到完整 3-step 通过结果
- 然后再比较单卡 `test11` 与 2 卡 `test12` 的总吞吐

## 6.3 Test13 状态

`test13` 是当前 strict training forward 的最小验证：

- `script/test13_vllm_batched_strict_min.sh`
- 路线：`Singularity + vllm_batched rollout + strict_interleaved forward`

当前结论：

- 已成功通过
- 正确运行方式是：
  - `srun -p A800Z --gres=gpu:1 --time=10:00 bash script/test13_vllm_batched_strict_min.sh`
- 关键结果：
  - `strict_forward_success=4`
  - `strict_forward_failed=0`
  - `rollout_success_count=4`
  - `rollout_failed_count=0`
  - `peak_memory_mb=37198`
- 已完成 `step 0`

这意味着：

- `strict_interleaved` 已经从“只存在于代码里的 experimental 路径”
- 进入“最小 smoke 已验证可运行”的阶段

但仍需注意：

- 当前 step 日志里仍出现 `loss nan`
- 所以下一步更像是：
  - 放大 strict smoke 规模
  - 观察 `nan` 是否稳定复现
  - 判断是 reward / advantage / mask / logprob 数值问题，还是仅日志表现问题

## 6.4 Test14 状态

`test14` 是 strict 路径的小规模放大验证：

- `script/test14_vllm_batched_strict_scale.sh`
- 路线：`Singularity + vllm_batched rollout + strict_interleaved forward`

当前结论：

- strict rollout / forward 继续成功
- `strict_forward_success=4`
- `strict_forward_failed=0`
- `rollout_success_count=8`
- `rollout_failed_count=0`
- reward 数值正常（step 0: mean `0.812`, min `0.500`, max `1.000`）
- 但 `loss nan` 与 `KL nan` 继续稳定复现

这说明：

- 当前问题不像是 strict 输入重建失败
- 也不像是 reward 本身先坏掉
- 更像是当前 GRPO 实现中的 `policy/ref logprob / KL` 数值稳定性问题
- 并且这个问题不是 strict 专属，因为 text-only 路径也出现过同样的 `loss nan + KL nan`

因此下一步应当先做：

- 不再继续单纯放大 smoke
- 先做 GRPO 数值诊断：
  - `policy_logps` 是否含 `nan/inf`
  - `ref_logps` 是否含 `nan/inf`
  - `KL` 的 token 级统计
  - `loss_mask.sum()` 是否异常
  - `advantages` 范围是否异常

## 9. 不要重复做的事

- 不要再从头重新分析 interleaved inference 是否存在
- 不要再把 `0.8.4` 当成主线希望版本
- 不要把早期 HF 脚本里的“新开 user/assistant 轮次”误当成作者原始的 continuation 逻辑
- 不要再默认 `output/sft_v9b_merged` 存在
  - 当前很多测试实际用的是：
    `/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B`
- 不要把 `text-only` 成功当成 Echo rollout 成功

## 10. 当前一页纸结论

一句话总结给接手者：

> Echo 项目当前的 vLLM 路线已经收敛出一个可行解：
> `0.8.4` 太旧，模型支持缺失；
> `0.8.5` 宿主机 wheel 被当前集群的 `glibc/Triton` 卡住；
> 但 `0.8.5` 在 Ubuntu 22.04 / glibc 2.35 的 Singularity 容器里，已经成功通过 `import + LLM load + 单音频 inference`；
> 在此基础上，新的 batched Echo rollout controller 已经完成，并且按作者原始 `inference_multiturn.py` 的“同一条 prompt 续写”逻辑实现；它现在已经成功接入现有 custom rollout worker 链路，并通过了 `test11` 的小规模放大 smoke、`test12` 的 2 卡 pool 核心验证、`test13` 的 strict_interleaved 最小 smoke，以及 `test14` 的 strict 小规模放大 smoke。当前阶段不再是“链路能不能接通”，而是“定位并修复 GRPO 的 `loss/KL nan` 数值问题，再继续放大多卡 rollout 与 strict training”。

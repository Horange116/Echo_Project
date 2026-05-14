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

## 0. 最终目标

这个项目的最终目标不是长期停留在“custom 工程替代版能跑”，而是：

- **尽量贴合论文原文复现**

更具体地说，优先级是：

1. 先把论文核心机制复现对
   - audio-interleaved reasoning
   - 同一条 prompt 续写
   - `<seg>` 后裁音频再插回
   - vLLM rollout 主线
2. 再把工程链路跑稳
   - batched rollout
   - worker
   - strict interleaved training
   - 数值稳定性
3. 最后尽量往论文原生形态靠
   - 多卡多 worker / 更高吞吐
   - 更贴近论文的训练形态
   - 能回到更原生的 VERL 主入口就尽量回
   - actor / rollout / ref 的分布式与解耦也应逐步靠近

所有当前的 custom 修补、诊断和 smoke，都应理解为：

- **为回到论文原文主线服务**
- 不是长期停留在替代实现上

## 10.1 Test16 补充结论

`test16` 已继续把问题往前推进一层：

- `input_features` 正常
- `feature_attention_mask` 正常
- 但 `thinker(...)` 前向输出的 `logits` 已经全为 `nan`
- 并且把音频特征去掉后，text-only control forward 仍然是 `nan`

所以最新结论是：

- 当前问题不在多模态音频特征分支
- 不在 strict 输入重建
- 不在 reward / advantage / KL 这一层
- 而在更早的地方：
  - **模型本体（+LoRA adapter）的 thinker 前向本身**

因此下一步应改成：

1. 单独诊断 `policy_model` / `ref_model` 的 thinker 前向
2. 检查 model load / adapter load 后权重是否已经异常
3. 做最小 text-only thinker 对照，不经过 rollout / GRPO

## 10.2 Test17 最终责任边界

`test17` 已把问题彻底切开：

- `base model` 单独 text-only thinker forward 完全正常
- 一旦加载 `output/grpo_smoke/checkpoints/final` 的 LoRA adapter：
  - `policy model` logits 立刻全 `nan`
  - `ref model` logits 也立刻全 `nan`
- 权重扫描显示：
  - `base model`: `0 nan / 0 inf`
  - `policy LoRA`: `390 nan / 392 inf`
  - `ref LoRA`: `390 nan / 392 inf`
- 第一处异常权重：
  - `base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.default.weight`

最终结论：

- 当前 blocker **不是** rollout/controller/strict input/multimodal feature
- **不是** GRPO loss / advantage / KL 实现层
- 而是：
  - **LoRA checkpoint `output/grpo_smoke/checkpoints/final` 本身已损坏**
  - 损坏权重被保存到了磁盘
  - 因此任何加载该 adapter 的 thinker forward 都会直接输出 `nan`

下一步主线应调整为：

1. 停止继续复用损坏的 adapter checkpoint
2. 回溯哪一个训练 step/保存点第一次写入 `nan/inf`
3. 在训练/保存前加入权重与 logits `nan/inf` 保护
4. 用干净 adapter 重新做最小 thinker forward、strict smoke、再回到 rollout/GRPO 放大

## 10.3 Test18 回溯与防护

`test18` 已把“坏 checkpoint 从哪开始、以后怎么拦住”查清楚：

回溯扫描结果：

- `output/grpo_smoke/checkpoints/step_5`: `0 nan / 0 inf`
- `output/grpo_smoke/checkpoints/step_10`: `0 nan / 0 inf`
- `output/grpo_smoke/checkpoints/final`: `1,871,846 nan / 18,312,005 inf`

结论：

- 第一个损坏保存点是：
  - `output/grpo_smoke/checkpoints/final`
- `step_5` 和 `step_10` 都是干净的
- 损坏发生在 `step_10 -> final` 之间的训练步骤中

当前训练路径新增了最小保护：

1. `check_model_for_nan_inf()` 扫描可训练参数
2. strict 路径 `loss.isnan()/isinf()` 时跳过 backward
3. `optimizer.step()` 后立刻扫描 LoRA 权重
4. `save_pretrained()` 前再次扫描；若检测到损坏则跳过保存坏 checkpoint

这意味着：

- 当前主问题已经不再是“哪里 first nan”
- 而是“用干净 checkpoint 重新接着跑，并观察保护在哪一步触发”

下一步主线：

1. 从干净 checkpoint（优先 `step_10`）继续，而不是再用 `final`
2. 重新做最小 thinker forward 验证 LoRA 仍然干净
3. 再重新做 strict 最小 smoke
4. 若再次触发保护，则继续精确回溯是哪一个训练 step 把权重写坏

## 10.4 Test19 / Test20 恢复验证

恢复链已经验证完：

- `test19` 证明：
  - `step_10` 的 LoRA 权重仍然干净
  - `policy/ref thinker forward` 都正常
- `test20` 证明：
  - 基于 `step_10` 的 strict 最小 smoke 可以正常 forward
  - `thinker logits` 正常
  - `loss` 也不是 `nan`
  - 但一次 `optimizer.step()` 后，LoRA 权重立刻重新出现 `nan/inf`

也就是说，当前最关键的新结论是：

- 问题**不在 checkpoint 起点**
- 不在 strict input
- 不在 multimodal feature
- 不在 thinker forward
- 不在 loss 本身是否为 `nan`
- 而在：
  - **一次训练更新（`backward + optimizer.step()`）之后，LoRA 参数被立刻写坏**

当前保护工作正常：

1. `loss.isnan()/isinf()` 没触发（说明 loss 表面正常）
2. `optimizer.step()` 后权重扫描触发
3. 保存前防护拦住了坏 checkpoint，没有再次把坏权重写盘

现在主线应继续收敛到：

1. 检查 LoRA 参数的 `grad` 在 step 前是否已含 `nan/inf`
2. 如果 `grad` 干净，则检查是否是 optimizer update 本身把参数写坏
3. 重点看优化器、学习率、梯度缩放/裁剪、以及 LoRA 参数更新配置

## 10.5 Test21 最终切开 grad vs step

`test21` 已把训练更新阶段继续切开：

- backward 后：
  - LoRA `grad` 完全干净（`nan=False`, `inf=False`）
- step 前：
  - LoRA `param` 仍然干净
- `optimizer.step()` 后：
  - 同一 LoRA 参数立刻变成 `nan/inf`

因此现在已经可以明确说：

- 不是 `grad` 先坏
- **是 `optimizer.step()` 本身把参数写坏**

当前主嫌疑不再是：

- rollout
- strict input
- forward/logits
- loss
- grad

而是：

1. optimizer 类型/参数组配置
2. LoRA 参数 dtype 与 optimizer state dtype 的交互
3. AdamW / eps / weight_decay / betas 配置
4. mixed precision / scaler / state 初始化问题

## 10.6 Test22 最终根因

`test22` 已把 optimizer 责任继续收敛到具体数值因素：

- optimizer: `torch.optim.AdamW`
- `lr=1e-6`
- `betas=(0.9, 0.999)`
- `eps=1e-8`
- `weight_decay=0.01`

关键事实：

- LoRA trainable 参数是 `float16`
- `grad` 也是 `float16`
- optimizer state 里的：
  - `exp_avg` 是 `float16`
  - `exp_avg_sq` 也是 `float16`

对照实验表明：

- 改 `weight_decay=0`：**无效**
- 改 `lr=1e-7`：**无效**
- 只把 `eps` 从 `1e-8` 提到 `1e-4`：**立刻修复**
- `eps=1e-3`：同样稳定

最终根因：

- **AdamW 默认 `eps=1e-8` 在 `float16` LoRA 参数下下溢为 `0`**
- `exp_avg_sq` 也因 `float16` 精度而大量下溢到 `0`
- 结果 `denom = sqrt(exp_avg_sq) + eps = 0 + 0 = 0`
- `optimizer.step()` 出现除零，参数立刻写成 `nan/inf`

因此当前主线修复已经很清楚：

1. 把 AdamW `eps` 提高到 `1e-4`
2. 从干净 checkpoint（优先 `step_10`）重新做最小 strict smoke
3. 验证 LoRA 参数 step 后不再损坏
4. 稳定后再恢复到更大规模 strict / rollout smoke

## 10.7 Test23 修复验证通过

`test23` 已完成正式修复验证：

- 在 `scripts/rl/rollout_smoke_test.py` 中，把 AdamW `eps` 从 `1e-8` 改到 `1e-4`
- 基于干净的 `step_10` 重新跑最小 strict smoke

结果：

- forward 正常
- logits 正常
- loss 正常
- grad 正常
- `optimizer.step()` 后 LoRA 参数仍然干净
- 保存前防护未触发

也就是说：

- 根因定位完成
- 修复条件已验证有效
- 当前训练链已经从“会在 step 后立刻写坏参数”恢复成“最小严格训练 smoke 正常”

因此现在主线可以切回到放大验证：

1. 先恢复更大一点的 strict smoke
2. 再恢复更大一点的 rollout / GRPO scale smoke
3. 最后再继续多卡多 worker 扩展

## 10.8 Test24 修复后 strict 放大通过

`test24` 已证明：

- `eps=1e-4` 修复不仅能通过最小 strict smoke
- 也能通过放大一档的 strict 验证

结果摘要：

- `strict_forward_success = 8`
- `strict_forward_failed = 0`
- `rollout_success_count = 8`
- `rollout_failed_count = 0`
- `2 steps` 全部正常完成
- logits 正常
- loss 正常
- grad 正常
- `optimizer.step()` 后 LoRA 参数仍然干净
- 训练保护零触发

这意味着：

- strict 路径已经从“最小修复验证通过”
- 进入到“放大一档仍稳定”的状态

因此下一步主线可以继续切回：

1. 更大一点的 rollout / GRPO scale smoke
2. 再继续多卡多 worker 放大

## 10.9 Test25 修复后 GRPO scale 放大通过

`test25` 已验证：

- `eps=1e-4` 修复不仅在 strict 路径稳定
- 在 `20 samples × 3 steps` 的 rollout / GRPO scale-up 下也完全稳定

结果摘要：

- `steps = 3 / 3`
- `rollout_success_count = 12`
- `rollout_failed_count = 0`
- `strict_forward_success = 12`
- `strict_forward_failed = 0`
- `nan/inf` 触发 = `0`
- `weights_healthy` 报警 = `0`
- `worker_restart_count = 0`
- 峰值显存约 `68.5GB / 80GB`

补充说明：

- `checkpoint_every = 30`
- 当前 `max_steps = 3`
- 因此 `checkpoints/` 目录为空是正常现象，不代表保存失败

这意味着：

- 修复后的 custom strict / rollout / GRPO 链已经在更大 scale 下通过
- 当前下一步可以正式切回：
1. 多卡多 worker 放大
2. 再逐步往更原生的 VERL 训练形态靠近

## 10.10 Test26 修复后多卡多 worker 放大通过

`test26` 已验证：

- `eps=1e-4` 修复不仅在单卡 scale-up 下稳定
- 在 `pool + 2 GPU + 2 workers` 的多卡多 worker 形态下也完全稳定

结果摘要：

- `strict_forward_success = 12`
- `strict_forward_failed = 0`
- `rollout_success_count = 12`
- `rollout_failed_count = 0`
- `3 steps` 全部正常完成
- `[pool] 2 workers on devices [0, 1]`
- `worker_restart_count = 0`
- worker crash = `0`
- `weights_healthy / nan-inf / SKIPPING = 0`
- 峰值显存约 `68.5GB / 80GB`

补充说明：

- 这次采用 `sbatch` 提交，避免了交互式 `srun` 的 10 分钟限制
- `checkpoint_every = 30`
- 当前 `max_steps = 3`
- 因此没有真实保存 checkpoint 是正常现象，不代表保存失败

这意味着：

- 当前 custom strict / rollout / GRPO / pool 多卡链路已经整体稳定
- 下一步可以继续往：
1. 更贴近论文的训练形态
2. 更原生的 VERL 训练入口
  推进

## 10.11 Test27 论文对齐方向明确

`test27` 已完成“当前 custom 链路 vs 论文原生 VERL 路线”的最小对齐分析和验证。

当前最主要还未对齐的 3 个点：

1. VERL `multiturn_rl_6` 中的 **confidence-weighted accuracy**
2. `rollout.n=8` 的真实稳定性验证
3. FSDP / 更原生的多卡训练形态

本轮选择的“最值得先对齐的最小点”是：

- **先对齐 reward**
  - 把 VERL 的 confidence-weighted accuracy 引入 custom pipeline

原因：

- 这是论文算法层面的差距
- 对训练信号质量有直接影响
- 改动范围小于 FSDP / VERL 主入口回归
- 比单纯把 `n=2` 扩到 `n=8` 更有论文对齐价值

最小验证结果：

- `script/test27_verl_reward_compat.py`
- `script/test27_verl_reward_compat.sh`

已经证明：

- 当前 custom rollout 文本输出可以直接喂给 VERL reward
- 在**没有 old_log_probs** 的模式下：
  - custom reward 与 VERL reward 在样本上完全一致

这意味着下一步真正需要补的是：

1. rollout 输出中传递 `old_log_probs`
2. 在 `echo_rl/rewards.py` 中支持 confidence-weighted accuracy
3. 再做一次带 logprobs 的 reward 对齐验证

## 10.12 Test28 reward 算法层对齐完成

`test28` 已完成 custom reward 与 VERL confidence-weighted reward 的最小论文对齐。

完成内容：

- 在 custom pipeline 中接通了 confidence/logprob 信号传递链：
  - `echo_interleaved_rollout_controller.py`
  - `isolated_rollout_worker.py`
  - `grpo_utils.py`
  - `rollout_rewards.py`
  - `rewards.py`
  - `rollout_smoke_test.py`
- `rollout_smoke_test.py` 中 `all_metrics` 现在会保存 `avg_logprob`

验证结果：

- confidence-weighted 核心公式验证通过
- `r_acc` synthetic 场景 `8/8` 通过
- 与 VERL 公式在 6 个 logprob 测试点 exact match
- `avg_logprob=None` 时保持旧 binary 行为，backward compatibility 正常
- 真实 rollout 数据无回归

这意味着：

- 当前最小论文算法差距已经补齐
- 现在主线可以继续切回：
  1. `rollout.n=8` 稳定性验证
  2. 然后再评估更原生的 VERL 训练入口 / 训练形态推进

## 10.13 Test29 rollout.n=8 稳定性通过

`test29` 已验证：

- 当前修复后的 custom strict / rollout / GRPO 链
- 在更贴近论文默认配置的 `NUM_ROLLOUTS=8` 下
- 仍然稳定

结果摘要：

- `NUM_ROLLOUTS = 8`
- `strict_forward_success = 8`
- `strict_forward_failed = 0`
- `rollout_success_count = 8`
- `rollout_failed_count = 0`
- `1 step` 正常完成
- `nan/inf` 触发 = `0`
- `worker_restart_count = 0`
- worker crash = `0`
- 峰值显存约 `37GB / 80GB`

补充说明：

- 这次使用 `sbatch` 提交
- `checkpoint_every = 30`
- 当前只跑 `1 step`
- 因此没有真实保存 checkpoint 是正常现象，不代表保存失败

这意味着：

- reward 对齐之后，`rollout.n=8` 的最小稳定性也已验证通过
- 当前下一步可以正式切到：
1. 更原生 VERL 训练入口的最小可行性验证
2. 或 actor / rollout / ref 更贴论文的解耦验证

## 10.15 A1 / A2 已完成

当前阶段 A 的前两步已经完成：

### A1：角色边界拆清

- 新建 `scripts/rl/engine_roles.py`
- 将当前 custom 入口中的三类职责拆清：
  - rollout collection
  - ref scoring
  - actor update

### A2：统一 batch/data flow

- 新建 `scripts/rl/batch_schema.py`
- 引入轻量 `TrainingBatch` dataclass
- 当前链路已不再主要依赖零散平行 list/dict
- 关键阶段现在围绕同一个 batch 对象流转：
  - `collect_rollouts`
  - `compute_rewards`
  - `build_advantages_from_metrics`
  - `encode_text_rollouts`
  - `update_actor_text`
  - `update_actor_strict`

这意味着：

- 当前 custom 链路不仅逻辑稳定
- 系统形态上也开始更接近 VERL 风格

因此下一步主线应进入：

- **A3：多卡资源分工的过渡性实现**

## 10.16 A3 资源分工已实现，但数值稳定性待收尾

`test31` / job `42152` 已完成当前 A3 的核心目标：

- training (`actor/ref`) 固定在 `GPU 0`
- rollout worker 固定在 `GPU 1`
- `pool` worker 模式正常工作
- rollout `2/2` 成功
- training `step 0` 成功结束

这说明：

- **A3 的结构性目标已完成**
  - 当前 custom 链路已经实现过渡性多卡资源分工

但这轮也暴露了一个重要提醒：

- 首个训练步后出现 `nan/inf`
- 位于第一个 ViT layer 的 LoRA 权重
- 虽然没有导致作业崩溃，但说明：
  - A3 形态下的数值稳定性不能直接视作完全收口

所以当前应如何理解 A3：

- 角色分工 / GPU 资源边界：**已完成**
- 在该新形态下的训练稳定性：**仍需继续验证**

## 10.17 A3 稳定性回归完成

`test32` / `sbatch_test32.sh` 已完成 A3 形态下的稳定性回归：

- training (`actor/ref`) 固定在 `GPU 0`
- rollout worker 固定在 `GPU 1`
- `pool` worker 正常工作
- `4/4` rollout 成功
- `2` 个训练步全部完成
- `0` 次 `nan/inf`
- `0` 次 `SKIPPING`
- `0` 次 `weights_healthy` 告警
- `0` 次 worker restart

因此 A3 的结论更新为：

- **A3 已完成**
- 资源分工不只是结构上成立
- 在 `AdamW(eps=1e-4)` 修复后的训练主线上，A3 形态下的数值稳定性也已收口

## 10.18 A4 放大验证完成

`test33` / `sbatch_test33.sh` 对阶段 A 的过渡版系统形态做了更大规模验证。

配置：

- `max_samples=8`
- `num_rollouts=4`
- `max_steps=3`
- `batch_size=2`
- `grpo_forward_mode=text_only`
- 资源分工保持：
  - training (`GPU 0`)
  - rollout worker (`GPU 1`)

注意：

- 因为 `max_steps=3`
- 实际执行的是 `3` 个 batch，而不是遍历完 `8` 条样本
- 每个 batch 包含 `2` 个 sample、每 sample `4` 个 rollout
- 所以本轮实际执行 rollout 总数为：
  - `3 * 2 * 4 = 24`

结果：

- `24/24` rollout 成功
- `0` rollout 失败
- `3` 个训练步全部完成
- `0` 次 `nan/inf`
- `0` 次 `SKIPPING`
- `0` 次 `weights_healthy` 告警
- `0` 次 worker restart

吞吐观察：

- Batch 0 rollout：`114.3s`
- Batch 1 rollout：`104.3s`
- Batch 2 rollout：`77.5s`
- `pool` worker 预热后，rollout 吞吐提升约 `32%`

结论：

- **A4 已通过**
- A1 / A2 / A3 / A4 全链路收口
- **阶段 A 可以视为整体完成**

## 10.14 Test30 原生 VERL 最小切入点已明确

`test30` 已完成“从当前 custom 稳定链路回归原生 VERL”的最小切入点验证。

结论：

- **最小可行切入点是数据格式对齐**
  - 将 custom JSONL 转换为 VERL 可读的 Parquet
  - 关键列：
    - `prompt`
    - `audios`
    - `answer`

验证结果：

- `script/test30_verl_data_align.sh`
- `script/test30_verl_data_step2.py`
- `4/4` 样本全部通过
- tokenization、多模态特征、audio token 边界、answer 保留全部正确

这说明：

- custom 数据内容已经可以被包装成 VERL 可接受的输入形态
- 数据格式本身不是当前回归原生 VERL 的阻塞点

当前真正的主阻塞变成：

1. `torch 2.9.0 + FSDP + Qwen2.5-Omni` 的 `SIGSEGV`
2. `verl/__init__.py` 顶层 Ray 依赖
3. 更原生的 DataProto / actor-rollout-ref 调用链对齐

因此当前主线应继续切到：

1. 围绕 native `main_ppo` / FSDP 可行性做最小验证
2. 数据格式问题可以暂时视为已打通，不再是首要阻塞

## 11. 当前还缺哪些步骤

当前最缺的不是再找根因，而是继续把修复后的训练链往论文要求的规模和形态上推。

按优先级排序：

1. 跑修复后的 strict 放大验证
   - 即 `eps=1e-4` 修复后的更大一档 strict smoke
2. 恢复更大一点的 rollout / GRPO scale smoke
   - 验证修复不只在最小 smoke 下成立
3. 继续多卡多 worker rollout 放大
   - 优先继续 `pool` / worker 分摊形态
4. 把 strict_interleaved 从“已能跑”推进到“稳定主线”
   - 持续减少对 `text_only` fallback 的依赖
5. 再评估并推进更原生的 VERL 训练入口
   - 尽量朝 `script/stage2_multiturn_rl.sh` 的训练形态靠拢

一句话：

- 现在应从“根因定位阶段”切回“修复后放大验证阶段”

## 12. 关于 GPU 卡数不足

论文原始路线是 8 卡训练，但当前环境里**拿不到 8 张卡并不阻止继续推进**。

如果后续空闲卡最多只有 3 张：

- **可以接受 3 卡作为缩小版复现**
- 这不等于论文硬件规模上的一比一复现
- 但仍然可以完成：
  - 核心机制复现
  - rollout / strict training 稳定性验证
  - 多 worker 并行形态验证

建议的表述方式：

- **算法与机制尽量贴原文**
- **硬件规模采用 3 GPU 的降配复现**

这意味着：

1. rollout 并行可以继续做
2. 多 worker / pool 仍然有意义
3. 训练 batch / 吞吐 / 速度需要下调
4. 最终总结时要明确说明与论文 8 卡配置的硬件差异

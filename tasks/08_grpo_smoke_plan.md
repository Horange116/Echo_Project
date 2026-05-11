# GRPO Smoke 方案 — v9b-diverse-cot-2epoch

日期: 2026-05-09

## 1. 架构总览

ms-swift 的 `GRPOTrainer` 基于标准文本生成，不支持 audio-interleaved 的多步推理（生成 seg → 插入音频 → 继续生成 → 检测重复 → finalize）。因此需要**自定义 GRPO 训练脚本**，复用现有的 `run_interleaved` 和 `rollout_reward`。

```
┌─────────────────────────────────────────────────────────┐
│                   grpo_smoke_train.py                    │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Data     │──>│ Interleaved  │──>│ rollout_reward  │  │
│  │ (44 qs)  │   │ Rollout × N  │   │ Compute reward  │  │
│  └──────────┘   └──────────────┘   └────────┬────────┘  │
│                                             │           │
│  ┌──────────┐   ┌──────────────┐            │           │
│  │ Policy   │<──│ GRPO Update  │<───────────┘           │
│  │ (LoRA)   │   │ + KL est.    │                        │
│  └──────────┘   └──────────────┘                        │
│                                                         │
│  ┌──────────┐   ┌──────────────┐                        │
│  │ Ref Model│   │ KL(reuse)    │                        │
│  └──────────┘   └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

## 2. 需要新增的脚本

### 2.1 `scripts/04_grpo_smoke/grpo_smoke_train.py`（主训练脚本）

**加载阶段：**
| 组件 | 来源 | 说明 |
|------|------|------|
| Policy model | v9b-2epoch checkpoint-3078 + LoRA (r=8, α=32) | 用 PEFT 加载，可训练 |
| Reference model | 同 checkpoint-3078 | 冻结 eval 模式，用于 KL |
| Processor | Qwen2.5-Omni-7B processor | 文本/音频预处理 |
| Data | `output/judge/split_rl.jsonl` | 44 条，每个包含 audio_path, question, choices, answer |

**Rollout 阶段：**
- 每个 query 执行 N=2 rollouts（smoke 测试，后续可扩大到 4）
- 复用 `scripts/interleaved_infer.py::run_interleaved`（策略 B 配置）
- 每次 rollout 输出: `(final_response, rollout_metadata)`

**Reward 阶段：**
- 调用 `echo_rl.rollout_rewards::rollout_reward(response, gt, meta)`
- 对 rollout 中的 token 计算 log-probabilities（policy + ref）

**训练阶段（标准 GRPO loss）：**

```
对于每个 prompt p，有 G=2 个 rollout:
  1. rollout_total[r] = rollout_reward(p, r)
  2. advantage[r] = (rollout_total[r] - mean(rollout_total)) / std(rollout_total)
  3. 对 rollout 中每个 token t:
     ratio = exp(log_probs_policy[t] - log_probs_old[t])
     clipped_ratio = clamp(ratio, 1-ε, 1+ε)
     loss = -min(ratio × advantage, clipped_ratio × advantage)
     + β × KL(π_θ || π_ref)  [每个 token 的 KL]
```

**参考实现：** ms-swift `grpo_trainer.py` 中 `_get_per_token_logps`（提取 token log-probs）和 GRPO loss 计算逻辑。

### 2.2 `scripts/04_grpo_smoke/submit_grpo_smoke.sh`

SLURM 提交脚本，1 GPU，~16G VRAM。

### 2.3 `scripts/04_grpo_smoke/grpo_utils.py`（辅助模块，可选）

如果主脚本过长，拆分出：
- `compute_grpo_loss()` — GRPO loss 计算
- `compute_kl()` — KL 散度计算
- `compute_advantage()` — 优势计算

## 3. 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Policy init | v9b-diverse-cot-2epoch ckpt-3078 | — |
| Reference | 同 checkpoint | KL 估计 |
| 训练方式 | LoRA (r=8, α=32, all-linear) | 轻量微调 |
| 数据 | split_rl.jsonl (44 条) | — |
| Rollouts/query | 2 | smoke 用 |
| GRPO loss | clipped surrogate + KL(β=0.04) | 标准 GRPO |
| 学习率 | 1e-5 | AdamW |
| Batch size | 4 (≈8 rollouts/step) | — |
| Epochs | 1 (≈11 steps) | smoke 够用 |
| Max rounds | 5 (strategy B) | stop+finalize |
| 精度 | float16 | — |
| Optimizer | AdamW (β₁=0.9, β₂=0.99) | — |
| Warmup | 0 steps (小数据) | — |

### 3.1 风险点

| 风险 | 影响 | 缓解方案 |
|------|------|----------|
| **生成速度慢** | 每个 rollout 跑 5 轮 interleaved inference，44×2=88 次 | Smoke 只跑 1 epoch，后续 scaling 用 vLLM |
| **GPU 内存不足** | Policy + Reference 双模型 + LoRA | Reference 用 CPU offload 或量化（NF4） |
| **log-prob 提取** | Interleaved 生成是多步的，需合并所有 round 的 token log-probs | 每轮生成后用 `model(**inputs).logits` 重新计算 |
| **PEFT + run_interleaved 兼容** | `run_interleaved` 需要特定模型接口 | 需要确保 LoRA 合并/解包不影响 interleaved 流程 |
| **GRPO 实现正确性** | 自定义 loss 可能实现有误 | 参考 ms-swift/trl 的标准实现，smoke 后对比 reward 趋势 |

## 4. 日志字段

| 字段 | 类型 | 来源 |
|------|------|------|
| `train/loss` | float | GRPO loss |
| `train/approx_kl` | float | KL(π_θ \|\| π_ref) 均值 |
| `reward/rollout_total` | float | rollout_total 均值 |
| `reward/base_total` | float | total (base) 均值 |
| `reward/accuracy` | float | accuracy reward 均值 |
| `reward/segment` | float | segment reward 均值 |
| `reward/format` | float | format reward 均值 |
| `reward/consistency` | float | consistency reward 均值 |
| `rollout/duplicate_penalty` | float | duplicate_penalty 均值 |
| `rollout/finalize_penalty` | float | finalize_penalty 均值 |
| `rollout/unique_segment_bonus` | float | unique_segment_bonus 均值 |
| `rollout/round_penalty` | float | round_penalty 均值 |
| `rollout/triggered_interleaved_rate` | float | triggered_interleaved=True 的比例 |
| `rollout/unique_segment_count` | float | unique_segment_count 均值 |
| `rollout/duplicate_seg_count` | float | duplicate_seg_count 均值 |
| `rollout/finalize_rate` | float | finalize_triggered=True 比例 |
| `rollout/answer_rate` | float | 有 answer 的比例 |
| `rollout/answer_correct_rate` | float | answer 正确的比例 |
| `train/grad_norm` | float | 梯度范数 |
| `train/learning_rate` | float | 当前学习率 |
| `train/epoch` | float | 当前 epoch |

日志格式：TensorBoard + JSONL（`output/grpo_smoke/logs/`）

## 5. 训练命令

```bash
# 交互式
cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project
conda activate qwen_echo
export QWEN_OMNI_SKIP_SPK=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1)

python -u scripts/04_grpo_smoke/grpo_smoke_train.py \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --adapter_path /path/to/checkpoint-3078 \
  --data_path output/judge/split_rl.jsonl \
  --output_dir output/grpo_smoke \
  --num_rollouts 2 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --max_rounds 5

# SLURM
sbatch scripts/04_grpo_smoke/submit_grpo_smoke.sh
```

## 6. 训练流程详细步骤

```
初始化:
  1. Load policy model (base + adapter) + LoRA
  2. Load reference model (same base + adapter, eval, CPU offload)
  3. Load processor
  4. Load dataset (44 samples)

训练循环 (× ~11 steps for 1 epoch on 44 samples, batch=4):
  for batch in dataloader:
    # 1. Rollout
    rollouts = []
    for query in batch:
      for _ in range(num_rollouts):
        result = run_interleaved(model, processor, audio_path, question, choices,
                                  gold_answer=answer, ...)  # Strategy B
        rollouts.append((result, answer))
    
    # 2. Compute rewards
    rewards = []
    for result, gt in rollouts:
      meta = build_meta(result)
      rew = rollout_reward(result["final_response"], gt, meta)
      rewards.append(rew)
    
    # 3. Compute token log-probs (re-forward)
    for result in rollouts:
      input_ids = ...  # tokenize prompt + completion
      policy_logps = compute_logps(model, input_ids)  # policy model
      ref_logps = compute_logps(ref_model, input_ids)   # ref model
    
    # 4. GRPO loss
    advantages = normalize(rewards)  # per-group normalization
    loss = grpo_loss(policy_logps, old_logps, advantages, ref_logps, beta=0.04)
    
    # 5. Update
    loss.backward()
    optimizer.step()
    old_logps = policy_logps.detach()  # update old policy for next step
    
    # 6. Log
    log_metrics(rewards, rollouts, kl, loss)
```

## 7. 关键代码参考

### log-probs 提取（参考 ms-swift grpo_trainer.py）

```python
def _get_per_token_logps(model, input_ids, attention_mask):
    """Get log-probs for each token in the completion."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    log_probs = logits.log_softmax(dim=-1)
    # 取实际生成的 token 的 log-prob
    per_token_logps = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return per_token_logps
```

### GRPO loss（参考 ms-swift/trl GRPOTrainer）

```python
def compute_grpo_loss(per_token_logps, old_per_token_logps, advantages, 
                       ref_per_token_logps=None, beta=0.04, epsilon=0.2):
    # ratio = exp(log_probs_policy - log_probs_old)
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    
    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    pg_loss = -torch.min(surr1, surr2)
    
    # KL penalty
    if ref_per_token_logps is not None:
        kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        pg_loss = pg_loss + beta * kl
    
    return pg_loss.mean()
```

## 8. 已实现的文件

| 文件 | 用途 |
|------|------|
| `scripts/04_grpo_smoke/grpo_utils.py` | GRPO loss、KL 估计、advantage 计算、metadata 构建 |
| `scripts/04_grpo_smoke/grpo_smoke_train.py` | 主训练循环（加载→rollout→reward→GRPO→log） |
| `scripts/04_grpo_smoke/submit_grpo_smoke.sh` | SLURM 提交脚本（A800Z, 1GPU, 80G） |

### 实现关键决策

**log-prob 近似**: 当前在 text-only 输入上计算 log-probs（不含 audio features）。这是因为 interleaved 推理中每轮的 audio context 不同，无法在训练时精确重放。Smoke 阶段用 text-only 近似足以验证训练流程。

**模型加载**: SFT checkpoint merge → unload → 叠加 trainable LoRA。参考模型用同样 checkpoint 但 frozen。

**KL 估计**: Schulman et al. 近似 `KL ≈ exp(log q - log p) - (log q - log p) - 1`，per-token 计算。

## 9. 当前状态（2026-05-10）

| 项目 | 状态 | 说明 |
|------|------|------|
| PEFT forward 错误 | ✅ 已修复 | `merge_and_unload()` → `get_peft_model()` 导致 `_forward_unimplemented` 错误。改用直接 `PeftModel.from_pretrained(base, adapter_path, is_trainable=True)` |
| 策略模型加载 | ✅ 成功 | 20,185,088 / 8,951,998,976 trainable params (0.23%) |
| Reference 模型加载 | ✅ 成功 | 冻结 eval 模式，与 policy 同 checkpoint |
| Job 41759 | ⏳ 运行中 | 1 GPU (node42, GPU 7), 80G 显存 |
| 训练进度 | ⏳ Step 0/11 | 44 samples, batch=4, 4 rollouts/query, 1 epoch |
| Rollout 模式 | 观察中 | 全 rollout 均为 1 unique seg → duplicate → finalize，pred=10% (start_percentage 问题) |
| 日志 | ⏳ 持续写入 | TensorBoard + JSONL (`output/grpo_smoke/logs/`) |

## 10. v9b-2epoch batch eval（2026-05-10）

### 结果（Job 41788, 20 samples, 重复 seg 触发 finalize 版本）
| 指标 | 值 |
|------|-----|
| 准确率 | 7/20 (35.0%) |
| 有解析出答案 | 10/20 (50%) |
| Unique segs = 1 | 20/20 (100%) |
| 平均 round | 2.0 |

**核心发现**: 所有样本都是 Round 1 找到 1 seg → Round 2 重复引用 → 触发 finalize。模型不会自然探索多个 seg。

### 推论
v9b-2epoch checkpoint 在 interleaved 推理下的 base behavior 是"找一个 seg 就够了"。这不是模型错误，而是 SFT 数据中没有多 seg 推理的示例。GRPO 需要从这种 base behavior 开始学习探索。

## 11. 推理流程修正（2026-05-10）

### 问题
即使 reward 不惩罚重复 seg，推理流程本身在检测到重复时也会中断并强制 finalize。这导致模型在 Round 2 重复引用 seg 时直接被截断，没有机会继续推理。

### 修改
1. `on_duplicate_seg` 默认值从 `"stop"` → `"ignore_continue"`: 重复 seg 不插入音频但继续生成
2. Point 2 逻辑（response 中 seg 全部已处理过）: 不再 `break` + finalize，而是记录 round，继续下一轮
3. Finalize 只在 `max_rounds` 用完且无答案时触发（安全网）

### 效果
模型现在可以在 Round 2+ 继续推理而不会被中断。Job 41790 正在运行验证。

## 12. Reward 设计修正（2026-05-10）

### 问题
模型在推理中重复引用同一个 audio segment 被 `duplicate_penalty` 扣分。但分析 Round 2 对话上下文发现，模型的行为是：Round 1 找到一个 seg 并被 SegStoppingCriteria 打断插入音频 → Round 2 自然地在推理中再次引用刚听过的内容。这是**用证据做推理的正常行为**，不应被惩罚。

### 修改
`duplicate_penalty` 默认值从 `-0.10` 改为 `0.0`（neutral）。模型不再因重复引用已用 seg 而被扣分。

### 当前 reward 设计
| 项 | 值 | 说明 |
|----|-----|------|
| `unique_segment_bonus` | +0.2/个 | 鼓励探索新 seg |
| `duplicate_penalty` | 0.0 | 重复引用已用 seg → neutral |
| `finalize_penalty` | -0.2 | 被强制终止 |
| `round_penalty` | -0.05 | 超轮数 / 轮数不足 |

## 13. Continue mode 三方案对比实验（2026-05-11）

### 背景
去掉 duplicate finalize 后（Job 41790），模型跑满 5 轮但 0% 准确率——模型不断重复同一个 seg，无法自然探索多 seg。核心问题是第二轮 user message 中的 continue prompt 让模型感觉是"重新开始推理"而不是"接着继续"。

### 三个方案

| 方案 | continue_mode | User 消息内容 | 原理 |
|------|-------------|-------------|------|
| silent | `silent` | `[audio]` 无文字 | 不给任何指令，让模型自然接着生成 |
| context | `context` | `[前文CoT] [audio]` | 用模型自己的推理文本做上下文锚点，无外部指令 |
| prompt3 | `prompt`（新文本） | `[audio] [新prompt]` | 改指令文字："You are still solving..." + "Continue from where you left off" |

### 运行链
```
41803 (silent) → 41813 (context) → 41814 (prompt3)
```
串联运行，共用同一张卡。结果产出在 `output/interleaved_eval/v9b_2epoch_{silent,context,prompt3}/`。

### 评估指标
- Accuracy（最终答案正确率）
- With answer rate（能解析出答案的比例）
- Avg unique segs（探索的音频段数）
- Avg rounds（用完 5 轮还是提前出答案）

## 14. Continue mode 三方案对比结果（2026-05-11）

### 汇总

| 指标 | silent | context | prompt3 |
|------|--------|---------|---------|
| **Accuracy** | **25.0%** (5/20) | 20.0% (4/20) | 20.0% (4/20) |
| Has answer | 55.0% (11/20) | 50.0% (10/20) | **65.0%** (13/20) |
| Avg segs | 1.00 | 1.00 | 1.00 |
| Avg rounds | 4.60 | 4.50 | 4.55 |
| Interleaved | 20/20 | 20/20 | 20/20 |
| 0 seg / 1 seg / >1 seg | 0 / 20 / 0 | 0 / 20 / 0 | 0 / 20 / 0 |

### 核心结论

**三个方案几乎没有区别。** 不论 continue_mode 如何，模型始终只生成 1 个 unique seg（Round 1 找到的第 1 个 seg），后续轮次全部重复引用同一个 seg 直到 max_rounds。

### 原因分析

这不是 prompt 工程问题，而是 SFT checkpoint 的 base behavior：
- v9b-2epoch checkpoint 在 interleaved 推理下的默认行为是"找到一个 seg 就够了"
- SFT 数据中没有多 seg 推理的示例，模型没有学会探索多个音频段
- 无论 silent/context/prompt3 如何包装新轮次的用户消息，模型在 Round 2+ 的行为完全一致：重复已用 seg

### 对比 baseline

| 版本 | 准确率 | 说明 |
|------|--------|------|
| 原始 finalize（强制截断） | 35.0% | 强制出答案，准确率最高但缺少多 seg 探索 |
| silent | 25.0% | 允许继续推理但模型只会重复 |
| context | 20.0% | 同 |
| prompt3 | 20.0% | 同 |

**推论**: 强制 finalize 虽然准确率高，但只是"逼模型猜答案"，没有真正的多步推理。GRPO 训练是改变 base behavior 的必要路径——需要让模型从 reward 信号中学会"探索多 seg → 更高准确率"。

## 15. 自定义生成循环（2026-05-11）

### 状态
✅ 已实现 `scripts/interleaved_infer_custom.py`，与原有逻辑完全隔离。

### 原理
用 token-by-token 的 KV cache 续写替代每轮 `model.generate()`：
- Round 1: `thinker.prefill()` → KV cache → 逐 token 解码，检测 `</seg>`
- Round 2+: 编码 segment 音频 → `thinker.get_audio_features()` → 构建 `[<|audio_bos|> <AUDIO>×N <|audio_eos|> continue_prompt]` → 插入 KV cache → 继续解码

### 已验证的关键能力
- `thinker.get_audio_features()` 独立编码任意音频段 ✅
- `thinker(input_ids, past_key_values, input_features, ...)` 带 KV cache 的 mini-prefill ✅
- KV cache 续写后继续解码 ✅
- 逐 token 检测 `</seg>` / `</answer>` 停止信号 ✅

### 性能收益
- 每轮只处理新增的 ~50-100 tokens（音频嵌入 + 继续 prompt），而不是全量历史对话
- 5 轮对话预期 ~3-4 倍加速

### 剩余风险
- `position_ids` / `cache_position` 未显式传递，RoPE 位置可能不精确（但在短序列上影响不大）
- `continue_mode` 暂时只支持 "prompt"（silent/context 需要不同的 KV cache 插入策略）

## 16. 自定义循环 Job 41825 测试结果（2026-05-11）

### 结果
| 指标 | 值 |
|------|-----|
| Job | 41825, TIMEOUT 8:00:16 |
| 完成 | 9/43 条 |
| 平均速度 | **~53 min/条** |
| 生成 `<seg>` | **0/9** |
| 保存 results.json | ❌（超时在最终写入前） |

### 核心问题

**性能 bug**: `_decode_one()` 每步 ~30s，KV cache 可能未生效，逐 token decode 在重复计算全部历史。

**Seg 生成**: 旧版自定义循环的逐 token 解码 + `skip_special_tokens=False` 的检测方式与原始 `SegStoppingCriteria` 行为不一致，导致模型从不生成 `<seg>` 就直接出 `</answer>` 或空响应。

### 修复（2026-05-11）

1. **性能**: 删除 `_decode_one()` 的 Python for 循环，改用 `thinker.generate()` + `SegAnswerStoppingCriteria`（与原始 `interleaved_infer.py` 一致的 stopping criteria）
2. **三个 decode 路径全部替换**:
   - Round 1: `thinker.forward()` prefill → `thinker.generate()` decode
   - Round 2+: `_insert_and_prefill()` KV cache 追加 → `thinker.generate()` decode
   - Finalization: `thinker.generate()` decode

## 17. `--continue_mode assistant_append`（2026-05-11）

### 原理
论文 `x ← x ⊕ ô ⊕ A_s:e` 要求将音频嵌入**直接追加到序列末尾**，而不是作为新 user turn。在 Qwen2.5-Omni 中，processor 不支持在 assistant 消息里放 audio，只能通过 token/embedding 层操作。

### 实现
Round 2+ 插入时，`continue_mode="assistant_append"` 只插入 `<|audio_bos|> <AUDIO>×N <|audio_eos|>`，不加任何 `build_continue_prompt()` 文字。这与 KV cache 追加的机制一致，区别仅在于继续 prompt 的有无。

### 三种 Round 2 策略对比

| 模式 | KV cache 追加内容 | 效果 |
|------|-------------------|------|
| `prompt` | audio + "Continue your reasoning..." | 新 user turn，打断推理流 |
| `silent` | audio（无文字） | 同 `assistant_append`（自定义循环中） |
| `assistant_append` | audio（无文字，直接续在 KV cache 末尾） | 符合论文 `x ← x ⊕ ô ⊕ A_s:e` |

## 18. 先不做的（后续扩展）

- ❌ vLLM 加速生成（smoke 不需要）
- ❌ 多 GPU / DeepSpeed（1 GPU smoke）
- ❌ verl 集成（后续 Stage 2 用）
- ❌ Reward model（直接用 rollout_reward）
- ❌ 分布式 KL 估计（单机够用）

## 19. 自定义循环性能问题分析（2026-05-11）

### 问题
自定义 KV-cache 循环的 token-by-token decode（`_decode_loop`）每次 `thinker()` 调用耗时 ~30s，导致每 sample 需要 ~53 分钟。

### 根因
Qwen2.5-Omni Thinker 模型的 direct `thinker()` forward 路径在 token-by-token decode 时可能存在以下问题：
- `cache_position` 未显式传递时，position_ids 计算路径与 `model.generate()` 内部的优化路径不同
- `_update_causal_mask` 在 direct forward 调用中会重新创建 4D causal mask（即使 attention_mask 正确传递）
- `model.generate()` 内部有大量 CUDA graph / kernel 优化，direct `thinker()` 调用没有这些优化
- 每次 Python 层 `thinker()` 调用的 kernel launch overhead 在 28 层 transformer 上被放大

### 解决方案
放弃 token-by-token decode，改用 `model.generate()` 做每个 round 的文本生成：

```
Round 1: model.generate() with full audio + prompt (proven fast: ~10-30 tok/s)
Round 2+: 重建完整对话（含 full audio + assistant response + seg audio + continue prompt）
          → processor 处理所有 audios → model.generate()
Finalization: 同上，用 finalize prompt
```

**优点**: `model.generate()` 已验证稳定工作，速度快（单次生成 ~10-30 tok/s）
**代价**: 每轮重新处理完整音频（~2-3s/轮），5 轮约 ~50s/sample，仍远快于 53 min/sample

### KV cache 的价值保留
KV cache 插入（`_insert_and_prefill`）已验证可以正确工作。后续如果需要优化，可以：
1. Round 1：`model.generate()` 正常生成，保存 past_key_values
2. Round 2+：`_insert_and_prefill` 插入 seg audio 到 KV cache → `model.generate(past_key_values=...)` 从 cache 生成
3. 需要验证 `model.generate()` with `past_key_values` 是否稳定工作

### 当前状态（2026-05-11）
| 项目 | 状态 | 说明 |
|------|------|------|
| `_decode_loop` (token-by-token) | ❌ 放弃 | ~30s/token，不可用 |
| `model.generate()` + conversation reconstruction | ⏳ 待测试 | 预计 ~50s/sample |
| KV cache audio insert + `model.generate()` | ⏳ 待验证 | 需要测试 past_key_values 兼容性 |
| `_insert_and_prefill` | ✅ 已验证 | thinker() mini-prefill with audio features 正确工作 |

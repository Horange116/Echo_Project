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
| `reward/total` | float | R(τ) 总 reward 均值 |
| `reward/format` | float | Rformat (0.5) 均值 |
| `reward/consistency` | float | Rconsist (0~-0.5) 均值 |
| `reward/accuracy` | float | Racc (0.5) 均值 |
| `reward/segment` | float | Rseg (0.5) 均值 |
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

## 12. Reward 设计修正 — 对齐论文（2026-05-12）

### 背景
原有 reward 设计（unique_segment_bonus +0.2/个、duplicate_penalty、finalize_penalty -0.2、round_penalty -0.05）与论文的标准化 reward 结构不匹配。模型"重复 seg"不一定是错误行为（可能是正常引用已用证据），原有的分段奖惩不利于引导"用 seg 推理来答对"这个核心目标。

### 论文 reward 结构

论文总 reward 公式：

```
R(τ) = Rformat(τ) + Rconsist(τ) + Racc(τ) + Rseg(τ)
```

| 组件 | 分值 | 说明 |
|------|------|------|
| Rformat | +0.5 | 正确使用封装标签（`<think>`/`<answer>`/`<seg>`）|
| Rconsist | -0.1/次, 最多 -0.5 | `</seg>` 后首个文本 token 是大写或 `<` 时扣分 |
| Racc | +0.5 | 答案匹配 ground truth |
| Rseg | +0.5 | 至少引用 1 个 segment **且**答案正确；否则 0 |

总范围: [-0.5, 1.5]

### 修改后的 reward 设计

```python
def compute_reward(response, gt_answer, seg_count, has_correct_answer):
    """论文对齐的 reward 计算."""
    reward = 0.0
    
    # Rformat: 标签结构奖励 (0.5)
    has_think = "<think>" in response and "</think>" in response
    has_answer_tag = "<answer>" in response and "</answer>" in response
    has_seg = "<seg>" in response
    if has_think and has_answer_tag:
        reward += 0.5  # Rformat
    
    # Rconsist: 连续性奖励 (0 ~ -0.5)
    consist_penalty = 0.0
    segments_end = [m.end() for m in re.finditer(r"</seg>", response)]
    for pos in segments_end:
        # 找到 </seg> 后第一个非空格、非audio token的字符
        rest = response[pos:]
        # 跳过 <|audio_bos|> ... <|audio_eos|> 这类环境注入 token
        rest_clean = re.sub(r"<\|audio_bos\|>.*?<\|audio_eos\|>", "", rest).strip()
        if not rest_clean:
            continue
        next_char = rest_clean[0]
        if next_char.isupper() or next_char == "<":
            consist_penalty -= 0.1
    reward += max(consist_penalty, -0.5)  # Rconsist
    
    # Racc: 准确率奖励 (0.5)
    if has_correct_answer:
        reward += 0.5  # Racc
    
    # Rseg: segment 奖励 (0.5, 条件性)
    if seg_count >= 1 and has_correct_answer:
        reward += 0.5  # Rseg
    
    return reward
```

### Rollout 行为惩罚（可选，保留为非 reward 的辅助信号）

论文中没有的 rollout 行为惩罚，移出 reward 主逻辑，改为辅助日志监控：

| 项 | 处理方式 | 说明 |
|----|---------|------|
| duplicate_seg | 监控计数，不扣分 | 重复引用已用 seg 是正常推理行为 |
| finalize | 让 Rformat/Racc 自然惩罚 | 被 finalize 的 response 通常没有规范标签或答案 |
| round_penalty | 移除 | 模型应该在用 seg 答对时获正奖励，不因轮数扣分 |

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

## 20. Reward 对齐论文实现完成（2026-05-12）

### 改动文件

| 文件 | 改动 |
|------|------|
| `echo_rl/rollout_rewards.py` | 移除旧 penalty 系统（duplicate_penalty, finalize_penalty, unique_segment_bonus, round_penalty），改为论文对齐的 `total_reward` 纯包装 |
| `scripts/04_grpo_smoke/grpo_smoke_train.py` | 移除 5 个旧 CLI args、移除 `reward_coef` 构造块、移除 `coef=reward_coef` 参数传递；`parse_rollout_metrics` 去掉旧惩罚字段；`log_metrics` 去掉旧 rollout 惩罚日志，控制台输出改为 `R (fmt cst acc seg)` 格式 |

### 当前 reward pipeline

```
rollout_reward(response, gt_answer)
  └─ total_reward(response, gt_answer)
       ├─ r_format(response)     → +0.5 max
       ├─ r_consist(response)    → 0 ~ -0.5
       ├─ r_acc(response, ans)   → +0.5 conditional
       └─ r_seg(response, ans)   → +0.5 conditional
  └─ rollout_total = round(total, 4)
```

### GRPO 流程（更新后）

```
Phase 1: run_interleaved() → G rollouts (对话重建每轮)
Phase 2: rollout_reward()  → paper reward per rollout
Phase 3: text-only log-probs (old policy + reference)
Phase 4: group-normalise advantages
Phase 5: GRPO loss (clipped surrogate + KL) → backward → update
Phase 6: log paper-aligned metrics
```

### 注意事项
- Rconsist 当前基于纯文本近似（response 中无音频 token）
- 切换到完整多模态 GRPO（verl）时，需在检查 `</seg>` 后下一个文本 token 前先跳过音频 token

## 21. 对话重建修正 + 数据扩增（2026-05-12）

### 问题
Job 41921 输出确认 Round 2+ 的 `build_conversation` 使用了多轮次对话结构，导致 `<|im_end|>` turn boundary 让模型把 Round 2 理解为"新问题"而非"继续推理"。所有 32 个 rollout 全部是 Round 1 产生 1 seg → Round 2 重复 seg → ... → 空转 5 轮。

### 修改
`build_conversation()` 重构：Round 2+ 不再创建独立 assistant + user 消息，而是把完整上下文拼成**单条 user 消息**：

```
Round 1:  user: [全音频A, prompt]
          asst: [模型生成, 在</seg>截断]

Round 2+: user: [全音频A, "原始问题\\n上一轮回答", 裁剪音频A_s:e]  ← 单消息, 无turn boundary
          asst: [模型续写]
```

同时：
- Round 2+ 重复 seg 触发的 `on_duplicate_seg="stop"` 现已正确截停进入 finalization（之前"已处理过"路径缺少 stop 逻辑）
- `max_rounds` 设为 10（安全阀），实际轮次取决于模型行为
- `max_new_tokens` 提升到 256

### 效果验证（Job 41922）
| 指标 | 旧格式 (41921) | 新格式 (41922) |
|------|:-:|:-:|
| 平均耗时 | ~37s/rollout | **~12.5s/rollout** |
| 平均轮数 | 5 | **2.1** |
| 重复 seg 截停 | ❌ 空转 5 轮 | ✅ 1 轮新 seg 后直接 finalize |

### 数据扩增
RL 数据集从 44 条扩至 **129 条**（3x），重新从 `judged_subset_600_by_source_type.jsonl` 按类型均衡采样。

### 当前训练配置（Job 41924）
- 数据: 129 条（3 条音频缺失）
- batch_size=4, num_rollouts=8, num_epochs=2
- max_rounds=10, max_new_tokens=256, finalize_max_new_tokens=64
- 预估: ~7 小时

## 22. CUDA driver error: invalid argument — Phase 3/5 thinker 崩溃（2026-05-12）

### 问题
GRPO 训练的 Phase 3（text-only log-prob 计算）和 Phase 5（有 grad 的 policy forward）中，调用 `model.get_base_model().thinker()` 做 text-only forward 时触发 `RuntimeError: CUDA driver error: invalid argument`。错误位置在 LoRA 线性层 `F.linear(input, self.weight, self.bias)`，每次重现位置不同但都在 14+ 次 forward 后。

### 崩溃记录

| Job | Phase | 崩溃位置 | 修复尝试 | 结果 |
|-----|-------|---------|---------|------|
| 41924 | Phase 3 | 首次 old_logps 调用 | — | ❌ |
| 41925 | Phase 3 | 首次 old_logps 调用 | 加 `torch.cuda.synchronize()` | ❌ |
| 41926 | Phase 5 | pair 0（首次） | ref_model 常驻 GPU | ❌（Phase 3 通过，Phase 5 在 pair 0 崩溃） |
| 41927 | Phase 5 | pair 11 | `train()` 模式贯穿 Phase 3-5 | ❌（Phase 3 64 次 forward 全通，Phase 5 pair 11 崩溃） |
| 41928 | Phase 5 | pair 14 | LoRA dtype 从 float32→float16 | ❌（dtype 确认修复，Phase 5 继续崩溃于 pair 14） |
| 41929 | Phase 3+5 合并 | pair 14 | 合并为单循环 + train 模式 | ❌（pair 14 仍崩溃） |
| 41930 | Phase 3+5 合并（限 8 pair） | ✅ **成功** | 仅处理 8 个 pair | ✅ Phase 5 forward 通过 |
| 41930 | Phase 3+5 合并（限 8 pair） | compute_grpo_loss | shape 不匹配（调试限制导致） | ❌（非 CUDA 错误） |
| 41931 | Phase 3+5 合并（32 pair） | pair 14 | 同上 | ❌（确定是累计调用次数问题） |

### 确定性结论
1. **LoRA dtype float32 ↔ base model float16 不匹配** — ✅ 已修复。诊断确认修复后 LoRA 参数为 float16
2. **14+ 次 `thinker()` forward（有 grad）后 CUDA 状态损坏** — 每次 Phase 3（`no_grad`）都能跑完全部 64 次 forward，但 Phase 5（有 grad）或合并循环（有 grad）总是在第 14 次附近崩溃
3. **Phase 3 在 `no_grad` 下 64 次全部成功** — 说明 `no_grad` 模式下 thinker forward 路径不同，不会触发该错误
4. **8 次有 grad 的 forward 全部通过**（Job 41930）— 确认限制在约 14 次

### 尝试过的修复（均无效）
- `torch.cuda.synchronize()` + `empty_cache()` 清理
- `gc.collect()` 回收 Python 对象
- ref_model 常驻 GPU（消除 Phase 1→3 的大块内存重分配）
- `train()` 模式贯穿 Phase 3-5（避免 eval→train 切换）
- LoRA 参数统一为 float16
- Phase 3+5 合并为单次循环（avoid no_grad→grad transition）

### 根因推测
`model.get_base_model().thinker()` 直接在 LoRA-monkey-patched 的子模块上做带 grad tracking 的 forward。每次调用构造 autograd 图，不同序列长度导致 CUDA 反复分配/释放不同大小的 tensor，约 14 次后 CUDA 内存分配器状态损坏，触发 `invalid argument`。

该错误仅在 `train()` + 有 grad 模式下出现，`no_grad()` 或 `eval()` 模式下不出现。

### 待验证的修复方向
1. **Batch 所有序列为单次 forward**（代码已改好，未运行）：32 个 rollout 的 text 拼成一个 (32, max_T) 的 padded batch，只调用 1 次 `thinker()`（有 grad）+ 1 次 ref（no_grad），彻底避免累计调用问题
2. **改用 `model.generate()` 的 byproduct 获取 log-probs**：不在 Phase 3+5 重新 forward，而是从 rollout 生成过程中保存每步 logits
3. **直接操作 timeline：不涉及thinker，用 rollout generation 的过程变量计算 log-probs**

## 23. GRPO per-token log-prob 必要性分析（2026-05-12）

### 核心结论

GRPO 训练无法绕过对 completion 的重新 forward，原因：

```
GRPO loss:  -min(ratio × adv, clip(ratio) × adv) + β × KL

ratio(token_t) = exp(log πθ(token_t) - log πold(token_t))
KL(token_t)    = exp(log πref - log πθ) - (log πref - log πθ) - 1
```

- **advantage 是 rollout-level 共享的**（同一个 group 内所有 rollout 标准化后的 z-score 作用到每个 token）
- **ratio 是 per-token 的**（每个 token 在 πθ 和 πold 下的概率比不同），所以 per-token log-prob 仍然必要
- **KL 是 per-token 的**，需要 πθ 和 πref 每个位置的 log-prob
- 不能简化为 sequence-level ratio（会变成另一个算法，长序列数值不稳定）

### 三个 log-prob 来源

| 类型 | 用途 | 能否从 generate 获取 | 是否必须 forward |
|------|------|:---:|:---:|
| πold (old policy logps) | ratio 分母 | ✅ `output_scores=True` 可获取 | 可省 |
| πθ (current policy logps) | ratio 分子，需要梯度 | ❌ generate 不提供梯度 | ✅ 必须 |
| πref (reference logps) | KL 惩罚 | ❌ 不同模型 | ✅ 必须 |

### audio token mask

插入的 seg 音频 token（`<|audio_bos|>...<|audio_eos|>`）不是模型生成的，在 GRPO loss 中必须用 `completion_mask` 排除，否则模型会对环境注入的 token 产生梯度更新。

### 解决方案

**Batched forward**（当前实现）：32 个 rollout 的 prompt+completion 序列拼成 (32, max_T) padded batch，πθ forward 1 次（有 grad），πref forward 1 次（no grad），πold = πθ.detach()。总共 2 次 `thinker()` 调用替代原来的 64 次。


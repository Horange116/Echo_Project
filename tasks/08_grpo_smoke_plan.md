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

## 9. 先不做的（后续扩展）

- ❌ vLLM 加速生成（smoke 不需要）
- ❌ 多 GPU / DeepSpeed（1 GPU smoke）
- ❌ verl 集成（后续 Stage 2 用）
- ❌ Reward model（直接用 rollout_reward）
- ❌ 分布式 KL 估计（单机够用）

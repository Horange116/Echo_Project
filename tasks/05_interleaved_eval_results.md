# Interleaved Eval 冒烟测试 — v9b-diverse-cot-2epoch

日期: 2026-05-09

## 测试配置

- **模型**: Qwen2.5-Omni-7B
- **Adapter**: v9b-diverse-cot-2epoch (checkpoint-3078)
- **数据**: eval_manifest_500.jsonl 前 20 条
- **参数**: max_rounds=5, max_new_tokens=128, temperature=0.7

## 测试结果对比

### 基本版（原始脚本 `batch_interleaved_smoke.py`，无 duplicate guard）

| 指标 | 值 |
|------|-----|
| 正确率 | 0/20 (0%) |
| 平均 round | 4.7 |
| 平均 seg/样本 | 4.7 |
| Answer 输出 | 1/20 (5%) — 仅 1 样本有 answer |
| 多轮行为 | ✅ 模型可以连续生成不同 seg（最多 5 轮） |
| 核心问题 | 模型永不主动输出 `</answer>`，跑到 max_rounds 耗尽 |

### 完整版（`batch_interleaved_eval.py`，带 duplicate guard + finalize）

| 指标 | 值 |
|------|-----|
| 正确率 | 6/20 (30%) |
| 平均 round | 2 (均在 round 2 被 duplicate guard 截停) |
| 平均 seg/样本 | 1.0 (仅 1 个 unique seg) |
| Answer 输出 | 11/20 (55%) — finalize 成功 |
| 多轮行为 | ❌ duplicate guard 在 round 2 截停，无多轮 |
| 核心问题 | duplicate guard 过于激进，消除多轮可能性 |

### 关键洞察

两个版本有相反的问题：

```
基本版: 多轮 seg ✓ → answer ✗ (跑满 5 轮也不停)
完整版: 多轮 seg ✗ → answer ✓ (round 2 被截停，但 finalize 强出答案)
```

**根因**: v9b-diverse-cot-2epoch 在 SFT 训练时是一次性生成 `<seg>...<seg>...</answer>`。换成 interleaved 逐轮交互后，模型不知道何时该停止生成 seg、输出答案。

## 按类型准确率（完整版）

| 类型 | 样本数 | 正确 | 准确率 |
|------|--------|------|--------|
| repeated_event_gap | 2 | 2 | 100% |
| duration_compare | 1 | 1 | 100% |
| count_before | 2 | 1 | 50% |
| duration_percentage | 4 | 1 | 25% |
| start_percentage | 6 | 1 | 17% |
| overlap | 4 | 0 | 0% |
| gap | 1 | 0 | 0% |

## Segment 质量

- 所有 20 样本均成功检测并裁剪 `<seg>`（20/20 = 100%）
- 所有裁剪文件均存在于磁盘（20/20 = 100%）
- 解析错误: 0

## 输出文件

- `output/interleaved_eval/v9b_2epoch_smoke20.jsonl` — 完整版逐样本结果
- `output/interleaved_eval/v9b_2epoch_smoke20_summary.json` — 完整版汇总
- `output/interleaved_eval/original_smoke20/batch_result.json` — 基本版结果

## 后续建议

1. **Short-term**: 用完整版+放宽 duplicate guard（`max_duplicate_segments=2` 或 `on_duplicate_seg=ignore_continue`），在保留多轮的同时用 finalize 兜底
2. **Medium-term**: SFT 阶段加入模拟逐轮交互的训练数据，让模型学会"生成 seg → 暂停 → 听音频 → 继续 → 最终输出 answer"的行为
3. **Long-term**: GRPO 训练中通过 reward 引导 interleaved 行为（`r_seg` reward 鼓励多轮 seg 使用，`r_format` 确保 answer 格式）

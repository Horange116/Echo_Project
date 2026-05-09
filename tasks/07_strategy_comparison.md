# Interleaved 策略 A/B/C/D 对比 — v9b-diverse-cot-2epoch

日期: 2026-05-09

## 策略定义

| 策略 | on_duplicate_seg | finalize_on_stop | 说明 |
|------|-----------------|-----------------|------|
| **A** | ignore_continue | False | 无 guard，无 finalize（≈原版 basic） |
| **B** | stop | True | 遇 dup 即停 + finalize（当前默认） |
| **C** | ignore_continue | True | 忽略 dup + finalize 兜底 |
| **D** | insert_once_continue | True | 每 seg 只插入一次 + finalize 兜底 |

## 测试配置

- **模型**: Qwen2.5-Omni-7B + v9b-diverse-cot-2epoch (checkpoint-3078)
- **数据**: eval_manifest_500.jsonl 前 20 条（有序）
- **参数**: max_rounds=5, temperature=0.7, max_new_tokens=128

## 核心结果

| 策略 | 准确率 | Has answer | Finalized | Avg rounds | Avg segs | Base avg | Rollout avg | Delta |
|------|--------|-----------|-----------|------------|----------|----------|-------------|-------|
| **A** | 0% (0/20) | 0/20 | 0/20 | 2.1 | 1.0 | -0.105 | -0.005 | +0.100 |
| **B** | **35% (7/20)** | 10/20 | 10/20 | 2.1 | 1.0 | +0.310 | +0.310 | +0.000 |
| **C** | 25% (5/20) | 8/20 | 8/20 | 2.1 | 1.0 | +0.235 | +0.255 | +0.020 |
| **D** | 15% (3/20) | 7/20 | 7/20 | 2.0 | 1.0 | +0.077 | +0.107 | +0.030 |

### Rollout_reward 组件分解

| 组件 | A | B | C | D |
|------|---|---|---|---|
| format | +0.000 | +0.150 | +0.150 | +0.113 |
| consistency | -0.105 | -0.190 | -0.165 | -0.185 |
| accuracy | +0.000 | **+0.175** | +0.125 | +0.075 |
| segment | +0.000 | **+0.175** | +0.125 | +0.075 |
| duplicate_penalty | +0.000 | +0.000 | +0.000 | +0.000 |
| round_penalty | +0.000 | +0.000 | +0.000 | +0.000 |
| finalize_penalty | +0.000 | **-0.100** | -0.080 | -0.070 |
| unique_segment_bonus | +0.100 | +0.100 | +0.100 | +0.100 |

## 关键发现

### 1. 模型不具备多轮 seg 能力
所有 4 种策略都只产生 **1 unique seg/样本**（平均）。v9b-diverse-cot-2epoch 是单轮 SFT checkpoint，在 interleaved 场景下：
- Round 1: 正常生成 `<seg>`
- Round 2+: 重复生成相同 seg（或生成无 seg 的文本）
- 无法根据已插入的音频段产生新的 seg

### 2. Finalize 是准确率的关键
- 有 finalize 的 3 个策略（B/C/D）都产出了 answer（7-10/20）
- 无 finalize 的 A 产出 0 answer → 0% 准确率
- Finalize_penalty（-0.20/次）虽然降低了 rollout_total，但换来准确率大幅提升

### 3. 激进截停策略最优
**B (stop+finalize) 综合最优**：35% 准确率，rollout_total +0.310。原因是：
- 模型 Round 2 就产生重复 seg → 仅 1 seg，但数据量也足够 single-seg 样本答题
- 过早截停反而节省计算量，不影响准确率
- C/D 等待更多轮次但没有产生新 seg，徒增计算

### 4. Rollout_reward 评价
- `unique_segment_bonus` 反映了 seg 质量（但所有策略相同，因模型无法多轮）
- `finalize_penalty` + `accuracy`/`segment` 共同决定了策略排序
- `duplicate_penalty` 和 `round_penalty` 在当前场景未触发

### 5. 训练需求确认
GRPO 训练是解决多轮 seg 问题的必要条件。reward 设计可以：
- `unique_segment_bonus` 激励模型在不同轮次生成不同 seg
- `duplicate_penalty` 抑制重复 seg 行为
- `finalize_penalty` 作为 finalize 的备用出口，但要抑制滥用
- `round_penalty` 鼓励在合理轮数内完成（2-5 轮）

## 输出文件

- `output/interleaved_eval/strategy_compare/{A,B,C,D}_*_smoke20.jsonl` — 各策略逐样本结果
- `output/interleaved_eval/strategy_compare/{A,B,C,D}_*_smoke20_summary.json` — 各策略摘要
- `output/interleaved_eval/strategy_compare/compare_strategies.py` — 对比分析脚本
- `scripts/03_interleaved/batch_interleaved_eval.py` — 更新后 batch 脚本
- `scripts/03_interleaved/submit_strategy_comparison.sh` — 提交脚本

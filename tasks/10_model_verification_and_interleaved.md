# 模型验证 + Interleaved 策略扩展测试

## Base 模型官方版本验证

### 发现
从 ModelScope 重新下载的 `Qwen/Qwen2.5-Omni-7B` 与本地已有模型 **完全一致**：
- 所有 5 个 safetensors 的 MD5 校验通过
- 所有配置文件的 JSON 结构一致
- `transformers_version: 4.50.0.dev0` 相同

### 结论
模型权重不是 MMAR 准确率差距的原因（我们 ~30% vs 论文复现 51.80%）。差距来源于评估管线差异（数据版本、prompt、音频处理等）。

## v9b2 2epoch MMAR Paper Prompt 评估（max_new_tokens=1024）

### 配置
- **模型**: Qwen2.5-Omni-7B + v9b-diverse-cot-2epoch (checkpoint-3078)
- **评估方式**: paper_appendix_E2 prompt，exact match
- **参数**: max_new_tokens=1024, batch_size=8

### 结果（1000 样本）

| 指标 | v9b2 1024t | Base 256t | v9b2 256t (partial 768) |
|------|-----------|-----------|------------------------|
| strict_acc | **28.8%** | 30.3% | 32.6% |
| fallback_acc | **34.4%** | 32.1% | — |
| has_think_answer | 67.2% | 69.5% | — |
| has_answer_tag | 70.3% | 74.4% | 60.2% |
| has_seg | **72.2%** | 0.0% | — |
| answer_in_choices | 62.1% | 56.8% | — |

### 关键发现
1. **has_seg=72.2%**：SFT 成功教会了模型输出 `<seg>` 标签（基座为 0%）
2. strict_acc 28.8% 略低于基座 30.3%，但 fallback_acc 34.4% 高于基座 32.1%
3. 1024 token 相比 256 提升了 has_answer_tag（60.2%→70.3%），减少了截断
4. 无 answer tag 的原因：67% gibberish（与基座相同），19% 有 think 无 answer，14% 截断

## Interleaved 推理策略扩展测试

### 已有策略对比

| 策略 | 准确率 | Has answer | 说明 |
|------|--------|-----------|------|
| A (ignore, no finalize) | 0% | 0/20 | 原始版，无 guard |
| B (stop + finalize) | **35%** | 10/20 | **最优** |
| C (ignore + finalize) | 25% | 8/20 | |
| D (insert once + finalize) | 15% | 7/20 | |
| silent mode | 25% | 11/20 | 新策略 |
| context mode | 20% | 10/20 | 新策略 |
| prompt3 | 20% | 13/20 | 不同 prompt |

### 核心问题
所有策略都只产生 **1 unique seg/样本**。v9b-diverse-cot-2epoch 是单轮 SFT checkpoint，无法在 interleaved 场景下多轮生成不同 seg。GRPO 训练是解决此问题的必要条件。

### 输出文件
- `output/MMAR_eval/v9b_2epoch_paper_prompt_1024/` — v9b2 1024 token MMAR 评估结果
- `output/interleaved_eval/v9b_2epoch_silent/` — silent mode 结果
- `output/interleaved_eval/v9b_2epoch_context/` — context mode 结果
- `output/interleaved_eval/v9b_2epoch_prompt3/` — prompt3 结果
- `output/MMAR_eval/base_paper_prompt/gibberish_report.json` — gibberish 分析

# Base Model MMAR Evaluation Investigation

## 目标
调查为什么 Qwen2.5-Omni-7B 基座在 MMAR 上的准确率（~30%）远低于论文报告的 57.33%。

## 发现

### Base 模型三种评估方式对比

| 指标 | Zero-shot (tok=64) | Paper prompt (tok=256) | Simple prompt (tok=128) |
|------|-------------------|----------------------|----------------------|
| 准确率 | 33.4% (fallback) | 30.3% (strict) / 32.1% (fallback) | **24.0% (MMAR word-token)** |
| cond1 (答案词匹配) | — | 48.9% | **40.4%** |
| cond2 (无错误选项词) | — | — | 43.2% |
| has_answer_tag | 70.5% | 74.4% | — |

### Gibberish 分析（base_paper_prompt, 1000 samples）

- **总 gibberish 率：20.9%**
  - pérdida_loop: 15.8%（已知 Qwen2.5-Omni bug，某些音频触发西班牙语循环）
  - number_loop: ~3.0%（时间戳/数字循环）
  - excessive_repeat: ~1.7%（内容词重复）
  - char_repeat / dominating_word: ~0.5%
- **按模态：** sound (38.8%) >> speech/music (~15-20%)
- **按类别：** Perception Layer (26.7%) > Signal Layer (23.3%)
- **Gibberish 音频更短**（均值 12s vs 正常 22s）

### 关键结论

1. 基座真实的 MMAR 音频理解能力 ≈ **cond1 40.4%**
2. MMAR 官方 word-token 匹配法 (cond1+cond2) 只有 24%，因为 cond2 被多选选项共享常见词拖累
3. **论文 57.33% 是 SFT + RL 后的水平**，论文 Table 1 中 SFT-only ≈ 43-45%，SFT+RL ≈ 57%
4. 基座 30-40% 是合理 baseline，差距不需要担心

### 已提交的 Job

- Job 41754: v9b2 paper prompt, max_new_tokens=1024（等 node42 空闲后运行）

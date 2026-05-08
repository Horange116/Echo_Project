# Audio-Interleaved 推理（Echo 论文机制）

日期: 2026-05-08（v2 主链路验证通过）

## 背景

Echo 论文的核心机制之一：模型在推理过程中可以**分段听音频**——先听完整音频做初步推理，遇到不确定的部分用 `<seg>start, end</seg>` 请求重新听特定片段，系统裁剪对应音频插回上下文，模型继续推理，直到给出最终答案。

对应代码: [scripts/interleaved_infer.py](../scripts/interleaved_infer.py)

## 整体流程

```
Round 1: [完整音频 + 问题] → 生成（带 SegStoppingCriteria）
  ├─ 检测到 </seg> → 停止生成 → 裁剪音频 → 插入 → Round 2
  └─ 检测到 </answer> → 结束（无 seg 直接回答）

Round 2: [完整音频 + 历史 + 裁剪片段 + 续听提示] → 生成
  ├─ 检测到 </seg>（新 seg） → 裁剪 → 插入 → Round 3
  ├─ 检测到 </seg>（已处理过的重复 seg） → duplicate_seg → Finalize
  └─ 检测到 </answer> → 结束

...（最多 max_rounds 轮，但重复 seg 提前结束）

Finalization: 当 stop_reason ∈ {duplicate_seg, max_rounds} 且从未有过 answer
  → 用 finalize prompt 让模型直接出 <answer>
```

## 关键组件

### 1. SegStoppingCriteria（2026-05-08 修复）

HuggingFace `StoppingCriteria` 自定义钩子，在模型生成**每个 token 后实时解码检查**：

- 优先级 1: `</seg>` 出现 → 立即停止，返回 `stop_reason="seg"`
- 优先级 2: `</answer>` 出现（且无 `</seg>`） → 立即停止，返回 `stop_reason="answer"`

这是最关键的优化。不加 stopping criteria 时，模型在同一轮生成 `<seg>...</seg>` 和 `<answer>...</answer>`，`has_answer()` 先短路导致 seg 永远不被处理。

### 2. 优先级反转（2026-05-08 修复）

循环内判断顺序从 `has_answer → extract_segs` 改为 `extract_segs → has_answer`：

```
修改前: generate → has_answer? YES → break（seg 从没被用过）
修改后: generate → extract_segs → 有 seg? → 处理 → continue（跳过 answer）
                                        → 无 seg? → has_answer? → break
```

### 3. 重复 seg 提前 finalize（2026-05-08 修复）

当 `parse_segments(response)` 返回 seg 但 `extract_latest_segments()` 返回空（全部已处理过）：

- 设置 `stop_reason = "duplicate_seg"`
- 记录 `num_duplicate_segments` 和 `duplicate_segments[]`
- `break` 出循环
- 触发 finalize round 生成最终答案

防止模型在同一个 seg 上无意义循环到 `max_rounds`。

### 4. 初始 Prompt

要求模型用 `<think>...</think><answer>...</answer>` 格式回答，思考时需要引用音频片段则用 `<seg>start, end</seg>` 标记。

### 5. 续听 Prompt

裁剪音频插入后，附带提示：
> "I have listened to the audio segment you referenced. Continue your reasoning and provide the final answer. Use <seg>start, end</seg> if you need to reference more segments."

### 6. 音频裁剪

- 用 `librosa.load()` 加载完整音频
- 解析 `<seg>start, end</seg>`
- clamp 到 `[0, duration]` 范围内
- 裁剪并保存为临时 `.wav` 文件
- 下一轮作为 user 输入插入对话上下文

### 7. 对话上下文构造

每轮对话包含：
1. `user: [完整音频 + 初始问题]` — 始终保留，提供全局上下文
2. `assistant: [历史推理文本]` — 模型之前生成的所有内容
3. `user: [裁剪音频 + 续听提示]` — 上一轮请求的音频片段（从第二轮开始）

### 8. Finalization 轮

当 interleaved 循环因 `duplicate_seg` 或 `max_rounds` 停止且从未出现 `<answer>` 时，触发 finalization：

- 构造 prompt: "Now provide the final answer using the information already analyzed. Answer in <answer>...</answer>."
- `stop_at_seg=False`（避免又停在 seg 上）
- 用 `finalize_max_new_tokens=64` 短生成

## 冒烟测试结果

### Job 41640（2026-05-08, temp=0.7, duplicate_iou_threshold=0.8, finalize_on_stop=true）

| # | sample | qa_type | trig | insert | dup | rounds | stop_reason | finalize | has_ans | pred | gold | corr |
|---|--------|---------|:----:|:------:|:---:|:------:|-------------|:--------:|:-------:|------|------|:----:|
| 1 | 01_gap | gap | ✓ | 1 | 1 | 2 | duplicate_seg | ✓ | ✓ | 0.7s | 0.1s | ✗ |
| 2 | 02_count_before | count_before | ✗ | 0 | 0 | 1 | has_answer | ✗ | ✓ | 2 | 2 | ✓ |
| 3 | 03_repeated_event_gap | repeated_event_gap | ✓ | 1 | 1 | 2 | duplicate_seg | ✓ | ✓ | 0.4s | 0.1s | ✗ |
| 4 | 04_duration_compare | duration_compare | ✓ | 1 | 1 | 2 | duplicate_seg | ✓ | ✓ | the whispering | the whispering | ✓ |
| 5 | 05_gap | gap | ✓ | 1 | 0 | 2 | has_answer | ✗ | ✓ | 0.7s | - | ✗ |

**汇总**:
- 4/5 触发 interleaved（模型生成 `<seg>`）
- 4/5 至少插入 1 个 segment（裁剪音频成功送入下一轮）
- 3 条 `duplicate_seg` 被提前拦截（不再跑满 5 轮）
- 1 条正常 `seg → insert → continue → answer`（sample 5）
- 5/5 最终都有答案
- 2/5 答案正确（sample 2 count_before = 2 ✓, sample 4 duration_compare = the whispering ✓）

### 修复前后对比

| 指标 | 修复前 (old code) | v2 (41640) |
|------|:----------------:|:----------:|
| 重复 seg 浪费轮数 | 5 (max_rounds) | **2** (duplicate_seg→finalize) |
| 重复样本耗时 | ~10s | **~5s** |
| duplicate_seg 检测 | 无此功能 | 3/5 命中 |
| finalize 触发 | 无此功能 | 3/5 |
| 最终答案覆盖率 | 部分样本空答案 | **5/5** |

## 修复记录

### 2026-05-08 v2 主链路修复

1. **`SegStoppingCriteria`**：在 `generate()` 过程中逐 token 检查，检测到 `</seg>` 立即停止，不让模型在同一轮写出 `<answer>`
2. **优先级反转**：先 `extract_latest_segments()` 再 `has_answer()`，有 seg 就 `continue` 跳过 answer 检查
3. **重复 seg 提前拦截**：`parse_segments()` 有结果但全部已处理过 → `duplicate_seg` + finalize，不跑到 max_rounds
4. **Finalize 兜底**：循环异常结束时有 finalization round 保证 `<answer>` 输出

## 当前局限

- **模型不会利用裁剪音频**：当前 SFT checkpoint 没有学习"听完片段后更新推理"的行为，容易在同个 seg 上死循环
- **幻觉时间戳**：模型可能生成超出音频时长的 seg（已有 clamp 保护不再崩溃，但模型不会自纠正）
- **插入后推理不稳定**：插入音频后模型有时能修正答案（sample 4），有时不能（sample 1、3）

## 下一步

这些是**模型训练问题**，非推理代码 bug：

1. **RL 训练（GRPO）**：给 reward 让模型学会——请求 seg → 听裁剪音频 → 更新推理 → 给出更准确的答案
2. **seg 多样性奖励**：惩罚请求相同 seg 的行为，鼓励探索不同时间段
3. **answer 及时奖励**：鼓励模型在信息足够时尽快给出答案

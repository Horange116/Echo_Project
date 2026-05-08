# Audio-Interleaved 推理（Echo 论文机制）

日期: 2026-05-08

## 背景

Echo 论文的核心机制之一：模型在推理过程中可以**分段听音频**——先听完整音频做初步推理，遇到不确定的部分用 `<seg>start, end</seg>` 请求重新听特定片段，系统裁剪对应音频插回上下文，模型继续推理，直到给出最终答案。

对应代码: [scripts/03_interleaved/interleaved_infer.py](../scripts/03_interleaved/interleaved_infer.py)

## 整体流程

```
Round 1: [完整音频 + 问题] → 模型生成文本
  ├─ 检测到 </seg> → 裁剪音频 → 进入 Round 2
  └─ 检测到 </answer> → 结束

Round 2: [完整音频 + 历史 + 裁剪片段 + 继续推理提示] → 模型生成
  ├─ 检测到 </seg> → 裁剪音频 → 进入 Round 3
  └─ 检测到 </answer> → 结束

...（最多 max_rounds 轮）
```

## 关键组件

### 1. SegStoppingCriteria

HuggingFace `StoppingCriteria` 自定义钩子，在模型生成每个 token 后实时解码检查：

- `</answer>` → 立即停止，记录 `stop_reason = "answer"`
- `</seg>` → 立即停止，记录 `stop_reason = "seg"`

这是最关键的优化。不加 stopping criteria 时，模型生成的 `<seg>` 被埋在大段文本中无法触发 interleaved 机制。

### 2. 初始 Prompt

要求模型用 `<think>...</think><answer>...</answer>` 格式回答，思考时需要引用音频片段则用 `<seg>start, end</seg>` 标记。

### 3. 续听 Prompt

裁剪音频插入后，附带提示：
> "I have listened to the audio segment you referenced. Continue your reasoning and provide the final answer. Use <seg>start, end</seg> if you need to reference more segments."

### 4. 音频裁剪

- 用 `librosa.load()` 加载完整音频
- 解析 `<seg>start, end</seg>`（支持 `s` 后缀如 `<seg>1.5s, 2.0s</seg>`）
- clamp 到 `[0, duration]` 范围内
- 裁剪并保存为临时 `.wav` 文件
- 下一轮作为 user 输入插入对话上下文

### 5. 对话上下文构造

每轮对话包含：
1. `user: [完整音频 + 初始问题]` — 始终保留，提供全局上下文
2. `assistant: [历史推理文本]` — 模型之前生成的所有内容
3. `user: [裁剪音频 + 续听提示]` — 上一轮请求的音频片段（从第二轮开始）

## 对话示例（sample 3: repeated_event_gap）

```
Round 1: 模型生成 "<think><seg>0.0, 0.245</seg>..."
         → 检测到 </seg>，裁剪 0~0.24s，保存

Round 2: 输入 [完整音频 + 历史 + 0~0.24s 片段 + 续听提示]
         模型生成 "The first alarm clock ends at 0.245s. <seg>1.449, 2.031</seg>"
         → 检测到 </seg>，裁剪 1.45~2.03s，保存

Round 3: 输入 [完整音频 + 历史 + 1.45~2.03s 片段 + 续听提示]
         模型又引用了同一段 seg（重复请求）

Round 4: 输入 [完整音频 + 历史 + 1.45~2.03s 片段 + 续听提示]
         模型生成 "<think>...gap of 1.191 seconds...</think><answer>0.7 seconds</answer>"
         → 检测到 </answer>，结束！
```

## 冒烟测试结果（5 样本）

| 样本 | 题型 | Interleaved 触发 | 最终答案 | GT | 结果 |
|------|------|:-:|:-:|:-:|:-:|
| 1 | gap | ✅ 触发但卡在同个 seg 循环 5 轮 | 空 | 0.1s | ❌ |
| 2 | count_before | ✅ 触发但未收敛 | 空 | 2 | ❌ |
| 3 | repeated_event_gap | ✅ seg→裁剪→再听→推理 | **0.7 seconds** | 0.7s | ✅ |
| 4 | duration_compare | ✅ seg→裁剪→再听→推理 | **the whispering** | the whispering | ✅ |
| 5 | gap | 直接回答（无需 seg） | 0.4 seconds | 0.7s | ❌ |

## 当前局限

- **模型不会利用裁剪音频**：SFT checkpoint 没有学习"听完片段后更新推理"的行为，容易在同个 seg 上死循环（sample 1、2）
- **幻觉时间戳**：模型可能生成超出音频时长的 seg（已有 clamp 保护不再崩溃，但模型不会自纠正）
- **重复 seg**：同一轮里生成多个 `<seg>`，目前只处理第一个

## 下一步

这些是**模型训练问题**，非推理代码 bug：

1. **RL 训练（GRPO）**：给 reward 让模型学会——请求 seg → 听裁剪音频 → 更新推理 → 给出更准确的答案
2. **seg 多样性奖励**：惩罚请求相同 seg 的行为，鼓励探索不同时间段
3. **answer 及时奖励**：鼓励模型在信息足够时尽快给出答案

# Audio-Interleaved Inference Continuation 与连续性奖励说明

日期：2026-05-11

本文档记录对 Echo 论文中 audio-interleaved inference 继续推理机制，以及 `Rconsist` 连续性奖励的理解。该分析用于指导后续 `interleaved_infer.py` 与 RL rollout 的实现。

## 1. 核心理解

Echo 的 audio-interleaved reasoning 不是简单地让模型在文本中输出 `<seg>` 标签，也不是在生成 `<seg>` 后只把裁剪音频单独喂给模型。

更准确地说，每当模型生成一个完整的 segment tag pair：

```text
<seg>s, e</seg>
```

推理系统会暂停当前生成，从原始音频中裁剪 `s-e` 对应片段，并把以下内容重新组成扩展上下文后送回 LALM：

```text
原始音频 A
+ 原始问题 q
+ 当前已经生成的文本 o
+ 裁剪得到的音频片段 A[s:e]
```

也就是论文中的形式：

```text
x'_i = (x_i ⊕ o ⊕ A_{s:e})
```

其中：

```text
x_i = (A, q)
o = 当前已经生成的推理文本，包含刚刚生成的 <seg>s,e</seg>
A_{s:e} = 从原始音频 A 中裁剪出的片段
```

因此，audio-interleaved inference 的关键不是“重新问模型一个新问题”，而是“在原始上下文和已生成推理后追加新的音频证据，让模型继续生成”。

## 2. 为什么会需要连续性奖励

在工程实现上，插入音频片段后，上下文大致变成：

```text
... <seg>3.2, 5.6</seg> [audio segment tokens] ...
```

模型继续生成时，可能会把插入的音频片段理解为一个新的上下文边界或新轮次输入，从而倾向于重新开启一句话、重新开始一段推理，或者立刻生成下一个标签。

常见不理想形式包括：

```text
I will listen to <seg>1.0, 2.0</seg>
<seg>3.0, 4.0</seg>
<answer>A</answer>
```

或：

```text
I will listen to <seg>1.0, 2.0</seg> The audio contains ...
```

这些形式虽然含有 `<seg>`，但并没有把 `<seg>` 自然嵌入当前分析句中，也没有稳定形成“引用片段后立刻分析该片段”的行为。

Echo 希望的形式更接近：

```text
I need to check the early sound. In <seg>1.0, 2.0</seg>, a dog barking is clearly heard, which supports option B.
```

也就是：

```text
引用前：说明为什么要听这一段
<seg>s,e</seg>
引用后：紧接着分析这一段听到了什么、如何支持答案
```

## 3. Rconsist 连续性奖励

为了解决上述断裂问题，Echo 引入了 `Rconsist` 连续性奖励。

该奖励检查每个 `</seg>` 后的下一个文本 token。注意：中间插入的音频 token 属于环境注入内容，不应视为文本 token。实际检查时应忽略插入的音频片段，只看模型后续生成的文本。

如果 `</seg>` 后的下一个文本 token 是：

```text
大写字母
```

通常表示模型开启了新句子，例如：

```text
</seg> The audio contains ...
```

如果 `</seg>` 后的下一个文本 token 是：

```text
<
```

通常表示模型直接开始下一个标签，例如：

```text
</seg><seg>...
```

或推理提前进入结构标签。

这两种情况都会被惩罚：

```text
每次 -0.1
最多累计到 -0.5
```

该奖励的目的不是禁止所有新句子，而是鼓励模型把 `<seg>` 作为当前分析句的一部分，并在 segment 后自然接续分析，而不是让 segment tag 变成孤立标记。

## 4. 对当前项目实现的要求

### 4.1 interleaved inference 的上下文构造

后续实现 `interleaved_infer.py` 或 RL rollout 时，不应使用：

```text
裁剪音频片段 + 新 prompt
```

而应该使用：

```text
原始音频 + 原始 prompt + 已生成文本 + 裁剪音频片段
```

否则模型会更容易把裁剪音频当成新问题输入，导致推理断裂。

推荐伪代码：

```python
context = [full_audio, question]
output = ""

while True:
    text = model.generate_until(context, pattern="<seg>...</seg> or </answer>")
    output += text

    seg = parse_seg(text)
    if not seg:
        break

    audio_clip = clip_audio(full_audio, seg.start, seg.end)
    context = context + [text, audio_clip]
```

### 4.2 reward 实现中的近似检查

如果 RL 阶段的 response 文本中没有显式音频 token，可以先在纯文本 response 上近似实现 `Rconsist`：

```python
def r_consist(response):
    penalty = 0.0
    for each </seg> in response:
        next_char = first_non_space_char_after_seg(response)
        if next_char.isupper() or next_char == "<":
            penalty -= 0.1
    return max(penalty, -0.5)
```

更严格的 interleaved rollout 版本中，应当在逻辑上忽略插入的 audio segment tokens，再检查下一段由模型生成的文本。

## 5. 与 VLM-R3 的对应关系

VLM-R3 的视觉交错推理机制是：

```text
模型生成 bbox
-> 外部系统裁剪图像区域
-> zoom 后重新编码为 visual tokens
-> 插回上下文
-> 模型继续推理
```

Echo 的音频交错推理机制是：

```text
模型生成 <seg>s,e</seg>
-> 外部系统裁剪音频片段
-> 编码为 audio tokens
-> 插回上下文
-> 模型继续推理
```

二者共同点是：

```text
模型负责决定“看/听哪里”
外部系统负责执行“裁剪/插回”
模型基于新证据继续推理
插入的模态 token 是环境状态，不是模型动作
训练 loss / policy gradient 不应作用在插入的模态 token 上
```

## 6. 当前结论

Echo 的 interleaved continuation 应理解为：

```text
原始输入 + 已生成推理 + 新裁剪音频证据
```

被重新送回 LALM 继续生成。

`Rconsist` 连续性奖励的核心作用是防止模型在插入音频后把推理断开，迫使模型在 `<seg>` 后紧接着完成对该片段的分析。后续实现 RL reward 和 interleaved rollout 时，需要把这一点作为核心约束。

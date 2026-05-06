# Echo / Audio-Interleaved Reasoning 实验进度上下文

本文档用于给新的对话或新的助手快速同步当前实验状态。项目目标是参考论文 **Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning**，基于 Qwen2.5-Omni 构建带 `<seg>` 时间片段引用的音频推理能力，并进一步探索 SFT 与 RL。

## 1. 论文流程的核心理解

论文不是简单地“生成 CoT 然后做 SFT”。完整流程大致是：

```text
带时间标注的音频数据
-> Qwen2.5-Omni 提取音频文本信息 A1/A2/A3
-> 结合强标注时间元数据 A4/A5
-> DeepSeek-R1 生成 QA-CoT
-> DeepSeek-R1 重评 QA valid / COT valid
-> 根据重评结果分成 EAQA-SFT / EAQA-RL
-> SFT 得到 cold-start model
-> 推理时启用 audio-interleaved inference
-> 使用 EAQA-RL 做 RL
```

其中：

- `A1`: comprehensive audio description
- `A2`: speech information, if speech exists
- `A3`: music information, if music exists
- `A4`: strong event segments with timestamps
- `A5`: audio duration

论文的数据分流逻辑是：

```text
QA valid = Yes 且 COT valid = Yes
-> 放入 SFT 数据

QA valid = Yes 且 COT valid = No
-> 去掉 CoT，放入 RL 数据

QA valid = No
-> 丢弃
```

论文 SFT 的目标是让模型先学会 audio-grounded reasoning：

```text
<think>... <seg>start, end</seg> ...</think><answer>...</answer>
```

论文 RL 阶段更进一步，不只是让模型输出 `<seg>`，而是在推理时：

1. 模型生成到 `<seg>s, e</seg>`
2. 暂停生成
3. 从原始音频里裁剪 `s-e` 片段
4. 把该音频片段插入上下文
5. 继续生成后续推理

这个机制是论文所谓 **audio-interleaved reasoning** 的关键。

## 2. 当前已经完成的工作

### 2.1 AudioSet-Strong 数据准备

已经把 AudioSet-Strong parquet 转换成了训练需要的结构：

- 音频文件，主要为 `.flac`
- metadata jsonl
- `segment_id`
- `audio_path`
- `duration`
- `labels`
- `events`: 包含 `start`, `end`, `label`

parquet 中的 `audio.bytes` 本身就是 FLAC 字节，所以输出为 `.flac` 是正常的。

强标注 TSV 与 parquet 的匹配结论：

- 不能直接用完整 `segment_id` 匹配
- parquet 中 `video_id` 与 TSV 中 `segment_id` 的 YouTube id 部分可以匹配
- parquet 里已经包含 events 强标注信息，因此后续主要直接使用 parquet 里的 `events`

已完成批量转换和合并，支持断点续写、跳过已经处理过的 parquet。

### 2.2 A1/A2/A3 生成

已经使用本地 Qwen2.5-Omni 对音频生成三类信息：

- `a1_description`
- `a2_speech`
- `a3_music`

对应 prompt 来源于论文作者给出的三个文件：

- `qwen_o_describe.txt`
- `qwen_o_speech.txt`
- `qwen_o_music.txt`

生成脚本经过多次修正，主要处理过这些问题：

- Qwen processor 参数问题
- `StopIteration`
- `audios` 参数无效
- A1 偶发生成异常文本，例如单字符重复、大量数字、突然截断
- A2/A3 与标签冲突时，优先基于强标注做 fallback
- 输入 jsonl 后续会更新，因此脚本支持跳过已生成项

当前 A1/A2/A3 已经可用，但不是人工完美质量。

### 2.3 QA skeleton 构建

为了降低 DeepSeek-R1 成本，我们没有完全按照论文让 DeepSeek 全量自由生成 QA-CoT，而是构建了 QA skeleton。

已生成的 skeleton 类型包括：

- `start_percentage`
- `duration_percentage`
- `gap`
- `overlap`
- `count_before`
- `repeated_event_gap`
- `duration_compare`
- `order`

skeleton 的目标：

- 先用规则从强标注事件中构造可验证的问题骨架
- 控制不同题型比例
- 避免 DeepSeek 生成大量无效题
- 后续 DeepSeek 或模板只负责改写/生成 CoT

后续优化过题型比例，使不同类型尽量均匀出现，而不是先生成一大批 A 类再生成 B 类。

### 2.4 SFT 数据生成

SFT 数据不是完全复现论文的 DeepSeek 全量生成路线，而是混合方案：

1. 一部分由 DeepSeek-R1 生成或润色 CoT
2. 一部分由 skeleton + 本地模板生成 CoT
3. 最后清洗合并为 strict SFT jsonl

清洗逻辑包括：

- 只保留最终训练需要的 `messages` 和 `audios`
- 去除中间字段，避免训练 token 浪费
- 检查 `<think>...</think><answer>...</answer>`
- 检查 `<seg>start, end</seg>`
- 检查 `<answer>` 是否在 choices 中
- 清理异常字符
- 过滤 `/m/`、`/t/` 这类标签污染
- 去重

最终清洗数据曾得到约 12k 条左右样本。

重要文件：

```text
E:\Agent_Project\clean_merge_sft_data.py
E:\Agent_Project\eaqa_sft_train_clean.jsonl
E:\Agent_Project\eaqa_sft_train_clean_report.json
E:\Agent_Project\eaqa_sft_train_clean_strict_report.json
```

远程训练使用的 strict 数据路径类似：

```text
/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/eaqa_sft_train_clean_strict.jsonl
```

### 2.5 SFT 训练

训练基座模型：

```text
/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/
```

训练方式：

- ms-swift
- LoRA
- `torch_dtype float16`
- learning rate `5e-6`
- batch / gradient accumulation 后续做过调整
- `QWEN_OMNI_SKIP_SPK=1`

训练脚本：

```text
C:\Users\13093\AppData\Roaming\MobaXterm\slash\mx86_64b\RemoteFiles\1971914_8_23\test_SFT.sh
```

训练过程中遇到的 checkpoint resume 问题：

- 直接 `--resume_from_checkpoint` 会触发 optimizer/scheduler 加载
- 当前 torch 版本低于 2.6，Transformers 因 CVE-2025-32434 安全限制拒绝 `torch.load`
- 临时方案：复制 checkpoint，删除副本中的 optimizer/scheduler/rng/scaler 文件，再 resume

相关修改：

```bash
SRC_CKPT=".../v7-20260505-145145/checkpoint-749"
DST_CKPT=".../v7-20260505-145145/checkpoint-749-no-optim"

cp -a "$SRC_CKPT" "$DST_CKPT"
rm -f "$DST_CKPT"/optimizer.pt
rm -f "$DST_CKPT"/scheduler.pt
rm -f "$DST_CKPT"/scaler.pt
rm -f "$DST_CKPT"/rng_state*.pth

--resume_from_checkpoint "$DST_CKPT"
```

## 3. 当前训练/评估结果

### 3.1 已经观察到的正向效果

SFT 后模型在格式方面明显提升：

- `<think>...</think><answer>...</answer>` 基本稳定
- `<answer>` 在 choices 中的比例大幅提升
- 能输出 `<seg>`，但不够稳定

### 3.2 v7 评估结果

评估 report：

```text
C:\Users\13093\AppData\Roaming\MobaXterm\slash\mx86_64b\RemoteFiles\1971914_10_114\eval_report_20260505-172853.json
```

关键结果：

```json
{
  "processed": 100,
  "has_think_answer": 99,
  "has_seg": 60,
  "fully_structured": 60,
  "answer_in_choices": 96,
  "answer_correct": 30
}
```

结论：

- 格式有明显提升
- answer in choices 很好
- answer accuracy 仍然偏低
- `<seg>` 对百分比题型尤其不稳定

分题型看，结构最拖后腿的是：

```text
start_percentage
duration_percentage
```

它们的 `<seg>` / fully structured 比例很低。

### 3.3 v8 另一批 100 条评估

评估 report：

```text
C:\Users\13093\AppData\Roaming\MobaXterm\slash\mx86_64b\RemoteFiles\1971914_10_119\eval_report_20260505-192207.json
```

关键结果：

```json
{
  "processed": 100,
  "has_think_answer": 100,
  "has_seg": 27,
  "fully_structured": 27,
  "answer_in_choices": 99,
  "answer_correct": 47
}
```

注意：这次评估的 `start_index` 与 v7 不同，因此不能直接比较模型好坏。它说明：

- answer accuracy 在这批数据上更高
- 但 `<seg>` 格式反而不稳定
- `duration_compare` 和 `order` 结构最好
- `duration_percentage`、`overlap`、`gap` 等结构较弱

### 3.4 当前判断

继续堆 SFT epoch 的收益有限。loss 已经进入平台期。当前更大的瓶颈不是模型没训练够，而是：

- 数据质量与题型设计
- percentage/gap/overlap 类问题对 `<seg>` 的诱导不够强
- 缺少论文式 DeepSeek-R1 语义重评
- 还没有真正实现 audio-interleaved inference 和 RL

## 4. 已完成的评估脚本

已有批量评估脚本，能统计：

- `has_think_answer`
- `has_seg`
- `fully_structured`
- `answer_in_choices`
- `answer_correct`
- `type_stats`

评估脚本路径：

```text
C:\Users\13093\AppData\Roaming\MobaXterm\slash\mx86_64b\RemoteFiles\1971914_8_42\eval_afterSFT.py
```

后续曾调大 batch：

```python
EVAL_BATCH_SIZE = 8
```

建议进行更稳定的模型对比时，固定同一批 eval：

```python
START_INDEX = 13000
MAX_SAMPLES = 500
```

分别评估不同 checkpoint，避免样本变化导致结论不稳。

## 5. RL 数据状态

已确认论文式 RL 数据格式：

```text
C:\Users\13093\AppData\Roaming\MobaXterm\slash\mx86_64b\RemoteFiles\1971914_14_116\EAQA_RL.jsonl
```

样例结构：

```json
{
  "id": "AudioSet_0",
  "audio_path": "audios/AudioSet/audio_21.wav",
  "question": "...",
  "multi_choice": ["...", "...", "...", "..."],
  "answer": "..."
}
```

特点：

- 没有 CoT
- 没有 `messages`
- 用于 RL，而不是普通 SFT

## 6. 还没有完成的关键工作

### 6.1 DeepSeek-R1 语义重评

论文中会用 judge prompt 检查：

```text
[QA valid]: Yes/No
[COT valid]: Yes/No
```

它判断：

- 问题是否必须依赖音频信息
- answer 是否唯一
- choices 是否有歧义
- CoT 是否编造 A1-A5 里没有的信息
- CoT 是否逻辑连贯

我们目前做过格式/规则清洗，但没有完整做这个语义重评。

### 6.2 严格 SFT/RL 数据分流

论文分流：

```text
QA valid Yes + COT valid Yes -> SFT
QA valid Yes + COT valid No -> RL
QA invalid -> 丢弃
```

我们现在的 SFT 数据主要靠规则清洗和混合生成，没有完整复现这一分流。

### 6.3 Audio-interleaved inference

还没有实现：

```text
生成 <seg>s, e</seg>
-> 暂停
-> 裁剪音频 s-e
-> 插回上下文
-> 继续生成
```

当前评估主要还是：

```text
完整音频 + question -> 一次性生成 response
```

这与论文真正的 interleaved 推理机制不同。

### 6.4 RL 训练

还没有搭建正式 RL 流程。

论文 RL 配置包括：

- GRPO/PPO-style
- EAQA-RL
- 8 rollouts per query
- batch size 64
- mini-batch size 32
- learning rate 1e-6
- KL coefficient 0.04

奖励函数：

```text
R = Rformat + Rconsist + Racc + Rseg
```

其中：

- `Rformat`: `<think>/<answer>/<seg>` 格式
- `Rconsist`: `</seg>` 后是否自然接续分析
- `Racc`: answer 是否匹配 ground truth
- `Rseg`: answer 正确且至少引用一个 segment

## 7. 当前最合理的下一步

不建议继续盲目增加 SFT epoch。

更合理的路线：

1. 固定同一批 eval，至少 500 条，对当前最新 checkpoint 做稳定评估。
2. 对 500-1000 条 SFT/RL 候选数据跑 DeepSeek-R1 QA/COT judge，估计数据噪声。
3. 根据 judge 结果清理 SFT 数据，必要时重新训练一个更干净的 cold-start model。
4. 实现最小版本 audio-interleaved inference。
5. 用 EAQA_RL.jsonl 做小规模 RL 流程验证。
6. RL 初期可以先过滤 `start_percentage` 和 `duration_percentage`，因为它们当前 `<seg>` 格式最差。
7. 先验证格式 reward 是否能提升 `<seg>` 引用，再逐步加入答案正确 reward。

## 8. 新集群环境迁移注意事项

用户正在把环境迁移到另一个集群。新集群路径类似：

```text
/Work21/2025/jiangpeiyuan/anaconda3/envs/qwen_echo
```

遇到过的问题：

1. conda strict priority 导致 Python 3.10 创建失败
2. conda-forge Python 需要更高 glibc，旧集群报 `GLIBC_2.28 not found`
3. `pip freeze` 中有大量 `@ file:///home/task...` 本地构建路径，不能直接迁移
4. `av==17.0.1` 会尝试编译 FFmpeg，因缺少 libavformat/libavcodec dev headers 失败
5. `/tmp` 所在根分区只有约 655MB，安装 PyTorch 时空间不足

建议环境安装不要全量使用 `qwen_echo_pip.txt`，而是安装核心依赖：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.52.4 tokenizers==0.21.4 accelerate==1.13.0 peft==0.15.2 safetensors sentencepiece==0.2.1 protobuf qwen-omni-utils==0.0.9 ms-swift==3.5.2

pip install librosa==0.11.0 soundfile==0.13.1 soxr==1.0.0 scipy==1.15.3 numpy==1.26.4 pandas==2.3.3 pyarrow==24.0.0 tqdm datasets==3.3.2

pip install tensorboard==2.20.0 openai==2.32.0 requests pydantic==2.11.10 pyyaml rich tiktoken==0.12.0 modelscope==1.36.1
```

由于 `/tmp` 太小，需要设置：

```bash
mkdir -p /Work21/2025/jiangpeiyuan/tmp
mkdir -p /Work21/2025/jiangpeiyuan/pip_cache

export TMPDIR=/Work21/2025/jiangpeiyuan/tmp
export TEMP=/Work21/2025/jiangpeiyuan/tmp
export TMP=/Work21/2025/jiangpeiyuan/tmp
export PIP_CACHE_DIR=/Work21/2025/jiangpeiyuan/pip_cache
```

## 9. 关键结论

当前项目不是从零开始。已经完成了：

- AudioSet-Strong 数据准备
- A1/A2/A3 生成
- SFT 数据构建与清洗
- Qwen2.5-Omni LoRA SFT
- 批量评估脚本
- RL 数据格式确认

但若严格复现论文，还缺：

- DeepSeek-R1 语义重评
- 按 QA/COT valid 分流 SFT/RL
- audio-interleaved inference
- RL reward function
- RL 训练流程

当前最准确的状态是：

```text
SFT cold-start model 已经初步完成，但还没有真正进入论文的 audio-interleaved RL 阶段。
```


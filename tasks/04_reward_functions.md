# RL Reward Functions

日期: 2026-05-08（实现完成，63 单元测试通过）

## 背景

Echo 论文中的 GRPO 训练需要 reward signal。当前需要纯 Python 实现（不依赖模型推理、不接入 verl/TRL），方便后续集成到训练循环中。

核心思路：所有 reward function 操作模型输出文本，提取 tag 内容做判断，不需要加载模型或 GPU。

## 模块结构

```
echo_rl/
├── __init__.py
└── rewards.py          # 全部 reward function
tests/
└── test_rewards.py     # 63 个 pytest 单元测试
```

## Reward 组件

### 提取函数（非 reward，供内部调用）

| 函数 | 作用 |
|------|------|
| `extract_answer(response)` | 提取 `<answer>...</answer>` 内容 |
| `extract_segments(response)` | 提取所有 `<seg>start,end</seg>`，返回 `[(float, float)]` |
| `has_think(response)` | 检查是否有闭合的 `<think>...</think>` |
| `has_answer_tag(response)` | 检查是否有闭合的 `<answer>...</answer>` |
| `normalize_answer(text)` | 标准化答案：strip/lowercase → 合并空格 → 去尾句点 → 去 "second(s)" |

### Reward 函数

| 函数 | 范围 | 说明 |
|------|------|------|
| `r_format(response)` | [0, 0.5] | 格式 reward。有 `<think>...</think>` 得 0.25，有 `<answer>...</answer>` 得 0.25 |
| `r_consist(response, mode="paper")` | [-0.5, 0] (paper) / [0, 0.5] (positive) | 一致性 reward。每个 `</seg>` 后首字符大写或 `<` 视为不连贯，每次 -0.1，上限 -0.5 |
| `r_acc(response, gt_answer)` | {0, 0.5} | 准确率 reward。`normalize_answer(pred) == normalize_answer(gt)` 得 0.5 |
| `r_seg(response, gt_answer)` | {0, 0.5} | Segment 使用 reward。答案正确且包含 `<seg>` 得 0.5 |
| `total_reward(response, gt_answer, consist_mode="paper")` | dict | 组合所有 reward，返回 `{format, consistency, accuracy, segment, total}` |

### r_consist 两种模式

```
paper (默认):  -penalty ∈ [-0.5, 0]    0 = 无不连贯
positive:      max(0, 0.5 - penalty)   [0, 0.5] 兼容旧行为
```

## 测试数据

测试数据使用 Job 41640 的 5 条真实模型输出：

| Sample | 类型 | 预测 | 答案 | 正确 |
|--------|------|------|------|:----:|
| S1 | gap | 0.7s | 0.1s | ✗ |
| S2 | count_before | 2 | 2 | ✓ |
| S3 | repeated_event_gap | 0.4s | 0.1s | ✗ |
| S4 | duration_compare | the whispering | the whispering | ✓ |
| S5 | gap | 0.7s | - | ✗ |

## 测试覆盖

63 个测试，覆盖：

- `extract_answer`: 6 条 — basic, trailing, nested, empty, no tag, malformed
- `extract_segments`: 6 条 — basic, none, multiple, whitespace, empty, malformed
- `has_think` / `has_answer_tag`: 各 3 条
- `normalize_answer`: 7 条 — strip/lower, trailing period, collapse ws, text, numeric, singular "second"
- `r_format`: 6 条 — both, none, only think, only answer, malformed, empty
- `r_consist`: 16 条 — paper mode (9) + positive mode (6) + invalid mode raises
- `r_acc`: 6 条 — correct/wrong numeric, correct text, no answer, empty gt, unit normalization
- `r_seg`: 5 条 — correct with/without seg, wrong with/without seg, empty
- `total_reward`: 6 条 — direct correct, interleaved correct/wrong, positive compat, empty, no tags

## Reward 分布示例

Job 41640 样本的 reward 分布（paper mode）：

```
S2 (correct, no segs):
  format=0.5, consistency=0.0, accuracy=0.5, segment=0.0, total=1.0

S4 (correct, interleaved, finalize):
  format=0.25, consistency=-0.2, accuracy=0.5, segment=0.5, total=1.05

S1 (wrong, interleaved, finalize):
  format=0.25, consistency=-0.4, accuracy=0.0, segment=0.0, total=-0.15
```

## 运行测试

```bash
cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project
python -m pytest tests/test_rewards.py -v
```

## 依赖

- 无外部依赖（仅 `re`, `typing`, `pytest`）
- 不加载模型，不需要 GPU
- 不接入 verl/TRL

## 下一步

接入 GRPO 训练循环时，将 `total_reward()` 作为 reward function 传入训练框架。

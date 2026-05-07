# 固定评估集 + 统一评估报告

日期: 2026-05-07

## 背景

当前项目已有 `output/testCode/eval_afterSFT.py` 能做评估，但 v7/v8 评估时使用的样本不同（硬编码 `START_INDEX` 和 `MAX_SAMPLES`），导致不同 checkpoint 的评估结果不可直接比较。

## 任务要求

1. **build_eval_manifest.py**: 从输入 JSONL 中固定抽取或切片样本，输出 `eval_manifest.jsonl`
   - 支持 `--input_jsonl`, `--output_manifest`, `--start_index`, `--max_samples`, `--seed`
   - 每条至少保留: `id`, `audio_path`/`audios`, `question`, `choices`/`multi_choice`, `answer`, `type`, 原始 index
   - 输出顺序固定，同参数重复运行结果一致
   - 字段名兼容：`question`/`choices`/`multi_choice`/`answer`/`audio_path`/`audios`/`type`/`qa_type`

2. **支持 `--eval_manifest` 的评估脚本**: 读取 manifest 推理，不硬编码 START_INDEX
   - 支持 `--model_path`, `--adapter_path`(可选), `--eval_manifest`, `--output_report`, `--max_new_tokens`, `--batch_size`
   - 统计: `processed`, `has_think_answer`, `has_seg`, `fully_structured`, `answer_in_choices`, `answer_correct`, `type_stats`

3. **report 额外保存**: `checkpoint_path`, `adapter_path`, `eval_manifest_path`, `generation_config`, `timestamp`
   - 每条样本结果保存到 `predictions.jsonl`: `id`, `type`, `question`, `choices`, `gold_answer`, `pred_answer`, `has_think_answer`, `has_seg`, `fully_structured`, `answer_in_choices`, `answer_correct`, `response`

4. **不删除旧脚本，不大规模重构目录**

## 交付物

### 新增文件

| 文件 | 路径 | 说明 |
|------|------|------|
| build_eval_manifest.py | `scripts/build_eval_manifest.py` | 构建固定评估集清单 |
| eval_from_manifest.py | `scripts/eval_from_manifest.py` | 基于 manifest 的评估脚本 |

### scripts/build_eval_manifest.py

**功能**: 从输入 JSONL 按 start_index + max_samples 抽取，以固定 seed 打乱输出。

```bash
python3 scripts/build_eval_manifest.py \
  --input_jsonl output/GeneratedData/qa_skeleton.jsonl \
  --output_manifest output/GeneratedData/eval_manifest_500.jsonl \
  --start_index 13000 \
  --max_samples 500 \
  --seed 42
```

**manifest 每行字段**: `id`, `audio_path`, `question`, `choices`(list), `answer`, `type`, `original_index`

### scripts/eval_from_manifest.py

**功能**: 读取 manifest，逐批推理，输出统一报告 + 逐条预测结果。

```bash
# 带 LoRA adapter
python scripts/eval_from_manifest.py \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --adapter_path /path/to/lora/checkpoint \
  --eval_manifest output/GeneratedData/eval_manifest_500.jsonl \
  --output_dir output/eval_results/my_eval \
  --batch_size 16 \
  --max_new_tokens 256

# 纯 base model
python scripts/eval_from_manifest.py \
  --model_path /path/to/Qwen2.5-Omni-7B \
  --eval_manifest output/GeneratedData/eval_manifest_500.jsonl \
  --output_dir output/eval_results/base_eval
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `eval_report.json` | 总体统计 + 按类型统计 + 配置信息 |
| `predictions.jsonl` | 每条样本的详细结果 |

**eval_report.json 结构**:

```json
{
  "checkpoint_path":           // 模型路径
  "adapter_path":              // adapter 路径（或 null）
  "eval_manifest_path":        // manifest 路径
  "generation_config": {       // 推理参数
    "max_new_tokens": 256,
    "batch_size": 16,
    "sample_rate": 16000
  },
  "timestamp":                 // 运行时间戳
  "num_samples_in_manifest":   // manifest 总条数
  "stats": {                   // 绝对计数
    "processed", "has_think_answer", "has_seg",
    "fully_structured", "answer_in_choices",
    "answer_correct", "failed"
  },
  "rates": {                   // 比例
    "has_think_answer", "has_seg", "fully_structured",
    "answer_in_choices", "answer_acc"
  },
  "type_stats": {              // 按题型细分
    "start_percentage": { "processed", "fully_structured", "has_seg", "answer_correct" },
    "duration_percentage": { ... },
    ...
  }
}
```

### 遗留文件（未删除）

- `output/testCode/eval_afterSFT.py` — 旧评估脚本，原封保留

## 验收标准

1. 同一 checkpoint + 同一 eval_manifest 连续跑两次，`processed` 和 `type_stats` 一致 ✅（已通过 diff 验证）
2. 不同 checkpoint 可复用同一个 eval_manifest 直接比较
3. manifest 已预生成: `output/GeneratedData/eval_manifest_500.jsonl`（13000行起，500条，seed=42）

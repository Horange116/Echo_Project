# DeepSeek-R1 / OpenAI-compatible 语义重评

日期: 2026-05-07

## 背景

论文中对候选 QA-CoT 数据做了 judge 语义重评，按 [QA valid] / [COT valid] 分流：
- QA valid Yes + COT valid Yes → SFT
- QA valid Yes + COT valid No → RL（去掉 CoT）
- QA valid No → 丢弃

之前的清洗只做了规则/格式清洗，没有做语义重评。

## 任务要求

1. 新增 `scripts/judge_eaqa_candidates.py`，对候选 QA-CoT 数据做论文式 judge
2. 支持自定义 `base_url`、`model`、`api_key_env`，不写死官方 DeepSeek 地址
3. 每条输出 `qa_valid` / `cot_valid` / `judge_raw` / `judge_error`
4. 支持断点续跑 (`--resume`) 和失败重试 (`--retry_failed`)
5. API key 从环境变量读取，不写入日志和文件

## 交付物

### 新增文件

| 文件 | 路径 | 说明 |
|------|------|------|
| judge_eaqa_candidates.py | `scripts/judge_eaqa_candidates.py` | 语义重评 Judge |
| build_judge_subset.py | `scripts/build_judge_subset.py` | 分层抽样（来源段 + 题型） |

### scripts/judge_eaqa_candidates.py 参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_jsonl` | 必填 | 输入候选数据 JSONL |
| `--output_jsonl` | 必填 | 输出判定结果 JSONL |
| `--report_json` | 必填 | 统计报告 JSON |
| `--base_url` | 必填 | OpenAI-compatible API base URL |
| `--api_key_env` | `ECHO_JUDGE_API_KEY` | API key 环境变量名 |
| `--model` | `deepseek-reasoner` | 模型名 |
| `--start_index` | 0 | 从第几行开始 |
| `--max_samples` | 0(全部) | 最多处理多少条 |
| `--sleep_seconds` | 0.2 | API 调用间隔 |
| `--timeout` | 120 | API 超时 |
| `--max_retries` | 3 | 失败最大重试次数 |
| `--temperature` | 0 | 生成温度 |
| `--max_tokens` | 128 | 最大生成 token |
| `--resume` | true | 断点续跑 |
| `--retry_failed` | false | 重试失败样本 |
| `--concurrency` | 1 | 并发线程数（>1 使用 ThreadPoolExecutor） |
| `--qps_limit` | 0 | 每秒请求数上限，0=不限 |
| `--progress_every` | 20 | 每 N 条打印一次进度 |

### scripts/build_judge_subset.py 参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_jsonl` | 必填 | 输入候选数据 JSONL |
| `--output_jsonl` | 必填 | 输出 subset JSONL |
| `--report_json` | 必填 | 统计报告 JSON |
| `--seed` | 42 | 随机种子 |
| `--polished_range` | `0:3000` | DeepSeek 润色段的 index 范围 |
| `--polished_per_type` | 38 | 润色段每个 type 抽多少条 |
| `--template_per_type` | 38 | 模板段每个 type 抽多少条 |
| `--max_total` | 600 | 最多输出多少条 |
| `--type_field_candidates` | `type,qa_type,skeleton_type` | type 字段优先级 |

subset 按 "source_group + type" 分层抽样：
- index in polished_range → `deepseek_polished`
- index >= polished_range → `template_or_unpolished`
- 每个 group 内再按 type 分组分别抽样
- 超过 max_total 则用固定 seed 随机截断

### 字段兼容

| 目标 | 优先级 |
|------|--------|
| choices | choices > multi_choice > options |
| CoT | cot > response > assistant_response > output > messages[-1].content > raw_response |
| type | type > qa_type > skeleton_type |
| audio_path | audio_path > audios[0] |

### 命令行示例

**smoke test（20条，快速验证链路）:**
```bash
export ECHO_JUDGE_API_KEY=sk-xxx

python scripts/judge_eaqa_candidates.py \
  --input_jsonl output/GeneratedData/qa_skeleton.jsonl \
  --output_jsonl output/judge/smoke_judged.jsonl \
  --report_json output/judge/smoke_report.json \
  --base_url https://api.xxx.com/v1 \
  --model deepseek-reasoner \
  --start_index 0 \
  --max_samples 20
```

**第三方中转 API（如 DeepSeek 兼容中转）:**
```bash
export ECHO_JUDGE_API_KEY=sk-your-third-party-key

python scripts/judge_eaqa_candidates.py \
  --input_jsonl output/GeneratedData/eaqa_sft_generated.jsonl \
  --output_jsonl output/judge/full_judged.jsonl \
  --report_json output/judge/full_report.json \
  --base_url https://third-party-api.com/v1 \
  --model deepseek-reasoner \
  --start_index 0 \
  --max_samples 500 \
  --sleep_seconds 0.5
```

**分层抽样 + 并发 judge（600 条诊断）:**
```bash
# 1. 构建分层抽样 subset
python scripts/build_judge_subset.py \
  --input_jsonl /path/to/candidates_with_a1_a3.jsonl \
  --output_jsonl output/judge/judge_subset_600.jsonl \
  --report_json output/judge/judge_subset_report.json \
  --polished_range 0:3000 \
  --polished_per_type 38 \
  --template_per_type 38 \
  --max_total 600

# 2. 并发 12 跑 judge
python scripts/judge_eaqa_candidates.py \
  --input_jsonl output/judge/judge_subset_600.jsonl \
  --output_jsonl output/judge/judged_600.jsonl \
  --report_json output/judge/judged_600_report.json \
  --base_url https://your-api/v1 \
  --model deepseek-reasoner \
  --concurrency 12 \
  --sleep_seconds 0 \
  --resume
```

**断点续跑:**
```bash
python scripts/judge_eaqa_candidates.py \
  --input_jsonl output/judge/judge_subset_600.jsonl \
  --output_jsonl output/judge/judged_600.jsonl \
  --base_url https://your-api/v1 \
  --concurrency 12 \
  --resume
```

**重试失败样本:**
```bash
python scripts/judge_eaqa_candidates.py \
  --input_jsonl output/GeneratedData/eaqa_sft_generated.jsonl \
  --output_jsonl output/judge/full_judged.jsonl \
  --report_json output/judge/full_report.json \
  --base_url https://third-party-api.com/v1 \
  --model deepseek-reasoner \
  --start_index 0 \
  --max_samples 500 \
  --resume \
  --retry_failed
```

### report_json 示例

```json
{
  "input_jsonl": "data.jsonl",
  "output_jsonl": "judged.jsonl",
  "judge_config": {
    "base_url_host": "api.xxx.com",
    "model": "deepseek-reasoner",
    "temperature": 0,
    "max_tokens": 128,
    "max_retries": 3,
    "sleep_seconds": 0.2,
    "concurrency": 12,
    "qps_limit": 0,
    "resume": true,
    "retry_failed": false
  },
  "timestamp": "2026-05-07T12:00:00Z",
  "total_input": 500,
  "processed": 498,
  "skipped_existing": 2,
  "api_errors": 1,
  "parse_errors": 3,
  "qa_valid_yes": 350,
  "qa_valid_no": 148,
  "cot_valid_yes": 280,
  "cot_valid_no": 218,
  "both_valid": 250,
  "qa_valid_yes_cot_no": 100,
  "qa_invalid": 148,
  "by_type_stats": {
    "overlap": {
      "processed": 85,
      "qa_valid_yes": 60,
      "qa_valid_no": 25,
      "cot_valid_yes": 50,
      "cot_valid_no": 35,
      "both_valid": 45,
      "qa_valid_yes_cot_no": 15,
      "parse_errors": 0,
      "api_errors": 0
    }
  }
}
```

### 断点续跑流程

1. 第一次运行正常执行，写入 `output_jsonl`
2. 第二次加 `--resume`，脚本读取已存在的 output_jsonl，收集已有 `id`
3. 输入数据中 `id` 已存在的跳过
4. `--retry_failed` 时，仅跳过 `qa_valid` 或 `cot_valid` 不为 null 的样本
5. 每处理一条就 append，中途中断不会丢失
6. 恢复后直接继续处理未完成的样本

### 安全措施

- API key 仅从环境变量读取，不打印
- `judge_base_url_host` 只记录 hostname
- 错误日志不包含 Authorization header
- output_jsonl / report_json 不保存 key

## 执行结果

日期: 2026-05-08

对 600 条分层抽样数据（deepseek_polished 301 条 + template_or_unpolished 299 条）进行 DeepSeek-R1 judge，结果如下。

### 整体分布

| 分流 | 数量 | 占比 |
|------|------|------|
| SFT (both valid) | 498 | 83.0% |
| RL (QA=Y, COT=N) | 44 | 7.3% |
| 丢弃 (QA=N) | 58 | 9.7% |

### 按来源段

| source_group | total | QA=Y | QA=N | QA pass | COT=Y | COT=N |
|-------------|------|------|------|---------|------|------|
| deepseek_polished | 301 | 264 | 37 | 87.7% | 235 | 66 |
| template_or_unpolished | 299 | 278 | 21 | 93.0% | 266 | 33 |

### 按题型

| qa_type | total | QA=Y | QA=N | QA pass | COT=Y | COT=N |
|---------|------|------|------|---------|------|------|
| gap | 76 | 76 | 0 | 100.0% | 76 | 0 |
| duration_compare | 74 | 74 | 0 | 100.0% | 73 | 1 |
| repeated_event_gap | 76 | 76 | 0 | 100.0% | 75 | 1 |
| count_before | 75 | 74 | 1 | 98.7% | 65 | 10 |
| duration_percentage | 75 | 73 | 2 | 97.3% | 61 | 14 |
| order | 74 | 65 | 9 | 87.8% | 62 | 12 |
| overlap | 75 | 56 | 19 | 74.7% | 52 | 23 |
| start_percentage | 75 | 48 | 27 | 64.0% | 37 | 38 |

### 按来源段 × 题型

| source / qa_type | total | QA=Y | QA=N | QA pass |
|-----------------|------|------|------|---------|
| deepseek_polished/count_before | 37 | 37 | 0 | 100.0% |
| deepseek_polished/duration_compare | 37 | 37 | 0 | 100.0% |
| deepseek_polished/duration_percentage | 38 | 36 | 2 | 94.7% |
| deepseek_polished/gap | 38 | 38 | 0 | 100.0% |
| deepseek_polished/order | 37 | 33 | 4 | 89.2% |
| deepseek_polished/overlap | 38 | 24 | 14 | 63.2% |
| deepseek_polished/repeated_event_gap | 38 | 38 | 0 | 100.0% |
| deepseek_polished/start_percentage | 38 | 21 | 17 | 55.3% |
| template_or_unpolished/count_before | 38 | 37 | 1 | 97.4% |
| template_or_unpolished/duration_compare | 37 | 37 | 0 | 100.0% |
| template_or_unpolished/duration_percentage | 37 | 37 | 0 | 100.0% |
| template_or_unpolished/gap | 38 | 38 | 0 | 100.0% |
| template_or_unpolished/order | 37 | 32 | 5 | 86.5% |
| template_or_unpolished/overlap | 37 | 32 | 5 | 86.5% |
| template_or_unpolished/repeated_event_gap | 38 | 38 | 0 | 100.0% |
| template_or_unpolished/start_percentage | 37 | 27 | 10 | 73.0% |

### 洞察

- gap / duration_compare / repeated_event_gap 三类两类数据均接近 100%，质量稳定
- start_percentage 整体最差（64%），DeepSeek 润色段仅 55.3%，模板段 73.0%
- overlap 模板段 86.5% 尚可，但 DeepSeek 润色段仅 63.2%
- 模板段整体优于 DeepSeek 润色段（93.0% vs 87.7%）
- 零 API 错误 / 解析错误

---

## CoT 生成进度

日期: 2026-05-08

全部 30,904 条骨架（qa_skeleton.jsonl）已完成 CoT 生成。SFT 候选 24,861 条覆盖率达到 100%。

### 生成批次

| 来源 | 索引范围 | 数量 | 质量 |
|------|----------|------|------|
| DeepSeek-R1 API | 0-2999 | 3,210 | 100% |
| 模板生成 | 3000-12999 | 7,904 | 100% |
| 模板生成 | 13000-30903 | 17,904 | 100% |
| 模板补缺 | 0-2999 中遗漏 | 39 | 100% |
| **总计** | **0-30903** | **29,057** | |

### 输出文件

```
output/GeneratedData/
├── eaqa_sft_generated.jsonl                       # DeepSeek: 3,210
├── eaqa_sft_local_generated_3000_12999.jsonl      # 模板: 7,904
├── eaqa_sft_local_generated_13000_30903.jsonl     # 模板: 17,904
├── eaqa_sft_local_generated_remaining_39.jsonl    # 模板补缺: 39
└── qa_skeleton.jsonl                              # 原始骨架: 30,904
```

### SFT 候选质量统计（最终）

| 指标 | 值 |
|------|-----|
| SFT 候选数 | 24,861 |
| CoT 覆盖率 | **100%** |
| has_think_answer | 100% |
| has_seg | 100% |
| fully_structured | 100% |
| answer_in_choices | 100% |
| 平均 CoT 长度 | 296 chars |
| 平均 seg/样本 | 1.9 |
| 无效时间戳 | 0 |

### 相关脚本

- `output/GenerateDataToSFT/generate_sft_local_from_skeleton.py` — 模板 CoT 生成（原型）
- `scripts/02_judge_filter/stats_sft_candidates.py` — 质量统计
- `output/judge/sft_candidate_stats_final.json` — 最终报告

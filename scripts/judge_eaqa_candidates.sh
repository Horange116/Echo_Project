#!/bin/bash
# judge_eaqa_candidates.sh — 语义重评快捷启动脚本
# 用法: bash scripts/judge_eaqa_candidates.sh [--max_samples 20]

set -e

# ─── 配置区 ────────────────────────────────────
# 第三方中转 API 地址（按需修改）
BASE_URL="https://api.xxx.com/v1"
MODEL="deepseek-reasoner"
# API Key 环境变量名（默认 ECHO_JUDGE_API_KEY）
API_KEY_ENV="ECHO_JUDGE_API_KEY"

# 输入输出路径
INPUT_JSONL="output/GeneratedData/qa_skeleton.jsonl"
OUTPUT_DIR="output/judge"
# ──────────────────────────────────────────────

# 检查 API Key
if [ -z "${!API_KEY_ENV}" ]; then
    echo "错误: 环境变量 $API_KEY_ENV 未设置"
    echo "请先执行: export $API_KEY_ENV=sk-your-key"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 解析额外参数（如 --max_samples 20）
EXTRA_ARGS=("$@")

python scripts/judge_eaqa_candidates.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "${OUTPUT_DIR}/judged_${TIMESTAMP}.jsonl" \
    --report_json "${OUTPUT_DIR}/report_${TIMESTAMP}.json" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --api_key_env "$API_KEY_ENV" \
    --sleep_seconds 0.5 \
    "${EXTRA_ARGS[@]}"

echo "完成！"
echo "  输出: ${OUTPUT_DIR}/judged_${TIMESTAMP}.jsonl"
echo "  报告: ${OUTPUT_DIR}/report_${TIMESTAMP}.json"

#!/usr/bin/env python3
"""
论文式 Audio-QA CoT Filtering (Judge) 脚本。

对候选 QA-CoT 数据调用 OpenAI-compatible API 做语义重评，
输出每条样本的 [QA valid] / [COT valid] 判定，用于后续 SFT/RL 数据分流。

支持自定义 base_url / model / api_key_env，兼容第三方中转 API。
支持断点续跑、失败重试。

用法：
  python scripts/judge_eaqa_candidates.py \
      --input_jsonl data.jsonl \
      --output_jsonl judged.jsonl \
      --report_json report.json \
      --base_url https://api.xxx.com/v1 \
      --api_key_env ECHO_JUDGE_API_KEY \
      --model deepseek-reasoner \
      --start_index 0 \
      --max_samples 500
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

from openai import OpenAI


# ──────────────────────────────────────────────
# 字段兼容提取
# ──────────────────────────────────────────────

def _safe_json_loads(val):
    """尝试解析 JSON 字符串，包含 Python repr 风格的单引号兼容。"""
    if isinstance(val, (dict, list)):
        return val
    if not isinstance(val, str):
        return val
    # 标准 JSON
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    # 单引号兼容
    try:
        return json.loads(val.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        pass
    return val


def get_field(item, *keys, default=None):
    """按优先级取第一个存在的字段。"""
    for k in keys:
        if k in item and item[k] is not None:
            v = item[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            elif not isinstance(v, str):
                return v
    return default


def get_choices(item):
    """choices > multi_choice > options，返回 list。"""
    raw = get_field(item, "choices", "multi_choice", "options")
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, list):
        return parsed
    # 字符串靠正则提取
    m = re.findall(r"'([^']*)'", str(raw))
    if m:
        return m
    return [str(raw)]


def get_cot(item):
    """cot > response > assistant_response > output > messages[-1].content > raw_response。"""
    for k in ("cot", "response", "assistant_response", "output"):
        v = item.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()

    # messages 字段: 可能是字符串或 list
    msgs = item.get("messages")
    if msgs:
        if isinstance(msgs, str):
            try:
                msgs = _safe_json_loads(msgs)
            except Exception:
                pass
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

    # raw_response
    r = item.get("raw_response", "")
    if isinstance(r, str) and r.strip():
        return r.strip()

    return ""


def get_type(item):
    """type > qa_type > skeleton_type。"""
    return get_field(item, "type", "qa_type", "skeleton_type", default="unknown")


def get_audio_path(item):
    """audio_path > audios[0] > audios (string)。"""
    ap = item.get("audio_path")
    if ap and isinstance(ap, str) and ap.strip():
        return ap.strip()
    audios = item.get("audios")
    if audios:
        if isinstance(audios, list) and len(audios) > 0:
            return str(audios[0])
        if isinstance(audios, str):
            return audios.strip()
    return ""


def get_id(item):
    """id > skeleton_id > segment_id > hash。"""
    return get_field(item, "id", "skeleton_id", "segment_id", default="")


def _format_events(events_raw):
    """把 events 转成可读文本。"""
    ev = _safe_json_loads(events_raw)
    if not isinstance(ev, list):
        return str(events_raw) if events_raw else "N/A"
    lines = []
    for i, e in enumerate(ev, 1):
        s = e.get("start", "?")
        en = e.get("end", "?")
        lbl = e.get("label", "?")
        lines.append(f"  {i}. [{s}-{en}] {lbl}")
    return "\n" + "\n".join(lines)


# ──────────────────────────────────────────────
# Judge Prompt
# ──────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are an expert judge for evaluating audio question-answering data. "
    "Your task is to assess the quality of questions, answers, and reasoning chains. "
    "Respond only with the two lines:\n"
    "[QA valid]: Yes/No\n"
    "[COT valid]: Yes/No"
)

JUDGE_USER_PROMPT_TEMPLATE = """You are given the following information extracted from an audio clip:

**Audio Information**
A1 (Comprehensive Description): {a1}
A2 (Speech Information): {a2}
A3 (Music Information): {a3}
A4 (Strong Event Segments with Timestamps): {a4}
A5 (Audio Duration): {a5}

**Question**: {question}

**Choices**: {choices}

**Answer**: {answer}

**Candidate CoT (Chain of Thought)**:
{cot}

---
Evaluate the above according to the following criteria:

### 1. Question and Answer Quality
- Is the question necessarily dependent on audio information?
- Is the answer uniquely reasonable (not ambiguous)?
- Are the choices clear and unambiguous?

Give [QA valid]: Yes if the Q&A pair meets ALL criteria, otherwise No.

### 2. CoT Quality
- Does the CoT contain information NOT present in A1-A5 (i.e., hallucination)?
- Is the CoT logically coherent and consistent?
- Does the CoT naturally lead to the answer?
- Is the CoT fluent and free of obvious contradictions?

Give [COT valid]: Yes if the CoT meets ALL criteria, otherwise No.

Respond exactly:
[QA valid]: Yes/No
[COT valid]: Yes/No"""


def build_judge_messages(item):
    """从样本构造 judge prompt。"""
    a1 = item.get("a1_description") or "N/A"
    a2 = item.get("a2_speech") or "N/A"
    a3 = item.get("a3_music") or "N/A"
    a4_raw = item.get("events") or item.get("evidence_events") or "N/A"
    a4 = _format_events(a4_raw)
    a5 = str(item.get("duration", "N/A"))

    question = item.get("question", "")
    choices = get_choices(item)
    answer = str(item.get("answer", "") or "").strip()
    cot = get_cot(item) or "(no CoT provided)"

    choices_str = json.dumps(choices, ensure_ascii=False) if choices else "N/A"

    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        a1=a1, a2=a2, a3=a3, a4=a4, a5=a5,
        question=question,
        choices=choices_str,
        answer=answer,
        cot=cot,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ──────────────────────────────────────────────
# 解析 Judge 输出
# ──────────────────────────────────────────────

QA_VALID_PATTERN = re.compile(
    r"\[?\s*QA\s+valid\s*\]?\s*:\s*(Yes|No)", re.IGNORECASE
)
COT_VALID_PATTERN = re.compile(
    r"\[?\s*COT\s+valid\s*\]?\s*:\s*(Yes|No)", re.IGNORECASE
)


def parse_judge(raw_text):
    """从模型输出中解析 qa_valid / cot_valid。"""
    if not raw_text:
        return None, None, "empty response"
    qa_match = QA_VALID_PATTERN.search(raw_text)
    cot_match = COT_VALID_PATTERN.search(raw_text)
    errors = []
    qa = None
    cot = None
    if not qa_match:
        errors.append("qa_valid not found")
    else:
        qa = qa_match.group(1).lower() == "yes"
    if not cot_match:
        errors.append("cot_valid not found")
    else:
        cot = cot_match.group(1).lower() == "yes"
    parse_error = "; ".join(errors) if errors else None
    return qa, cot, parse_error


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="论文式 Audio-QA CoT Filtering (Judge)"
    )
    parser.add_argument("--input_jsonl", required=True, help="输入候选数据 JSONL")
    parser.add_argument("--output_jsonl", required=True, help="输出判定结果 JSONL")
    parser.add_argument("--report_json", required=True, help="统计报告 JSON")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key_env", default="ECHO_JUDGE_API_KEY",
                        help="API key 环境变量名 (默认 ECHO_JUDGE_API_KEY)")
    parser.add_argument("--model", default="deepseek-reasoner",
                        help="模型名 (默认 deepseek-reasoner)")
    parser.add_argument("--start_index", type=int, default=0, help="从第几行开始 (默认 0)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最多处理多少条 (0=全部, 默认 0)")
    parser.add_argument("--sleep_seconds", type=float, default=0.2,
                        help="每次 API 调用后等待秒数 (默认 0.2)")
    parser.add_argument("--timeout", type=int, default=120, help="API 超时秒数 (默认 120)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="API 失败最大重试次数 (默认 3)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="生成温度 (默认 0)")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="最大生成 token 数 (默认 128)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续跑，跳过 output_jsonl 中已有的 id (默认 true)")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry_failed", action="store_true", default=False,
                        help="重试已有 judge_error 的样本 (默认 false)")
    args = parser.parse_args()

    # ── API key ──
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"环境变量 {args.api_key_env} 未设置，请先 export {args.api_key_env}=your_key"
        )

    # ── 解析 base_url host (用于日志 & report) ──
    parsed_url = urlparse(args.base_url)
    judge_base_url_host = parsed_url.hostname or args.base_url

    # ── 初始化 client ──
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)

    # ── 读取输入 ──
    all_items = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line:
                all_items.append((line_no, json.loads(line)))

    sliced = all_items[args.start_index:]
    if args.max_samples > 0:
        sliced = sliced[: args.max_samples]

    # ── 断点续跑：读取已有 output ──
    existing_ids = set()
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    eid = get_id(existing)
                    if eid:
                        # 如果 retry_failed，跳过 judge_error 且 qa_valid/cot_valid 都为 null 的
                        if args.retry_failed:
                            if existing.get("qa_valid") is not None or existing.get("cot_valid") is not None:
                                existing_ids.add(eid)
                        else:
                            existing_ids.add(eid)
                except (json.JSONDecodeError, Exception):
                    continue
        print(f"[断点续跑] 已读取 {len(existing_ids)} 条已有判定")

    # ── 统计 ──
    stats = Counter()
    by_type_stats = {}

    def _inc_type(t, key):
        if t not in by_type_stats:
            by_type_stats[t] = Counter()
        by_type_stats[t][key] += 1

    # ── 输出文件 (append) ──
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_fp = open(args.output_jsonl, "a", encoding="utf-8")

    skip_count = 0
    total = len(sliced)

    for idx, (line_no, item) in enumerate(sliced):
        sample_id = get_id(item) or f"line_{line_no}"
        qa_type = get_type(item)

        # ── 跳过已有 ──
        if sample_id in existing_ids:
            skip_count += 1
            stats["skipped_existing"] += 1
            if (idx + 1) % 100 == 0:
                print(f"  [skip][{idx + 1}/{total}] ... 已跳过 {skip_count}")
            continue

        # ── 构造 prompt ──
        messages = build_judge_messages(item)

        # ── 调用 API ──
        judge_raw = ""
        judge_error = None
        qa_valid = None
        cot_valid = None

        for attempt in range(1 + args.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                judge_raw = (resp.choices[0].message.content or "").strip()

                # 解析
                qa_valid, cot_valid, parse_error = parse_judge(judge_raw)
                if parse_error:
                    stats["parse_errors"] += 1
                    _inc_type(qa_type, "parse_errors")
                    judge_error = parse_error
                else:
                    judge_error = None
                break  # 成功则跳出重试循环

            except Exception as e:
                judge_error = repr(e)
                if attempt < args.max_retries:
                    wait = 2 ** attempt
                    print(f"  [retry {attempt + 1}/{args.max_retries}] {sample_id}: {judge_error[:120]}")
                    time.sleep(wait)
                else:
                    stats["api_errors"] += 1
                    _inc_type(qa_type, "api_errors")
                    print(f"  [FAIL] {sample_id}: {judge_error[:160]}")

        # ── 累加统计 ──
        stats["processed"] += 1
        _inc_type(qa_type, "processed")
        if qa_valid is True:
            stats["qa_valid_yes"] += 1
            _inc_type(qa_type, "qa_valid_yes")
        elif qa_valid is False:
            stats["qa_valid_no"] += 1
            _inc_type(qa_type, "qa_valid_no")
        if cot_valid is True:
            stats["cot_valid_yes"] += 1
            _inc_type(qa_type, "cot_valid_yes")
        elif cot_valid is False:
            stats["cot_valid_no"] += 1
            _inc_type(qa_type, "cot_valid_no")
        if qa_valid is True and cot_valid is True:
            stats["both_valid"] += 1
            _inc_type(qa_type, "both_valid")
        if qa_valid is True and cot_valid is False:
            stats["qa_valid_yes_cot_no"] += 1
            _inc_type(qa_type, "qa_valid_yes_cot_no")
        if qa_valid is False:
            stats["qa_invalid"] += 1

        # ── 写入输出 ──
        output_record = dict(item)
        output_record["qa_valid"] = qa_valid
        output_record["cot_valid"] = cot_valid
        output_record["judge_raw"] = judge_raw
        output_record["judge_error"] = judge_error
        output_record["judge_model"] = args.model
        output_record["judge_base_url_host"] = judge_base_url_host
        output_record["judge_timestamp"] = datetime.now(timezone.utc).isoformat()

        out_fp.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        out_fp.flush()

        # ── 进度 ──
        if (idx + 1) % 10 == 0 or idx == 0:
            pct = (idx + 1) / total * 100 if total else 0
            print(
                f"  [{idx + 1}/{total} ({pct:.0f}%)] "
                f"id={sample_id} type={qa_type} "
                f"QA={'Y' if qa_valid is True else ('N' if qa_valid is False else '?')} "
                f"COT={'Y' if cot_valid is True else ('N' if cot_valid is False else '?')} "
                f"{'ERR' if judge_error else 'OK'}"
            )

        # ── 限速 ──
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    out_fp.close()

    # ── Report ──
    total_input = len(sliced) + skip_count
    report = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "judge_config": {
            "base_url_host": judge_base_url_host,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
            "sleep_seconds": args.sleep_seconds,
            "resume": args.resume,
            "retry_failed": args.retry_failed,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_input": total_input,
        "processed": stats["processed"],
        "skipped_existing": stats["skipped_existing"],
        "api_errors": stats.get("api_errors", 0),
        "parse_errors": stats.get("parse_errors", 0),
        "qa_valid_yes": stats.get("qa_valid_yes", 0),
        "qa_valid_no": stats.get("qa_valid_no", 0),
        "cot_valid_yes": stats.get("cot_valid_yes", 0),
        "cot_valid_no": stats.get("cot_valid_no", 0),
        "both_valid": stats.get("both_valid", 0),
        "qa_valid_yes_cot_no": stats.get("qa_valid_yes_cot_no", 0),
        "qa_invalid": stats.get("qa_invalid", 0),
        "by_type_stats": {
            k: dict(v) for k, v in sorted(by_type_stats.items())
        },
    }

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"Judge 完成")
    print(f"  总输入:     {total_input}")
    print(f"  已处理:     {stats['processed']}")
    print(f"  跳过(已有): {stats['skipped_existing']}")
    print(f"  API 错误:   {stats.get('api_errors', 0)}")
    print(f"  解析错误:   {stats.get('parse_errors', 0)}")
    print(f"  QA valid Y: {stats.get('qa_valid_yes', 0)}")
    print(f"  QA valid N: {stats.get('qa_valid_no', 0)}")
    print(f"  COT valid Y:{stats.get('cot_valid_yes', 0)}")
    print(f"  COT valid N:{stats.get('cot_valid_no', 0)}")
    print(f"  both_valid: {stats.get('both_valid', 0)}")
    print(f"  QA+Y COT-N: {stats.get('qa_valid_yes_cot_no', 0)}")
    print(f"  QA invalid: {stats.get('qa_invalid', 0)}")
    print(f"  输出:       {args.output_jsonl}")
    print(f"  报告:       {args.report_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()

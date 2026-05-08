#!/usr/bin/env python3
"""
论文式 Audio-QA CoT Filtering (Judge) 脚本。

对候选 QA-CoT 数据调用 OpenAI-compatible API 做语义重评，
输出每条样本的 [QA valid] / [COT valid] 判定，用于后续 SFT/RL 数据分流。

支持自定义 base_url / model / api_key_env，兼容第三方中转 API。
支持断点续跑、失败重试、并发调用。

用法：
  python scripts/judge_eaqa_candidates.py \
      --input_jsonl data.jsonl \
      --output_jsonl judged.jsonl \
      --report_json report.json \
      --base_url https://api.xxx.com/v1 \
      --api_key_env ECHO_JUDGE_API_KEY \
      --model deepseek-reasoner \
      --concurrency 12
"""

import argparse
import json
import os
import re
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse

from openai import OpenAI


# ──────────────────────────────────────────────
# 字段兼容提取
# ──────────────────────────────────────────────

def _safe_json_loads(val):
    if isinstance(val, (dict, list)):
        return val
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(val.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        pass
    return val


def get_field(item, *keys, default=None):
    for k in keys:
        if k in item and item[k] is not None:
            v = item[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            elif not isinstance(v, str):
                return v
    return default


def get_choices(item):
    raw = get_field(item, "choices", "multi_choice", "options")
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, list):
        return parsed
    m = re.findall(r"'([^']*)'", str(raw))
    if m:
        return m
    return [str(raw)]


def get_cot(item):
    for k in ("cot", "response", "assistant_response", "output"):
        v = item.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
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
    r = item.get("raw_response", "")
    if isinstance(r, str) and r.strip():
        return r.strip()
    return ""


def get_type(item):
    return get_field(item, "type", "qa_type", "skeleton_type", default="unknown")


def get_id(item):
    return get_field(item, "id", "skeleton_id", "segment_id", default="")


def _format_events(events_raw):
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
# 单条处理（供并发调用）
# ──────────────────────────────────────────────

def _call_api(client, args, messages):
    """调用 API，返回 (raw_text, error)。重试由外层负责。"""
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip(), None


def process_one_sample(item, line_no, args, client):
    """处理单条样本，返回结果 dict。"""
    sample_id = get_id(item) or f"line_{line_no}"
    qa_type = get_type(item)

    messages = build_judge_messages(item)
    judge_raw = ""
    judge_error = None
    is_api_error = False
    qa_valid = None
    cot_valid = None

    for attempt in range(1 + args.max_retries):
        try:
            judge_raw, err = _call_api(client, args, messages)
            qa_valid, cot_valid, parse_error = parse_judge(judge_raw)
            if parse_error:
                judge_error = parse_error
                is_api_error = False
            else:
                judge_error = None
            break
        except Exception as e:
            judge_error = repr(e)
            if attempt < args.max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                is_api_error = True

    output_record = dict(item)
    output_record["qa_valid"] = qa_valid
    output_record["cot_valid"] = cot_valid
    output_record["judge_raw"] = judge_raw
    output_record["judge_error"] = judge_error
    output_record["judge_model"] = args.model
    output_record["judge_sample_id"] = sample_id
    output_record["judge_timestamp"] = datetime.now(timezone.utc).isoformat()

    return {
        "sample_id": sample_id,
        "qa_type": qa_type,
        "qa_valid": qa_valid,
        "cot_valid": cot_valid,
        "judge_error": judge_error,
        "is_api_error": is_api_error,
        "output_record": output_record,
    }


# ──────────────────────────────────────────────
# QPS 限速器
# ──────────────────────────────────────────────

class RateLimiter:
    """简单的 QPS 限速器。"""
    def __init__(self, qps_limit):
        self.lock = threading.Lock()
        self.last_ts = 0.0
        self.min_interval = 1.0 / qps_limit if qps_limit > 0 else 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.time()
            elapsed = now - self.last_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_ts = time.time()


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="论文式 Audio-QA CoT Filtering (Judge)"
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key_env", default="ECHO_JUDGE_API_KEY")
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry_failed", action="store_true", default=False)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发线程数 (默认 1)")
    parser.add_argument("--qps_limit", type=float, default=0.0,
                        help="每秒请求数上限，0=不限 (默认 0)")
    parser.add_argument("--progress_every", type=int, default=20,
                        help="每 N 条打印一次进度 (默认 20)")
    args = parser.parse_args()

    # ── API key ──
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"环境变量 {args.api_key_env} 未设置，请先 export {args.api_key_env}=your_key"
        )
    parsed_url = urlparse(args.base_url)
    judge_base_url_host = parsed_url.hostname or args.base_url

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
    total = len(sliced)

    # ── 断点续跑 ──
    existing_ids = set()
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    eid = get_id(existing) or existing.get("judge_sample_id")
                    if eid:
                        if args.retry_failed:
                            if existing.get("qa_valid") is not None or existing.get("cot_valid") is not None:
                                existing_ids.add(eid)
                        else:
                            existing_ids.add(eid)
                except (json.JSONDecodeError, Exception):
                    continue
        print(f"[断点续跑] 已读取 {len(existing_ids)} 条已有判定")

    # ── 准备 ---
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 检查是否有任务需要做
    todo = [(line_no, item) for line_no, item in sliced
            if (get_id(item) or f"line_{line_no}") not in existing_ids]

    print(f"总输入: {total}, 已有: {len(existing_ids)}, 待处理: {len(todo)}")

    # ── 统计（在所有分支前初始化） ──
    stats = Counter()
    by_type_stats = {}
    stats_lock = threading.RLock()
    skip_count = total - len(todo)

    def _inc_type(t, key):
        with stats_lock:
            if t not in by_type_stats:
                by_type_stats[t] = Counter()
            by_type_stats[t][key] += 1

    if not todo:
        print("没有待处理的样本，直接生成 report。")
    else:
        file_lock = threading.Lock()
        rate_limiter = RateLimiter(args.qps_limit) if args.qps_limit > 0 else None

        client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
        done_count = 0

        def write_result(result):
            """写结果 & 更新统计（线程安全）。"""
            nonlocal done_count
            with stats_lock:
                done_count += 1
                stats["processed"] += 1
                qa_type = result["qa_type"]
                qa = result["qa_valid"]
                cot = result["cot_valid"]
                jerr = result["judge_error"]
                is_api_err = result.get("is_api_error", False)

                if qa is True:
                    stats["qa_valid_yes"] += 1
                    _inc_type(qa_type, "qa_valid_yes")
                elif qa is False:
                    stats["qa_valid_no"] += 1
                    _inc_type(qa_type, "qa_valid_no")
                if cot is True:
                    stats["cot_valid_yes"] += 1
                    _inc_type(qa_type, "cot_valid_yes")
                elif cot is False:
                    stats["cot_valid_no"] += 1
                    _inc_type(qa_type, "cot_valid_no")
                if qa is True and cot is True:
                    stats["both_valid"] += 1
                    _inc_type(qa_type, "both_valid")
                if qa is True and cot is False:
                    stats["qa_valid_yes_cot_no"] += 1
                    _inc_type(qa_type, "qa_valid_yes_cot_no")
                if qa is False:
                    stats["qa_invalid"] += 1
                if jerr and is_api_err:
                    stats["api_errors"] += 1
                    _inc_type(qa_type, "api_errors")
                elif jerr and not is_api_err:
                    stats["parse_errors"] += 1
                    _inc_type(qa_type, "parse_errors")

                # 写文件
                with file_lock:
                    with open(args.output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result["output_record"], ensure_ascii=False) + "\n")

                # 进度
                if done_count % args.progress_every == 0 or done_count == 1:
                    pct = done_count / len(todo) * 100 if todo else 0
                    errs = stats.get("api_errors", 0) + stats.get("parse_errors", 0)
                    print(
                        f"  [{done_count}/{len(todo)} ({pct:.0f}%)] "
                        f"QA=Y:{stats.get('qa_valid_yes',0)} N:{stats.get('qa_valid_no',0)} "
                        f"COT=Y:{stats.get('cot_valid_yes',0)} N:{stats.get('cot_valid_no',0)} "
                        f"err={errs}"
                    )

        def worker_task(line_no, item):
            """线程工作函数。"""
            sample_id = get_id(item) or f"line_{line_no}"
            if sample_id in existing_ids:
                return
            if rate_limiter:
                rate_limiter.wait()
            result = process_one_sample(item, line_no, args, client)
            write_result(result)

        # ── 并发执行 ──
        actual_concurrency = min(args.concurrency, len(todo)) if todo else 1
        print(f"启动并发: {actual_concurrency} 线程, 处理 {len(todo)} 条")

        if args.concurrency <= 1:
            for line_no, item in todo:
                worker_task(line_no, item)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(worker_task, line_no, item)
                           for line_no, item in todo}
                for future in as_completed(futures):
                    exc = future.exception()
                    if exc:
                        print(f"  [线程异常] {repr(exc)[:200]}")

    # ── Report ──
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
            "concurrency": args.concurrency,
            "qps_limit": args.qps_limit,
            "resume": args.resume,
            "retry_failed": args.retry_failed,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_input": total,
        "processed": stats.get("processed", 0),
        "skipped_existing": skip_count,
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
    print(f"  总输入:     {total}")
    print(f"  已处理:     {stats.get('processed', 0)}")
    print(f"  跳过(已有): {skip_count}")
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

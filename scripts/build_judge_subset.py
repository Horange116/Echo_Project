#!/usr/bin/env python3
"""
分层抽样脚本 — 按"数据来源段 + 题型"构造 judge subset。

用于从候选数据中按 source_group（deepseek_polished / template_or_unpolished）
和 type 分层抽样，确保各来源段、各题型都有代表性样本。

用法：
  python scripts/build_judge_subset.py \
      --input_jsonl /path/to/candidates.jsonl \
      --output_jsonl output/judge/judge_subset.jsonl \
      --report_json output/judge/judge_subset_report.json \
      --polished_range 0:3000 \
      --polished_per_type 38 \
      --template_per_type 38 \
      --max_total 600 \
      --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter, OrderedDict


def parse_range(s):
    """解析 "0:3000" -> (0, 3000)。"""
    parts = s.split(":")
    return int(parts[0]), int(parts[1])


def is_in_range(index, range_str):
    """检查 index 是否在范围内。"""
    lo, hi = parse_range(range_str)
    return lo <= index < hi


def get_type_from_item(item, field_candidates):
    """按优先级从 item 中取 type 字段。"""
    for key in field_candidates:
        v = item.get(key)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
        if v:
            return str(v)
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="按数据来源段 + 题型分层抽样"
    )
    parser.add_argument("--input_jsonl", required=True, help="输入候选数据 JSONL")
    parser.add_argument("--output_jsonl", required=True, help="输出 subset JSONL")
    parser.add_argument("--report_json", required=True, help="统计报告 JSON")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--polished_range", default="0:3000",
                        help="DeepSeek 润色数据的 index 范围 (默认 0:3000)")
    parser.add_argument("--polished_per_type", type=int, default=38,
                        help="润色段每类抽多少条 (默认 38)")
    parser.add_argument("--template_per_type", type=int, default=38,
                        help="模板段每类抽多少条 (默认 38)")
    parser.add_argument("--max_total", type=int, default=600,
                        help="最多输出多少条 (默认 600)")
    parser.add_argument("--type_field_candidates", default="type,qa_type,skeleton_type",
                        help="type 字段优先级，逗号分隔 (默认 type,qa_type,skeleton_type)")
    args = parser.parse_args()

    type_fields = [t.strip() for t in args.type_field_candidates.split(",")]

    # ── 读取输入 ──
    all_items = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                all_items.append((idx, json.loads(line)))

    # ── 分组 ──
    lo, hi = parse_range(args.polished_range)
    pools = {
        "deepseek_polished": OrderedDict(),
        "template_or_unpolished": OrderedDict(),
    }

    for idx, item in all_items:
        if lo <= idx < hi:
            group = "deepseek_polished"
        else:
            group = "template_or_unpolished"

        qa_type = get_type_from_item(item, type_fields)
        if qa_type not in pools[group]:
            pools[group][qa_type] = []
        pools[group][qa_type].append((idx, item))

    # ── 可用数量报告 ──
    available = {}
    for group in pools:
        available[group] = {}
        for t, items in pools[group].items():
            available[group][t] = len(items)

    # ── 抽样 ──
    rng = random.Random(args.seed)
    per_type = {
        "deepseek_polished": args.polished_per_type,
        "template_or_unpolished": args.template_per_type,
    }

    selected = []  # [(idx, item, group), ...]
    selected_by_group = Counter()
    selected_by_group_type = Counter()

    for group in ["deepseek_polished", "template_or_unpolished"]:
        for t in sorted(pools[group].keys()):
            pool = pools[group][t]
            n = min(per_type[group], len(pool))
            rng.shuffle(pool)
            for idx, item in pool[:n]:
                selected.append((idx, item, group))
                selected_by_group[group] += 1
                selected_by_group_type[f"{group}/{t}"] += 1

    # ── max_total 截断 ──
    if args.max_total > 0 and len(selected) > args.max_total:
        rng.shuffle(selected)
        selected = selected[:args.max_total]
        # 重新统计截断后的分布
        selected_by_group = Counter()
        selected_by_group_type = Counter()
        for idx, item, group in selected:
            selected_by_group[group] += 1
            t = get_type_from_item(item, type_fields)
            selected_by_group_type[f"{group}/{t}"] += 1

    # ── 排序：按原始 index 排序，保持可追溯 ──
    selected.sort(key=lambda x: x[0])

    # ── 写入输出 ──
    out_dir = os.path.dirname(args.output_jsonl) if args.output_jsonl else ""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for idx, item, group in selected:
            out = dict(item)
            out["original_index"] = idx
            out["judge_source_group"] = group
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ── Report ──
    # 重新按 source_group + type 统计可用
    available_by_sg_type = {}
    for group in pools:
        for t, items in pools[group].items():
            key = f"{group}/{t}"
            available_by_sg_type[key] = len(items)

    report = {
        "input_jsonl": args.input_jsonl,
        "polished_range": args.polished_range,
        "config": {
            "polished_per_type": args.polished_per_type,
            "template_per_type": args.template_per_type,
            "max_total": args.max_total,
            "seed": args.seed,
            "type_field_candidates": args.type_field_candidates,
        },
        "total_input": len(all_items),
        "total_selected": len(selected),
        "selected_by_source_group": dict(selected_by_group),
        "selected_by_source_group_and_type": dict(selected_by_group_type),
        "available_by_source_group_and_type": available_by_sg_type,
    }

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 打印摘要 ──
    print(f"Subset 构建完成")
    print(f"  总输入:     {len(all_items)}")
    print(f"  选取:       {len(selected)}")
    print(f"  润色段:     {selected_by_group.get('deepseek_polished', 0)}")
    print(f"  模板段:     {selected_by_group.get('template_or_unpolished', 0)}")
    print(f"  输出:       {args.output_jsonl}")
    print(f"  报告:       {args.report_json}")
    for key, count in sorted(selected_by_group_type.items()):
        print(f"    {key}: {count}")


if __name__ == "__main__":
    main()

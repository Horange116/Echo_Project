#!/usr/bin/env python3
"""
构建固定评估集清单 (eval_manifest.jsonl)。

从输入的 JSONL 中按 start_index + max_samples 抽取样本，
以固定 seed 打乱后输出，确保同样参数重复运行结果完全一致。

支持字段名兼容:
  question / choices / multi_choice / answer / audio_path / audios / type / qa_type
"""

import argparse
import json
import random
import re


def _parse_choices(val):
    """兼容 choices 可能是字符串列表或字符串表示。"""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        # 尝试 JSON 解析 (如 ["a", "b"])
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # 尝试 Python repr 风格的列表 (如 "['a', 'b']")
        try:
            parsed = json.loads(val.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        # 正则回退: 提取所有单引号内容
        m = re.findall(r"'([^']*)'", val)
        if m:
            return m
    return val


def build_item(line_no, raw):
    """从原始 JSON item 中提取标准化字段。"""
    # id
    item_id = (
        raw.get("skeleton_id")
        or raw.get("segment_id")
        or raw.get("id")
        or f"line_{line_no}"
    )

    # audio_path: 优先 audio_path 字段，回退 audios 列表的第一个
    audio_path = raw.get("audio_path") or ""
    if not audio_path:
        audios = raw.get("audios") or []
        if isinstance(audios, list) and len(audios) > 0:
            audio_path = audios[0]
        elif isinstance(audios, str):
            audio_path = audios

    # question
    question = raw.get("question") or ""

    # choices
    choices = raw.get("choices") or raw.get("multi_choice") or []
    choices = _parse_choices(choices)

    # answer
    answer = str(raw.get("answer") or raw.get("ground_truth") or "").strip()

    # type
    qa_type = raw.get("type") or raw.get("qa_type") or "unknown"

    return {
        "id": str(item_id),
        "audio_path": str(audio_path),
        "question": str(question),
        "choices": choices,
        "answer": answer,
        "type": str(qa_type),
        "original_index": line_no,
    }


def main():
    parser = argparse.ArgumentParser(
        description="构建固定评估集清单 (eval_manifest.jsonl)"
    )
    parser.add_argument(
        "--input_jsonl", required=True,
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output_manifest", required=True,
        help="输出的 eval_manifest.jsonl 路径"
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="从第几行开始取 (0-indexed, 默认 0)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="最多取多少条 (默认 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，用于打乱顺序 (默认 42)"
    )
    args = parser.parse_args()

    # 读取输入
    all_items = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line:
                all_items.append((line_no, json.loads(line)))

    # start_index 越界保护
    if args.start_index >= len(all_items):
        print(f"警告: start_index={args.start_index} 超出文件总行数 {len(all_items)}，输出为空。")
        selected = []
    else:
        sliced = all_items[args.start_index:]
        rng = random.Random(args.seed)
        rng.shuffle(sliced)
        selected = sliced[: args.max_samples]

    # 写入 manifest
    manifest_items = []
    for line_no, raw in selected:
        manifest_items.append(build_item(line_no, raw))

    with open(args.output_manifest, "w", encoding="utf-8") as f:
        for item in manifest_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计各类别数量
    type_counts = {}
    for item in manifest_items:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"manifest 已写入: {args.output_manifest}")
    print(f"  总样本数: {len(manifest_items)}")
    print(f"  输入文件: {args.input_jsonl}")
    print(f"  start_index: {args.start_index}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  seed: {args.seed}")
    print(f"  类型分布: {json.dumps(type_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

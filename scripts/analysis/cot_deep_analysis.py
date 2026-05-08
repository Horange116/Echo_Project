#!/usr/bin/env python3
"""
Chain-of-Thought 深度分析脚本
支持两种输入格式：
  1) 有显式 cot 字段 + qa_type 的 JSONL
  2) messages 格式（assistant 回复中含 <think>），从 source_skeleton 提取 qa_type
"""

import json
import re
import sys
from collections import Counter, defaultdict

# ── config ──
INPUT = sys.argv[1] if len(sys.argv) > 1 else "/home/s2025244189/s2025244265/Projects/Echo_Project/output/judge/candidates_merged_full.jsonl"

SEG_PATTERN = re.compile(r"<seg>\s*[\d.]+\s*,\s*[\d.]+\s*</seg>")
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOKEN_SPLIT = re.compile(r"[\w']+|[^\w\s]")

def tokenize(text: str) -> list[str]:
    return TOKEN_SPLIT.findall(text)

def get_think_text(text: str) -> str:
    m = THINK_PATTERN.search(text)
    return m.group(1).strip() if m else ""

def normalize_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_think_from_messages(item: dict) -> str:
    msgs = item.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "").strip()
    return ""

def extract_qa_type_from_source(item: dict) -> str:
    ss = item.get("source_skeleton", {})
    if isinstance(ss, dict):
        return ss.get("qa_type", "unknown")
    return "unknown"

def extract_id(item: dict) -> str:
    for k in ("id", "skeleton_id"):
        v = item.get(k)
        if v:
            return str(v)
    return ""

# ── 加载 ──
print(f"Loading data from {INPUT} ...")
data = []
with open(INPUT) as f:
    for line in f:
        data.append(json.loads(line))
print(f"Total lines: {len(data)}")

# 提取 CoT 文本
has_explicit_cot = data[0].get("cot") if data else None
has_messages = data[0].get("messages") if data else None

samples = []
if has_explicit_cot:
    print("Detected format: explicit 'cot' field")
    for item in data:
        cot = item.get("cot", "")
        if not cot:
            continue
        think = get_think_text(cot)
        if not think:
            continue
        samples.append({
            "id": extract_id(item),
            "qa_type": item.get("qa_type", "unknown"),
            "cot": cot,
            "think": think,
            "tokens": tokenize(think),
            "words": think.split(),
        })
elif has_messages:
    print("Detected format: 'messages' with assistant response")
    for item in data:
        assistant_text = extract_think_from_messages(item)
        if not assistant_text:
            continue
        think = get_think_text(assistant_text)
        if not think:
            continue
        samples.append({
            "id": extract_id(item),
            "qa_type": extract_qa_type_from_source(item),
            "cot": assistant_text,
            "think": think,
            "tokens": tokenize(think),
            "words": think.split(),
        })
else:
    # fallback: 遍历找任何含有 think 标签的文本字段
    print("Unknown format, searching for <think> in all string fields...")
    for item in data:
        for val in item.values():
            if isinstance(val, str) and "<think>" in val:
                think = get_think_text(val)
                if think:
                    samples.append({
                        "id": extract_id(item),
                        "qa_type": item.get("qa_type", extract_qa_type_from_source(item)),
                        "cot": val,
                        "think": think,
                        "tokens": tokenize(think),
                        "words": think.split(),
                    })
                break

print(f"Samples with valid CoT: {len(samples)}")
print(f"qa_type distribution: {dict(Counter(s['qa_type'] for s in samples))}\n")

# ═══════════════════════════════════════════════
# 1. 平均 CoT word count / token count
# ═══════════════════════════════════════════════
print("=" * 70)
print("1. 平均 CoT word count / token count")
print("=" * 70)
word_counts = [len(s["words"]) for s in samples]
token_counts = [len(s["tokens"]) for s in samples]
char_counts = [len(s["think"]) for s in samples]

avg_words = sum(word_counts) / len(word_counts)
avg_tokens = sum(token_counts) / len(token_counts)
avg_chars = sum(char_counts) / len(char_counts)
min_w, max_w = min(word_counts), max(word_counts)
min_t, max_t = min(token_counts), max(token_counts)

print(f"  Word count:  avg={avg_words:.1f},  min={min_w}, max={max_w}")
print(f"  Token count: avg={avg_tokens:.1f},  min={min_t}, max={max_t}")
print(f"  Char count:  avg={avg_chars:.1f}")
print()

# ═══════════════════════════════════════════════
# 2. 每个 type 的平均 CoT 长度
# ═══════════════════════════════════════════════
print("=" * 70)
print("2. 每个 type 的平均 CoT 长度")
print("=" * 70)
type_stats = defaultdict(list)
for s in samples:
    type_stats[s["qa_type"]].append(len(s["words"]))

for t in sorted(type_stats.keys()):
    vals = type_stats[t]
    avg = sum(vals) / len(vals)
    print(f"  {t:25s}: count={len(vals):5d}, avg_words={avg:.1f}")
print()

# ═══════════════════════════════════════════════
# 3. 不同 CoT 模板/句式的重复率
# ═══════════════════════════════════════════════
# 将 CoT 泛化为句式模板：保留句式结构，替换具体数值和实体
print("=" * 70)
print("3. 不同 CoT 模板/句式的重复率")
print("=" * 70)

def generalize_template(text: str) -> str:
    """将 CoT 泛化为句式模板"""
    # 替换具体数字（包括小数）为 NUM
    text = re.sub(r'\b\d+\.?\d*\b', 'NUM', text)
    # 替换 <seg> 内时间为 SEG
    text = re.sub(r'<seg>\s*[\d.]+\s*,\s*[\d.]+\s*</seg>', '<seg>SEG</seg>', text)
    return text

templates = Counter()
for s in samples:
    tpl = generalize_template(s["think"])
    templates[tpl] += 1

total = len(samples)
unique_templates = len(templates)
template_repeat_count = sum(1 for v in templates.values() if v > 1)
template_repeat_rate = template_repeat_count / total * 100
top_repeated = templates.most_common(15)

print(f"  总样本数: {total}")
print(f"  唯一模板数: {unique_templates}")
print(f"  重复模板数: {template_repeat_count} ({template_repeat_rate:.1f}%)")
print(f"  模板重复率 (1 - 唯一/总数): {(1 - unique_templates/total)*100:.1f}%")
print()
print("  Top-15 高频模板:")
for tpl, cnt in top_repeated:
    pct = cnt / total * 100
    preview = tpl[:100] + "..." if len(tpl) > 100 else tpl
    print(f"    [{cnt:5d} ({pct:.1f}%)] {preview}")
print()

# ═══════════════════════════════════════════════
# 4. 开头句式 top-20
# ═══════════════════════════════════════════════
print("=" * 70)
print("4. 开头句式 top-20")
print("=" * 70)

def get_opening_pattern(text: str, n_words: int = 5) -> str:
    """提取前 n_words 个词作为开头句式（数值已泛化）"""
    words = generalize_template(text).split()[:n_words]
    return " ".join(words)

openings = Counter()
for s in samples:
    opening = get_opening_pattern(s["think"], 5)
    openings[opening] += 1

print(f"  唯一开头句式: {len(openings)}")
print(f"  Top-20 开头句式:")
for op, cnt in openings.most_common(20):
    pct = cnt / total * 100
    print(f"    [{cnt:5d} ({pct:.1f}%)] {op}")
print()

# ═══════════════════════════════════════════════
# 5. <seg> 后分析句式 top-20
# ═══════════════════════════════════════════════
print("=" * 70)
print('5. <seg> 后分析句式 top-20')
print("=" * 70)

SEG_PLUS_PATTERN = re.compile(r"<seg>\s*[\d.]+\s*,\s*[\d.]+\s*</seg>\s*(.{10,60}?)(?:<seg|$)", re.DOTALL)

seg_followups = Counter()
for s in samples:
    for m in SEG_PLUS_PATTERN.finditer(s["think"]):
        followup = m.group(1).strip()
        if len(followup) >= 5:
            seg_followups[followup] += 1

print(f"  唯一 <seg> 后句式: {len(seg_followups)}")
print(f"  Top-20 <seg> 后分析句式:")
for text, cnt in seg_followups.most_common(20):
    print(f"    [{cnt:5d}] {text[:80]}")
print()

# ═══════════════════════════════════════════════
# 6. n-gram 重复率
# ═══════════════════════════════════════════════
print("=" * 70)
print("6. n-gram 重复率")
print("=" * 70)

def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

for n in [2, 3, 4, 5]:
    all_ngrams: list[tuple] = []
    for s in samples:
        all_ngrams.extend(get_ngrams(s["tokens"], n))
    ngram_total = len(all_ngrams)
    ngram_unique = len(set(all_ngrams))
    repeat_rate = (1 - ngram_unique / ngram_total) * 100 if ngram_total else 0
    # 也计算重复 n-gram 占总 n-gram 的比例
    ngram_counter = Counter(all_ngrams)
    repeated_count = sum(v for v in ngram_counter.values() if v > 1)
    repeated_pct = repeated_count / ngram_total * 100 if ngram_total else 0
    most_common_ngram = ngram_counter.most_common(3)

    print(f"  {n}-gram: total={ngram_total}, unique={ngram_unique}, "
          f"unique_rate={ngram_unique/ngram_total*100:.1f}%, "
          f"repeat_rate={repeat_rate:.1f}%, "
          f"repeated_ngram_count_pct={repeated_pct:.1f}%")
    for ng, c in most_common_ngram:
        print(f"    最常见: {' '.join(ng)} (x{c})")
print()

# ═══════════════════════════════════════════════
# 7. 完全相同 CoT 数量
# ═══════════════════════════════════════════════
print("=" * 70)
print("7. 完全相同 CoT 数量")
print("=" * 70)

cot_exact = Counter()
for s in samples:
    cot_exact[s["think"]] += 1

exact_dup = sum(1 for v in cot_exact.values() if v > 1)
exact_dup_samples = sum(v for v in cot_exact.values() if v > 1)
exact_unique = len(cot_exact)

print(f"  唯一 CoT 数: {exact_unique}")
print(f"  重复 CoT 数: {exact_dup} (出现了 {exact_dup_samples} 次, "
      f"{exact_dup_samples/total*100:.1f}%)")
print(f"  去重后占比: {exact_unique/total*100:.1f}%")

# 展示最严重的重复
if cot_exact.most_common(1)[0][1] > 1:
    print("  重复次数最多的 CoT:")
    for cot_text, cnt in cot_exact.most_common(5):
        if cnt > 1:
            print(f"    x{cnt}: {cot_text[:100]}...")
print()

# ═══════════════════════════════════════════════
# 8. 近似相同 CoT 数量
# ═══════════════════════════════════════════════
print("=" * 70)
print("8. 近似相同 CoT 数量")
print("=" * 70)

# 使用归一化文本比较（去标点、去空格、转小写）
cot_norm = Counter()
for s in samples:
    cot_norm[normalize_text(s["think"])] += 1

norm_dup = sum(1 for v in cot_norm.values() if v > 1)
norm_dup_samples = sum(v for v in cot_norm.values() if v > 1)
norm_unique = len(cot_norm)

# 近似相同 = 归一化后相同但原文不同
approx_same = 0
approx_samples = 0
for norm, cnt in cot_norm.most_common():
    if cnt > 1:
        # 检查这些原文是否不完全一样
        origs = set()
        for s in samples:
            if normalize_text(s["think"]) == norm:
                origs.add(s["think"])
        if len(origs) > 1:
            approx_same += 1
            approx_samples += cnt

print(f"  归一化后唯一 CoT 数: {norm_unique}")
print(f"  归一化后重复 CoT 数: {norm_dup} (出现 {norm_dup_samples} 次, "
      f"{norm_dup_samples/total*100:.1f}%)")
print(f"  近似相同（原文不同但归一化后相同）: {approx_same} 组, "
      f"涉及 {approx_samples} 条 ({approx_samples/total*100:.1f}%)")
print()

# ═══════════════════════════════════════════════
# 9. 每条样本平均 segment 数
# ═══════════════════════════════════════════════
print("=" * 70)
print("9. 每条样本平均 segment 数")
print("=" * 70)

seg_counts = []
for s in samples:
    segs = SEG_PATTERN.findall(s["cot"])
    seg_counts.append(len(segs))

avg_seg = sum(seg_counts) / len(seg_counts)
seg_dist = Counter(seg_counts)
max_seg = max(seg_counts)
zero_seg = sum(1 for c in seg_counts if c == 0)
zero_seg_pct = zero_seg / len(seg_counts) * 100

print(f"  平均 segment 数: {avg_seg:.2f}")
print(f"  最大 segment 数: {max_seg}")
print(f"  无 segment 样本: {zero_seg} ({zero_seg_pct:.1f}%)")
print("  Segment 数分布:")
for n in sorted(seg_dist.keys())[:15]:
    pct = seg_dist[n] / len(seg_counts) * 100
    bar = "█" * int(pct / 2)
    print(f"    {n} segs: {seg_dist[n]:5d} ({pct:5.1f}%) {bar}")
if len(seg_dist) > 15:
    print(f"    ... 还有 {len(seg_dist) - 15} 个更长尾的分布")
print()

# ═══════════════════════════════════════════════
# 10. segment 前后是否有自然语言解释
# ═══════════════════════════════════════════════
print("=" * 70)
print("10. segment 前后是否有自然语言解释")
print("=" * 70)

BEFORE_PATTERN = re.compile(r"(\S.{0,40}?)\s*<seg>", re.DOTALL)
AFTER_PATTERN = re.compile(r"</seg>\s*(.{0,40}?)(?:<seg>|NUM|$)", re.DOTALL)

has_before = 0
has_after = 0
has_both = 0
has_neither = 0

for s in samples:
    think = s["think"]
    segs = SEG_PATTERN.findall(think)
    if not segs:
        continue

    b = bool(BEFORE_PATTERN.search(think))
    a = bool(AFTER_PATTERN.search(think))

    if b and a:
        has_both += 1
    elif b:
        has_before += 1
    elif a:
        has_after += 1
    else:
        has_neither += 1

samples_with_seg = has_before + has_after + has_both + has_neither
print(f"  有 segment 的样本数: {samples_with_seg}")
print(f"  前后都有解释: {has_both} ({has_both/samples_with_seg*100:.1f}%)")
print(f"  仅在 segment 前有解释: {has_before} ({has_before/samples_with_seg*100:.1f}%)")
print(f"  仅在 segment 后有解释: {has_after} ({has_after/samples_with_seg*100:.1f}%)")
print(f"  segment 前后无解释: {has_neither} ({has_neither/samples_with_seg*100:.1f}%)")
print()

# ═══════════════════════════════════════════════
# 11. 每个 type 的 CoT 多样性
# ═══════════════════════════════════════════════
print("=" * 70)
print("11. 每个 type 的 CoT 多样性")
print("=" * 70)

for t in sorted(type_stats.keys()):
    type_samples = [s for s in samples if s["qa_type"] == t]
    n = len(type_samples)

    # n-gram diversity
    type_tokens = []
    type_words = []
    for s in type_samples:
        type_tokens.extend(s["tokens"])
        type_words.extend(s["words"])

    # 词汇多样性 (type-token ratio)
    ttr = len(set(type_tokens)) / len(type_tokens) if type_tokens else 0

    # 模板多样性
    type_tpls = Counter()
    for s in type_samples:
        type_tpls[generalize_template(s["think"])] += 1
    tpl_diversity = len(type_tpls) / n if n else 0

    # 相同 CoT 比例
    type_exact = Counter()
    for s in type_samples:
        type_exact[s["think"]] += 1
    exact_unique_ratio = len(type_exact) / n if n else 0
    exact_dup_ratio = sum(1 for v in type_exact.values() if v > 1) / n * 100 if n else 0

    # 开头句式多样性（前3词）
    type_openings = Counter()
    for s in type_samples:
        op = generalize_template(s["think"]).split()[:3]
        type_openings[" ".join(op)] += 1
    opening_diversity = len(type_openings) / n if n else 0

    # 平均长度
    avg_len = sum(len(s["words"]) for s in type_samples) / n

    print(f"  [{t:25s}] n={n:5d}")
    print(f"          avg_len={avg_len:.1f}, TTR={ttr:.3f}, "
          f"template_diversity={tpl_diversity:.3f}, "
          f"exact_unique_ratio={exact_unique_ratio:.3f}")
    print(f"          exact_dup_rate={exact_dup_ratio:.1f}%, "
          f"opening_diversity={opening_diversity:.3f}")

    # 找出该 type 最常见的模板
    tpl_tpl, tpl_cnt = type_tpls.most_common(1)[0]
    print(f"          最常见模板: {tpl_tpl[:80]}... (x{tpl_cnt}, {tpl_cnt/n*100:.1f}%)")
    print()

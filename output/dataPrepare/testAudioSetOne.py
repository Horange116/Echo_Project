# -*- coding: utf-8 -*-
import csv
import json
from collections import Counter, defaultdict

import pandas as pd


PARQUET_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/mnt/bn/wdq-base1/data/ALMs/EAQA/audios/AudioSetStrong/data/train-00000-of-00216.parquet"
STRONG_TSV_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/mnt/bn/wdq-base1/data/ALMs/EAQA/audios/AudioSetStrong/audioset_train_strong.tsv"


def strong_variants(segment_id: str):
    """
    AudioSet strong usual format:
      b0RFKhbpFJA_30000

    maybe for ID:
      b0RFKhbpFJA_30000
      b0RFKhbpFJA_000030
      b0RFKhbpFJA_030
      b0RFKhbpFJA_30
      b0RFKhbpFJA
    """
    segment_id = str(segment_id)
    variants = [segment_id]

    if "_" not in segment_id:
        return variants

    youtube_id, suffix = segment_id.rsplit("_", 1)
    if suffix.isdigit():
        ms = int(suffix)
        sec = ms // 1000
        variants.extend([
            f"{youtube_id}_{sec:06d}",
            f"{youtube_id}_{sec:03d}",
            f"{youtube_id}_{sec}",
            youtube_id,
        ])

    return variants


def pretty(obj, max_len=1200):
    text = repr(obj)
    if len(text) > max_len:
        text = text[:max_len] + " ...<truncated>"
    return text


def load_strong_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        print("TSV columns:", reader.fieldnames)

        required = {"segment_id", "start_time_seconds", "end_time_seconds", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"TSV missing required columns: {missing}")

        for row in reader:
            rows.append(row)

    segment_ids = sorted({r["segment_id"] for r in rows})
    return rows, segment_ids


def main():
    print("Loading parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    print("\n=== Parquet Basic Info ===")
    print("rows:", len(df))
    print("columns:", df.columns.tolist())

    print("\n=== Parquet Head Samples ===")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"\n---- row {i} ----")
        for col in df.columns:
            print(f"{col}: {pretty(row[col])}")

    if "video_id" not in df.columns:
        raise ValueError("Parquet has no `video_id` column. Need another ID column to align.")

    parquet_ids = [str(x) for x in df["video_id"].tolist()]
    parquet_id_set = set(parquet_ids)

    print("\n=== Parquet video_id Stats ===")
    print("unique parquet video_id:", len(parquet_id_set))
    print("first 20 parquet video_id:")
    for x in parquet_ids[:20]:
        print(" ", x)

    print("\nLoading strong TSV...")
    strong_rows, strong_ids = load_strong_tsv(STRONG_TSV_PATH)
    strong_id_set = set(strong_ids)

    print("\n=== Strong TSV Stats ===")
    print("rows:", len(strong_rows))
    print("unique segment_id:", len(strong_id_set))
    print("first 20 segment_id:")
    for x in strong_ids[:20]:
        print(" ", x)

    print("\n=== Match Rule Test ===")

    direct_hits = parquet_id_set & strong_id_set

    converted_6 = set()
    converted_3 = set()
    converted_plain = set()
    youtube_only = set()

    for sid in strong_id_set:
        if "_" in sid:
            yt, suffix = sid.rsplit("_", 1)
            if suffix.isdigit():
                sec = int(suffix) // 1000
                converted_6.add(f"{yt}_{sec:06d}")
                converted_3.add(f"{yt}_{sec:03d}")
                converted_plain.add(f"{yt}_{sec}")
                youtube_only.add(yt)

    print("direct segment_id match:", len(parquet_id_set & strong_id_set))
    print("ms -> 6-digit seconds match:", len(parquet_id_set & converted_6))
    print("ms -> 3-digit seconds match:", len(parquet_id_set & converted_3))
    print("ms -> plain seconds match:", len(parquet_id_set & converted_plain))
    print("youtube_id only match:", len(parquet_id_set & youtube_only))

    strong_to_parquet = {}
    rule_counter = Counter()

    for sid in strong_id_set:
        variants = strong_variants(sid)
        hits = [v for v in variants if v in parquet_id_set]

        if hits:
            strong_to_parquet[sid] = hits[0]

            if hits[0] == sid:
                rule_counter["direct"] += 1
            elif "_" in sid and hits[0] == sid.rsplit("_", 1)[0]:
                rule_counter["youtube_id_only"] += 1
            elif "_" in hits[0]:
                suffix = hits[0].rsplit("_", 1)[1]
                if len(suffix) == 6:
                    rule_counter["ms_to_6digit_seconds"] += 1
                elif len(suffix) == 3:
                    rule_counter["ms_to_3digit_seconds"] += 1
                else:
                    rule_counter["ms_to_plain_seconds"] += 1
            else:
                rule_counter["other"] += 1

    matched_strong = set(strong_to_parquet.keys())
    unmatched_strong = [x for x in strong_ids if x not in matched_strong]

    print("\n=== Best Variant Match Summary ===")
    print("matched strong segment_ids:", len(matched_strong))
    print("unmatched strong segment_ids:", len(unmatched_strong))
    print("rule counter:", dict(rule_counter))

    print("\nSample matched pairs:")
    for sid in list(strong_to_parquet.keys())[:30]:
        print(f"  {sid}  ->  {strong_to_parquet[sid]}")

    print("\nSample unmatched strong segment_ids:")
    for sid in unmatched_strong[:30]:
        print(" ", sid)

    print("\n=== Duplicate / Ambiguity Check ===")
    reverse = defaultdict(list)
    for sid, pid in strong_to_parquet.items():
        reverse[pid].append(sid)

    duplicated = {pid: sids for pid, sids in reverse.items() if len(sids) > 1}
    print("parquet IDs matched by multiple strong segment_ids:", len(duplicated))

    for pid, sids in list(duplicated.items())[:20]:
        print(f"  {pid}: {sids[:10]}")

    print("\n=== Events Field Check ===")
    if "events" in df.columns:
        non_null_events = df["events"].dropna()
        print("non-null events rows:", len(non_null_events))

        for i, ev in enumerate(non_null_events.head(5)):
            print(f"\n---- events sample {i} ----")
            print(pretty(ev, max_len=2000))
    else:
        print("No `events` column found.")

    print("\n=== Audio Field Check ===")
    if "audio" in df.columns:
        for i, au in enumerate(df["audio"].head(5)):
            print(f"\n---- audio sample {i} ----")
            print(type(au))
            print(pretty(au, max_len=2000))
    else:
        print("No `audio` column found.")

    print("\nDone.")


if __name__ == "__main__":
    main()

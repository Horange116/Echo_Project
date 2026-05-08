# -*- coding: utf-8 -*-
import csv
import json
import hashlib
from pathlib import Path

import pandas as pd


PARQUET_DIR = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/mnt/bn/wdq-base1/data/ALMs/EAQA/audios/AudioSetStrong/data/"
STRONG_TSV_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/mnt/bn/wdq-base1/data/ALMs/EAQA/audios/AudioSetStrong/audioset_train_strong.tsv"
OUTPUT_DIR = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch"

OVERWRITE = False


def load_video_to_segment_id(tsv_path):
    video_to_segment = {}
    ambiguous = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            segment_id = row["segment_id"]
            video_id = segment_id.rsplit("_", 1)[0]

            if video_id not in video_to_segment:
                video_to_segment[video_id] = segment_id
            elif video_to_segment[video_id] != segment_id:
                ambiguous.setdefault(video_id, set()).update([
                    video_to_segment[video_id],
                    segment_id,
                ])

    for video_id in ambiguous:
        video_to_segment.pop(video_id, None)

    return video_to_segment, ambiguous


def get_audio_ext(audio_bytes):
    if audio_bytes.startswith(b"fLaC"):
        return ".flac"
    if audio_bytes.startswith(b"RIFF"):
        return ".wav"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return ".mp3"
    return ".audio"


def normalize_events(events):
    output = []
    if events is None:
        return output

    for ev in events:
        output.append({
            "start": float(ev["start"]),
            "end": float(ev["end"]),
            "label": str(ev["event_name"]),
        })

    output.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return output


def infer_duration(events):
    if not events:
        return None
    return max(float(ev["end"]) for ev in events)


def safe_name(path):
    stem = Path(path).stem
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process_one_parquet(parquet_path, output_root, video_to_segment):
    parquet_path = Path(parquet_path)
    parquet_out_dir = output_root / safe_name(parquet_path)
    audio_dir = parquet_out_dir / "audios"
    metadata_path = parquet_out_dir / "aligned_metadata.jsonl"
    done_marker = parquet_out_dir / "done.marker"
    summary_path = parquet_out_dir / "summary.json"

    if done_marker.exists() and not OVERWRITE:
        return {
            "parquet": str(parquet_path),
            "status": "skipped_done",
            "output_dir": str(parquet_out_dir),
        }

    parquet_out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    written = 0
    skipped_no_mapping = 0
    skipped_no_audio = 0
    skipped_no_events = 0

    with open(metadata_path, "w", encoding="utf-8") as fout:
        for row_idx, row in df.iterrows():
            video_id = str(row["video_id"])

            if video_id not in video_to_segment:
                skipped_no_mapping += 1
                continue

            audio_obj = row["audio"]
            audio_bytes = audio_obj.get("bytes") if isinstance(audio_obj, dict) else None

            if not audio_bytes:
                skipped_no_audio += 1
                continue

            events = normalize_events(row["events"])
            if not events:
                skipped_no_events += 1
                continue

            ext = get_audio_ext(audio_bytes)
            segment_id = video_to_segment[video_id]
            audio_path = audio_dir / f"{segment_id}{ext}"

            if not audio_path.exists() or OVERWRITE:
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)

            labels = []
            if "human_labels" in row and row["human_labels"] is not None:
                labels = [str(x) for x in row["human_labels"]]

            item = {
                "source_parquet": str(parquet_path),
                "row_index": int(row_idx),
                "video_id": video_id,
                "segment_id": segment_id,
                "audio_path": str(audio_path),
                "duration": infer_duration(events),
                "labels": labels,
                "events": events,
            }

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "parquet": str(parquet_path),
        "output_dir": str(parquet_out_dir),
        "rows": int(len(df)),
        "written": written,
        "skipped_no_mapping": skipped_no_mapping,
        "skipped_no_audio": skipped_no_audio,
        "skipped_no_events": skipped_no_events,
        "metadata_path": str(metadata_path),
        "audio_dir": str(audio_dir),
    }

    write_json(summary_path, summary)

    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("done\n")

    return {
        "parquet": str(parquet_path),
        "status": "processed",
        "output_dir": str(parquet_out_dir),
        "written": written,
    }


def main():
    parquet_dir = Path(PARQUET_DIR)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.jsonl"

    print("Loading TSV mapping...")
    video_to_segment, ambiguous = load_video_to_segment_id(STRONG_TSV_PATH)
    print("usable video_id mappings:", len(video_to_segment))
    print("ambiguous video_ids:", len(ambiguous))

    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    print("parquet files:", len(parquet_files))

    with open(manifest_path, "a", encoding="utf-8") as manifest:
        for idx, parquet_path in enumerate(parquet_files, 1):
            print(f"[{idx}/{len(parquet_files)}] {parquet_path}")
            result = process_one_parquet(parquet_path, output_root, video_to_segment)
            print("  status:", result["status"], "written:", result.get("written", "-"))
            manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
            manifest.flush()

    print("done")
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import json
from pathlib import Path


INPUT_ROOT = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/all_summary_metadata.jsonl"
STATE_PATH = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/dataPreparedRes/audioset_jsonl_batch/merge_state.json"

REQUIRE_DONE_MARKER = True


def load_state(state_path):
    path = Path(state_path)
    if not path.exists():
        return {"merged_files": []}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path, state):
    path = Path(state_path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def iter_candidate_jsonl(input_root):
    input_root = Path(input_root)

    for path in sorted(input_root.rglob("aligned_metadata.jsonl")):
        if path.resolve() == Path(OUTPUT_JSONL).resolve():
            continue

        if REQUIRE_DONE_MARKER:
            done_marker = path.parent / "done.marker"
            if not done_marker.exists():
                continue

        yield path


def count_lines(path):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def validate_jsonl_line(line, source_path, line_no):
    line = line.strip()
    if not line:
        return None

    try:
        obj = json.loads(line)
    except Exception as e:
        raise ValueError(
            "Invalid JSON in %s line %d: %s" % (source_path, line_no, str(e))
        )

    required = {"source_parquet", "row_index", "video_id", "segment_id", "audio_path", "duration", "events"}
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(
            "Missing keys in %s line %d: %s" % (source_path, line_no, sorted(missing))
        )

    return obj


def main():
    input_root = Path(INPUT_ROOT)
    output_jsonl = Path(OUTPUT_JSONL)
    state_path = Path(STATE_PATH)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    merged_files = set(state.get("merged_files", []))

    candidates = list(iter_candidate_jsonl(input_root))
    print("candidate files:", len(candidates))
    print("already merged:", len(merged_files))

    appended_files = 0
    appended_lines = 0
    skipped_files = 0

    with open(output_jsonl, "a", encoding="utf-8") as fout:
        for idx, path in enumerate(candidates, 1):
            path_key = str(path.resolve())

            if path_key in merged_files:
                skipped_files += 1
                continue

            print("[%d/%d] merging %s" % (idx, len(candidates), path))

            file_lines = 0

            with open(path, "r", encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, 1):
                    obj = validate_jsonl_line(line, path, line_no)
                    if obj is None:
                        continue

                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    file_lines += 1
                    appended_lines += 1

            fout.flush()

            merged_files.add(path_key)
            state["merged_files"] = sorted(merged_files)
            state["last_merged_file"] = path_key
            state["total_merged_files"] = len(merged_files)
            state["total_output_lines"] = count_lines(output_jsonl)
            save_state(state_path, state)

            appended_files += 1
            print("  lines:", file_lines)

    print("done")
    print("output:", output_jsonl)
    print("appended files:", appended_files)
    print("appended lines:", appended_lines)
    print("skipped files:", skipped_files)
    print("state:", state_path)


if __name__ == "__main__":
    main()

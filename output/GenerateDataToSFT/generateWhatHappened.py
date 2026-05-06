# -*- coding: utf-8 -*-
import os
import re
import json
import traceback
from pathlib import Path

import torch
import librosa
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
LEGACY_PROJECT_ROOT = WORKSPACE_ROOT / "Echo_Project"

MODEL_PATH = str(WORKSPACE_ROOT / "Model_Env/Qwen2.5-Omni-7B")
INPUT_JSONL = str(PROJECT_ROOT / "output/dataPreparedRes/audioset_jsonl_batch/filtered_temporal_metadata.jsonl")
OUTPUT_JSONL = str(PROJECT_ROOT / "output/GeneratedData/metadata_with_qwen_audio_info.jsonl")
ERROR_JSONL = str(PROJECT_ROOT / "output/GeneratedData/metadata_with_qwen_audio_info_errors.jsonl")

MAX_ITEMS = None
BATCH_SIZE = 24
SAMPLE_RATE = 16000

MAX_NEW_TOKENS_A1 = 192
MAX_NEW_TOKENS_A2 = 128
MAX_NEW_TOKENS_A3 = 128

NO_SPEECH_TEXT = "There is no speech present in this audio."
NO_MUSIC_TEXT = "There is no music present in this audio."

A1_PROMPT = (
    "Describe the audio as comprehensively as possible. Focus only on audible content, "
    "including sound sources, temporal order, overlaps, and approximate timing when clear. "
    "Do not mention visual content. Do not ask or answer a question."
)

A2_PROMPT = (
    "Give information about speech only if speech is present in this audio, including transcript "
    "if understandable, speaker emotion, speaker gender, and spoken language. If no speech is "
    "present, output exactly: There is no speech present in this audio."
)

A3_PROMPT = (
    "Give information about music only if music is present in this audio, including music genre, "
    "instruments, vocals, tempo, and mood. If no music is present, output exactly: "
    "There is no music present in this audio."
)

SPEECH_KEYWORDS = [
    "speech", "speaking", "speaker", "conversation", "narration", "monologue",
    "human voice", "male speech", "female speech", "child speech", "shout",
    "whisper", "laughter", "crying", "babbling", "yell", "scream"
]

MUSIC_KEYWORDS = [
    "music", "singing", "song", "musical", "instrument", "guitar", "piano",
    "drum", "violin", "synthesizer", "bass", "choir", "rapping", "beat",
    "percussion", "flute", "trumpet", "harmonica", "keyboard"
]

BAD_A1_PATTERNS = [
    "the human:",
    "human:",
    "assistant:",
    "given the audio",
    "question:",
    "answer:",
    "<answer>",
    "[answer]",
    "<think>",
    "[think]",
    "the a s t o r y",
    "the a b",
    "the a v o l u t i o n",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_done_ids(path):
    done = set()
    if not Path(path).exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = obj.get("segment_id")
                if sid:
                    done.add(str(sid))
            except Exception:
                pass

    return done


def normalize_project_path(path):
    if not path:
        return path

    p = Path(path)
    if p.exists():
        return str(p)

    text = str(path)
    legacy = str(LEGACY_PROJECT_ROOT)
    current = str(PROJECT_ROOT)
    if text.startswith(legacy):
        candidate = Path(current + text[len(legacy):])
        if candidate.exists():
            return str(candidate)

    return text


def clean_response(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def bad_a1_description(text):
    if not text:
        return True

    low = text.lower().strip()

    if any(p in low for p in BAD_A1_PATTERNS):
        return True

    words = re.findall(r"[A-Za-z]+", text)
    digits = re.findall(r"\d", text)

    if len(words) < 6:
        return True

    if len(text) > 200 and len(words) < 10:
        return True

    if re.search(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", text):
        return True

    if words:
        one_letter_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        if one_letter_ratio > 0.35:
            return True

    if len(digits) > 50:
        return True

    if re.search(r"(?:1000){5,}", text):
        return True

    if re.search(r"(?:0\s*){20,}", text):
        return True

    return False


def general_description_fallback(item, max_events=12):
    events = item.get("events", []) or []
    labels = item.get("labels", []) or []

    parts = []

    if labels:
        parts.append(
            "The audio contains "
            + ", ".join(str(x) for x in labels[:8])
            + "."
        )

    if events:
        ev_parts = []
        for ev in events[:max_events]:
            ev_parts.append(
                "%s from %.3f to %.3fs"
                % (
                    str(ev.get("label", "")),
                    float(ev.get("start")),
                    float(ev.get("end")),
                )
            )
        parts.append("Key audible events include " + "; ".join(ev_parts) + ".")

    if not parts:
        return "The audio contains several audible events, but a detailed description is not reliably available."

    return " ".join(parts)


def collect_trigger_text(item):
    parts = []
    for x in item.get("labels", []) or []:
        parts.append(str(x))
    for ev in item.get("events", []) or []:
        parts.append(str(ev.get("label", "")))
    return " ".join(parts).lower()


def has_keyword(text, keywords):
    return any(k in text for k in keywords)


def no_speech_answer(text):
    t = text.lower()
    return "no speech" in t or "no spoken" in t or "no human speech" in t


def no_music_answer(text):
    t = text.lower()
    return "no music" in t or "no musical" in t


def select_events(item, keywords):
    out = []
    for ev in item.get("events", []) or []:
        label = str(ev.get("label", ""))
        low = label.lower()
        if any(k in low for k in keywords):
            out.append(ev)
    return out


def format_segments(events, max_events=8):
    parts = []
    for ev in events[:max_events]:
        parts.append(
            "%s at %.3f-%.3fs"
            % (
                str(ev.get("label", "")),
                float(ev.get("start")),
                float(ev.get("end")),
            )
        )
    return "; ".join(parts)


def speech_fallback(item):
    evs = select_events(item, SPEECH_KEYWORDS)
    if not evs:
        return NO_SPEECH_TEXT
    return (
        "Speech is present in the audio. The event labels indicate: "
        + format_segments(evs)
        + ". Transcript, speaker emotion, and spoken language are not reliably available."
    )


def music_fallback(item):
    evs = select_events(item, MUSIC_KEYWORDS)
    if not evs:
        return NO_MUSIC_TEXT
    return (
        "Music is present in the audio. The event labels indicate: "
        + format_segments(evs)
        + ". Genre and instruments are not reliably available."
    )


def load_model_and_processor():
    os.environ["QWEN_OMNI_SKIP_SPK"] = "1"

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if hasattr(model, "disable_talker"):
        model.disable_talker()

    model.eval()
    return model, processor


def build_text(processor, audio_ref, prompt):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_ref},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    if isinstance(text, list):
        if len(text) == 1 and isinstance(text[0], str):
            text = text[0]
        elif all(isinstance(x, str) for x in text):
            text = "".join(text)
        else:
            raise TypeError("apply_chat_template returned non-string list.")

    if not isinstance(text, str):
        raise TypeError("chat template output is not string: %s" % type(text))

    return text


def infer_batch(model, processor, audio_list, audio_refs, prompt, max_new_tokens):
    texts = [build_text(processor, ref, prompt) for ref in audio_refs]

    inputs = processor(
        text=texts,
        audio=audio_list,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        try:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_audio=False,
            )
        except TypeError:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_len:]
    responses = processor.batch_decode(new_tokens, skip_special_tokens=True)

    return [clean_response(x) for x in responses]


def infer_batch_safe(model, processor, audio_list, audio_refs, prompt, max_new_tokens):
    try:
        return infer_batch(model, processor, audio_list, audio_refs, prompt, max_new_tokens)
    except Exception:
        results = []
        for audio, audio_ref in zip(audio_list, audio_refs):
            results.extend(
                infer_batch(
                    model,
                    processor,
                    [audio],
                    [audio_ref],
                    prompt,
                    max_new_tokens,
                )
            )
        return results


def load_next_batch(reader, done_ids, max_items_left):
    batch = []

    for line_no, item in reader:
        sid = str(item.get("segment_id", ""))

        if not sid or sid in done_ids:
            continue

        audio_path = normalize_project_path(item.get("audio_path"))
        item["audio_path"] = audio_path
        if not audio_path or not os.path.exists(audio_path):
            append_jsonl(ERROR_JSONL, {
                "line_no": line_no,
                "segment_id": sid,
                "error": "missing audio_path",
                "audio_path": audio_path,
            })
            done_ids.add(sid)
            continue

        batch.append((line_no, item))

        if len(batch) >= BATCH_SIZE:
            break

        if max_items_left is not None and len(batch) >= max_items_left:
            break

    return batch


def process_batch(model, processor, batch):
    items = [x[1] for x in batch]

    audio_list = []
    audio_refs = []

    for _, item in batch:
        audio_path = item["audio_path"]
        audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        audio_list.append(audio_data)
        audio_refs.append(audio_path)

    a1_list = infer_batch_safe(
        model,
        processor,
        audio_list,
        audio_refs,
        A1_PROMPT,
        MAX_NEW_TOKENS_A1,
    )

    trigger_texts = [collect_trigger_text(item) for item in items]
    need_speech = [has_keyword(t, SPEECH_KEYWORDS) for t in trigger_texts]
    need_music = [has_keyword(t, MUSIC_KEYWORDS) for t in trigger_texts]

    a2_list = [NO_SPEECH_TEXT for _ in items]
    speech_indices = [i for i, x in enumerate(need_speech) if x]

    if speech_indices:
        speech_audios = [audio_list[i] for i in speech_indices]
        speech_refs = [audio_refs[i] for i in speech_indices]
        speech_outputs = infer_batch_safe(
            model,
            processor,
            speech_audios,
            speech_refs,
            A2_PROMPT,
            MAX_NEW_TOKENS_A2,
        )
        for idx, val in zip(speech_indices, speech_outputs):
            a2_list[idx] = val

    a3_list = [NO_MUSIC_TEXT for _ in items]
    music_indices = [i for i, x in enumerate(need_music) if x]

    if music_indices:
        music_audios = [audio_list[i] for i in music_indices]
        music_refs = [audio_refs[i] for i in music_indices]
        music_outputs = infer_batch_safe(
            model,
            processor,
            music_audios,
            music_refs,
            A3_PROMPT,
            MAX_NEW_TOKENS_A3,
        )
        for idx, val in zip(music_indices, music_outputs):
            a3_list[idx] = val

    outputs = []

    for i, item in enumerate(items):
        a1 = a1_list[i]
        a2 = a2_list[i]
        a3 = a3_list[i]

        a1_fixed = False
        a2_fixed = False
        a3_fixed = False

        if bad_a1_description(a1):
            a1 = general_description_fallback(item)
            a1_fixed = True

        if need_speech[i] and no_speech_answer(a2):
            a2 = speech_fallback(item)
            a2_fixed = True

        if need_music[i] and no_music_answer(a3):
            a3 = music_fallback(item)
            a3_fixed = True

        out = dict(item)
        out["a1_description"] = a1
        out["a2_speech"] = a2
        out["a3_music"] = a3
        out["qwen_info_flags"] = {
            "speech_prompt_used": bool(need_speech[i]),
            "music_prompt_used": bool(need_music[i]),
            "a1_fallback_used": bool(a1_fixed),
            "speech_fallback_used": bool(a2_fixed),
            "music_fallback_used": bool(a3_fixed),
        }

        outputs.append(out)

    return outputs


def main():
    done_ids = load_done_ids(OUTPUT_JSONL)
    print("already done:", len(done_ids))

    model, processor = load_model_and_processor()
    reader = read_jsonl(INPUT_JSONL)

    processed = 0
    failed = 0

    while True:
        max_left = None if MAX_ITEMS is None else MAX_ITEMS - processed
        if max_left is not None and max_left <= 0:
            break

        batch = load_next_batch(reader, done_ids, max_left)
        if not batch:
            break

        try:
            outputs = process_batch(model, processor, batch)

            for out in outputs:
                append_jsonl(OUTPUT_JSONL, out)
                done_ids.add(str(out["segment_id"]))
                processed += 1
                print("processed:", processed, "segment_id:", out["segment_id"])

        except Exception as e:
            failed += len(batch)
            err = traceback.format_exc()

            for line_no, item in batch:
                sid = str(item.get("segment_id", ""))
                append_jsonl(ERROR_JSONL, {
                    "line_no": line_no,
                    "segment_id": sid,
                    "audio_path": item.get("audio_path"),
                    "error": repr(e),
                    "traceback": err,
                })
                done_ids.add(sid)

            print("batch failed:", repr(e))
            print(err)

    print("done")
    print("processed:", processed)
    print("failed:", failed)
    print("output:", OUTPUT_JSONL)
    print("errors:", ERROR_JSONL)


if __name__ == "__main__":
    main()

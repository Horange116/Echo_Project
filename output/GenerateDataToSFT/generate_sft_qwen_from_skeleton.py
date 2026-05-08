# -*- coding: utf-8 -*-
import json
import re
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

import generate_sft_local_from_skeleton as local_gen


INPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/qa_skeleton.jsonl"
OUTPUT_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eaqa_sft_qwen_refined_1000_2999.jsonl"
ERROR_JSONL = "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eaqa_sft_qwen_refined_1000_2999_errors.jsonl"

# DeepSeek covers [0, 999], Qwen refinement covers [1000, 2999],
# local template covers [3000, 12999].
START_INDEX = 1000
MAX_ITEMS = 2000

# TODO: change this to your local Qwen3-Omni/Qwen text model path.
MODEL_PATH = "/hpai/aios3.0/private/user/s2025244189/wjy/Qwen3-Omni-Instruct"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
DO_SAMPLE = False
REPETITION_PENALTY = 1.05

# If Qwen output fails validation, keep the deterministic local template sample
# instead of throwing away the item. This keeps the 1000-2999 range complete.
FALLBACK_TO_TEMPLATE = True

SEG_PATTERN = re.compile(r"<seg>\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*</seg>")


PROMPT_TEMPLATE = """You are polishing an audio QA chain-of-thought sample.

You must preserve the factual content. Do not change the answer, choices, timestamps, or numerical result.
Only rewrite the wording to sound more natural and less templated.

You must output both the rewritten question and the full assistant answer. Do not omit the <think> block.
If you cannot improve the assistant answer, copy the template assistant answer exactly.

Output exactly this structure and nothing else:
<question>...</question>
<think>...</think><answer>__ANSWER__</answer>

Rules:
1. The final answer must be exactly: __ANSWER__
2. Keep every evidence timestamp as <seg>start, end</seg>. Do not invent or modify timestamps.
3. Do not change the choices or the answer.
4. Keep the same temporal relation as the original question.
5. Use plain ASCII words in calculations. Write "times" instead of the multiplication sign, and "about" instead of the approximately-equal sign.
6. Do not mention metadata, labels, annotations, or this prompt.
7. The <think> block is required. Never output only <question> and <answer>.

Original question:
__QUESTION__

Choices:
__CHOICES__

Correct answer:
__ANSWER__

Evidence segments:
__EVIDENCE__

Template assistant answer:
__TEMPLATE_ASSISTANT__

Remember: output must include <think>...</think><answer>__ANSWER__</answer>.
"""


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
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
                sid = obj.get("skeleton_id") or obj.get("segment_id")
                if sid:
                    done.add(str(sid))
            except Exception:
                pass
    return done


def clean_generated_text(text):
    text = str(text or "")
    replacements = {
        "\u00a1\u00c1": " times ",
        "\u00a1\u00d6": " about ",
        "\u00d7": " times ",
        "\u2248": " about ",
        "\u2264": " less than or equal to ",
        "\u2265": " greater than or equal to ",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"</think>\s+<answer>", "</think><answer>", text)
    text = re.sub(r"<think>\s+", "<think>", text)
    text = re.sub(r"\s+</think>", "</think>", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def format_evidence(events):
    rows = []
    for ev in events or []:
        rows.append({
            "segment": local_gen.seg(ev),
            "sound": ev.get("label"),
            "start": local_gen.fmt_time(ev["start"]),
            "end": local_gen.fmt_time(ev["end"]),
        })
    return json.dumps(rows, ensure_ascii=False)


def build_prompt(item, template_assistant):
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("__QUESTION__", str(item["question"]))
    prompt = prompt.replace("__CHOICES__", json.dumps(item["choices"], ensure_ascii=False))
    prompt = prompt.replace("__ANSWER__", str(item["answer"]))
    prompt = prompt.replace("__EVIDENCE__", format_evidence(item.get("evidence_events", [])))
    prompt = prompt.replace("__TEMPLATE_ASSISTANT__", template_assistant)
    return prompt


def extract_output(text):
    text = clean_generated_text(text)
    q_start = text.find("<question>")
    q_end = text.find("</question>")
    if q_start < 0 or q_end < 0 or q_end <= q_start:
        raise ValueError("missing question wrapper")
    question = text[q_start + len("<question>"):q_end].strip()
    assistant = text[q_end + len("</question>"):].strip()
    a_start = assistant.find("<think>")
    a_end = assistant.rfind("</answer>")
    if a_start < 0 or a_end < 0:
        raise ValueError("missing assistant wrapper")
    assistant = clean_generated_text(assistant[a_start:a_end + len("</answer>")])
    return question, assistant


def extract_question_only(text):
    text = clean_generated_text(text)
    q_start = text.find("<question>")
    q_end = text.find("</question>")
    if q_start < 0 or q_end < 0 or q_end <= q_start:
        return ""
    return text[q_start + len("<question>"):q_end].strip()


def validate_assistant(assistant, answer, evidence_events):
    if not assistant.startswith("<think>") or not assistant.endswith("</answer>"):
        return False, "bad assistant wrapper"
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if tag not in assistant:
            return False, "missing tag %s" % tag
    answer_inside = assistant.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if answer_inside != answer:
        return False, "answer mismatch"
    if not SEG_PATTERN.search(assistant):
        return False, "missing seg"
    expected_segs = {local_gen.seg(ev) for ev in evidence_events or []}
    used_segs = set(SEG_PATTERN.findall(assistant))
    if not used_segs.issubset(expected_segs):
        return False, "timestamp modified or invented"
    return True, ""


def validate_output(obj):
    ok, reason = local_gen.validate_output(obj)
    if not ok:
        return ok, reason
    assistant = obj["messages"][1]["content"]
    sk = obj.get("source_skeleton", {})
    return validate_assistant(assistant, sk.get("answer"), sk.get("evidence_events", []))


def load_model():
    # Qwen3-Omni is not a plain CausalLM. Prefer its Omni conditional-generation
    # class, and fall back to AutoModelForCausalLM for ordinary text-only Qwen.
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration

        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return processor, model, "qwen3_omni"
    except Exception as omni_error:
        print("Qwen3-Omni load failed, try AutoModelForCausalLM fallback:", repr(omni_error))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model, "causal_lm"


def get_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def call_qwen(processor_or_tokenizer, model, prompt, model_kind):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(processor_or_tokenizer, "apply_chat_template"):
        text = processor_or_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = processor_or_tokenizer(text=text, return_tensors="pt")
    inputs = inputs.to(get_input_device(model))

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if DO_SAMPLE:
        generate_kwargs["temperature"] = TEMPERATURE

    with torch.no_grad():
        if model_kind == "qwen3_omni":
            try:
                generated = model.generate(**inputs, return_audio=False, **generate_kwargs)
            except TypeError:
                generated = model.generate(**inputs, **generate_kwargs)
        else:
            eos_id = getattr(processor_or_tokenizer, "eos_token_id", None)
            if eos_id is not None:
                generate_kwargs["pad_token_id"] = eos_id
            generated = model.generate(**inputs, **generate_kwargs)

    if isinstance(generated, tuple):
        generated = generated[0]
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    return processor_or_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def build_output(item, processor_or_tokenizer, model, model_kind):
    template_out = local_gen.build_output(item)
    template_assistant = template_out["messages"][1]["content"]
    template_question = item["question"]
    prompt = build_prompt(item, template_assistant)
    raw = call_qwen(processor_or_tokenizer, model, prompt, model_kind)
    try:
        rewritten_question, assistant = extract_output(raw)
        ok, reason = validate_assistant(assistant, item["answer"], item.get("evidence_events", []))
        if not ok:
            raise ValueError(reason)
        out = {
            "skeleton_id": item.get("skeleton_id"),
            "segment_id": item.get("segment_id"),
            "messages": [
                {"role": "user", "content": local_gen.make_user_content({**item, "question": rewritten_question})},
                {"role": "assistant", "content": assistant},
            ],
            "audios": [item["audio_path"]],
            "source_skeleton": item,
            "generator": "qwen_refine",
            "raw_response": raw,
            "template_assistant": template_assistant,
        }
    except Exception as e:
        if not FALLBACK_TO_TEMPLATE:
            raise
        rewritten_question = extract_question_only(raw) or template_question
        out = {
            "skeleton_id": item.get("skeleton_id"),
            "segment_id": item.get("segment_id"),
            "messages": [
                {"role": "user", "content": local_gen.make_user_content({**item, "question": rewritten_question})},
                {"role": "assistant", "content": template_assistant},
            ],
            "audios": [item["audio_path"]],
            "source_skeleton": item,
            "generator": "qwen_question_only_template_cot_fallback" if rewritten_question != template_question else "local_template_fallback_after_qwen_error",
        }
        out["qwen_error"] = repr(e)
        out["raw_response"] = raw
        out["template_assistant"] = template_assistant
        out["rewritten_question"] = rewritten_question

    ok, reason = validate_output(out)
    if not ok:
        raise ValueError(reason)
    return out


def main():
    processor_or_tokenizer, model, model_kind = load_model()
    print("loaded model kind:", model_kind)
    done = load_done_ids(OUTPUT_JSONL)
    print("already done:", len(done))

    made = 0
    skipped = 0
    failed = 0

    for idx, item in read_jsonl(INPUT_JSONL):
        if idx < START_INDEX:
            skipped += 1
            continue
        if made >= MAX_ITEMS:
            break

        sid = str(item.get("skeleton_id") or item.get("segment_id"))
        if not sid:
            failed += 1
            append_jsonl(ERROR_JSONL, {"line_no": idx, "error": "missing skeleton_id", "item": item})
            continue
        if sid in done:
            skipped += 1
            continue

        try:
            out = build_output(item, processor_or_tokenizer, model, model_kind)
            append_jsonl(OUTPUT_JSONL, out)
            done.add(sid)
            made += 1
            if made % 50 == 0:
                print("made:", made)
        except Exception as e:
            failed += 1
            append_jsonl(ERROR_JSONL, {
                "line_no": idx,
                "skeleton_id": sid,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })

    print("done")
    print("made:", made)
    print("skipped:", skipped)
    print("failed:", failed)
    print("output:", OUTPUT_JSONL)
    print("errors:", ERROR_JSONL)


if __name__ == "__main__":
    main()

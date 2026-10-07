"""Incrementally label interaction states with Qwen3-VL-32B."""

import argparse
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


FIELDS = [
    "discriminative_attribute",
    "shared_attribute",
    "absent_or_invisible_attribute",
    "generic_background",
    "negative_or_unknown_answer",
    "redundant_with_history",
    "uncertain_visual_evidence",
    "likely_useful_for_retrieval",
    "main_attribute",
    "reason",
]


def build_prompt(record):
    history = "\n".join(
        f"{i + 1}. {question}" for i, question in enumerate(record.get("history_questions", []))
    ) or "(none)"
    return f"""You are auditing a question-answer step in an interactive person re-identification system.

The four images are ordered candidates for the target person. Image 1 is the current top-ranked candidate; Images 2-4 are similar candidates. Judge the question and the witness answer using the images and context. Focus on whether the answer creates evidence that can distinguish the target from the candidates and improve retrieval. Do not use the known rank change as evidence for your judgment.

Initial description:
{record.get('initial_query', '')}

Previous questions:
{history}

Current question:
{record.get('question', '')}

Witness answer:
{record.get('answer', '')}

Return exactly one JSON object and no markdown. Use boolean values for all boolean fields.
Required fields:
- discriminative_attribute: the question-answer pair distinguishes at least some candidates using a visible attribute.
- shared_attribute: the answered attribute is visibly shared by most or all candidates.
- absent_or_invisible_attribute: the attribute is absent, occluded, too small, or not reliably visible.
- generic_background: the question mainly asks about generic scene/background information.
- negative_or_unknown_answer: the answer is negative, unknown, or does not provide usable visual evidence.
- redundant_with_history: the question repeats an already asked attribute or intent.
- uncertain_visual_evidence: the visual evidence is ambiguous or unreliable.
- likely_useful_for_retrieval: the complete question-answer pair is likely to help distinguish the target in retrieval.
- main_attribute: one short category such as upper_clothing, lower_clothing, shoes, hair, accessory, bag_or_carried_item, body_or_posture, background, or other.
- reason: one concise sentence grounded in the four images and answer.
"""


def parse_json(text):
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        value = json.loads(candidate)
    except Exception:
        return {}, candidate
    if not isinstance(value, dict):
        return {}, candidate
    return {field: value.get(field) for field in FIELDS}, candidate


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and row.get("audit_json"):
                done.add(row["id"])
    return done


def make_messages(record):
    content = [{"type": "image", "image": path} for path in record["candidates"]]
    content.append({"type": "text", "text": build_prompt(record)})
    return [{"role": "user", "content": content}]


def infer(model, processor, record, device, max_new_tokens):
    messages = make_messages(record)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    audit_json, audit_raw = parse_json(raw)
    return audit_json, audit_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.end:
        cases = cases[args.start : args.end]
    elif args.start:
        cases = cases[args.start :]
    if args.limit:
        cases = cases[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)

    print(f"cases={len(cases)} done={len(done)} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    completed = 0
    with out.open("a", encoding="utf-8") as handle:
        for record in cases:
            if record["id"] in done:
                continue
            result = dict(record)
            try:
                audit_json, audit_raw = infer(model, processor, record, device, args.max_new_tokens)
                result["audit_json"] = audit_json
                result["audit_raw"] = audit_raw
            except Exception as exc:
                result["audit_json"] = {}
                result["audit_raw"] = ""
                result["error"] = repr(exc)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if completed % args.print_every == 0 or completed == 1:
                print(f"completed={completed} total={len(cases)} id={record['id']}", flush=True)


if __name__ == "__main__":
    main()

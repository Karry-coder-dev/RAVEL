#!/usr/bin/env python3
import argparse
import json
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from PIL import Image

from prompt import prompt_question_generator_v3, wrap_question_prompt
from model.llava.conversation import conv_templates
from model.llava.mm_utils import process_images, tokenizer_image_token
from model.llava_reid_utils import load_question_model

IGNORE_INDEX = -100




def prepare_images(model, image_processor, paths, mode):
    images = [Image.open(path).convert("RGB") for path in paths]
    tensors = process_images(images, image_processor, model.config)
    tensors = [x.to(dtype=torch.bfloat16) for x in tensors]
    if mode == "rank1_only":
        tensors = [tensors[0]] + [torch.zeros_like(x) for x in tensors[1:]]
    elif mode == "no_image":
        tensors = [torch.zeros_like(x) for x in tensors]
    return tensors, [image.size for image in images]


def sequence_logprob(model, input_ids, labels, image_tensors, image_sizes):
    input_ids = input_ids.unsqueeze(0).to(model.device)
    labels = labels.unsqueeze(0).to(model.device)
    attention_mask = input_ids.ne(model.config.pad_token_id or 0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensors,
            image_sizes=image_sizes,
            modalities=["image"] * len(image_tensors),
            use_cache=False,
        )
    logits = outputs.logits
    if logits.shape[1] != labels.shape[1]:
        common = min(logits.shape[1], labels.shape[1])
        logits, labels = logits[:, -common:], labels[:, -common:]
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logps = F.log_softmax(shift_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (logps * mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1)).item()


def _visual_spans(input_ids, expanded_length, feature_lengths=None):
    image_positions = torch.where(input_ids[0].eq(-200))[0].tolist()
    if not image_positions:
        return []
    text_tokens = input_ids.shape[1] - len(image_positions)
    total_visual = expanded_length - text_tokens
    if total_visual <= 0:
        raise RuntimeError(
            f"Cannot align visual spans: input={input_ids.shape[1]}, "
            f"expanded={expanded_length}, images={len(image_positions)}"
        )
    if feature_lengths is None:
        base_len = total_visual // len(image_positions)
        remainder = total_visual % len(image_positions)
        feature_lengths = [base_len + (i < remainder) for i in range(len(image_positions))]
    spans = []
    cursor = 0
    previous = -1
    for position, visual_len in zip(image_positions, feature_lengths):
        cursor += position - previous - 1
        spans.append(list(range(cursor, cursor + visual_len)))
        cursor += visual_len
        previous = position
    return spans


def visual_reliance(model, input_ids, labels, image_tensors, image_sizes):
    input_ids = input_ids.unsqueeze(0).to(model.device)
    labels = labels.unsqueeze(0).to(model.device)
    attention_mask = input_ids.ne(model.config.pad_token_id or 0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prepared = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, labels,
            image_tensors, ["image"] * len(image_tensors), image_sizes
        )
        _, position_ids, expanded_mask, _, inputs_embeds, expanded_labels = prepared
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_mask,
            position_ids=position_ids,
            labels=expanded_labels,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions
    if not attentions or any(att is None for att in attentions):
        raise RuntimeError("The model returned no attention weights; eager attention is required.")
    feature_lengths = getattr(model, "_last_image_feature_lengths", None)
    spans = _visual_spans(input_ids, inputs_embeds.shape[1], feature_lengths)
    visual_mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.bool, device=model.device)
    candidate_masks = []
    for span in spans:
        mask = torch.zeros_like(visual_mask)
        mask[span] = True
        candidate_masks.append(mask)
        visual_mask[span] = True

    target_positions = torch.where(expanded_labels[0].ne(IGNORE_INDEX))[0]
    target_positions = target_positions[target_positions > 0] - 1
    if target_positions.numel() == 0:
        return {"vrr": 0.0, "candidate_vrr": [0.0] * len(spans)}

    layer_values = []
    for att in attentions[-4:]:
        att = att[0].float()
        rows = att[:, target_positions, :].mean(dim=0)
        layer_values.append(rows)
    rows = torch.stack(layer_values).mean(dim=0)
    denom = rows[:, expanded_mask[0].bool()].sum(dim=-1).clamp_min(1e-8)
    vrr = (rows[:, visual_mask].sum(dim=-1) / denom).mean().item()
    candidate_vrr = [
        (rows[:, mask].sum(dim=-1) / denom).mean().item()
        for mask in candidate_masks
    ]
    return {"vrr": vrr, "candidate_vrr": candidate_vrr}


def build_round_text_inputs(tokenizer, initial_query, questions, answers, target_question):
    user_prompt = wrap_question_prompt(
        prompt_question_generator_v3, initial_query, questions, answers, 4, batch=False
    )
    prompt_conv = deepcopy(conv_templates["qwen_reid"])
    prompt_conv.append_message(prompt_conv.roles[0], user_prompt)
    prompt_conv.append_message(prompt_conv.roles[1], None)
    prompt_ids = tokenizer_image_token(prompt_conv.get_prompt(), tokenizer, return_tensors="pt")

    full_conv = deepcopy(conv_templates["qwen_reid"])
    full_conv.append_message(full_conv.roles[0], user_prompt)
    full_conv.append_message(full_conv.roles[1], target_question.strip())
    full_ids = tokenizer_image_token(full_conv.get_prompt(), tokenizer, return_tensors="pt")
    labels = full_ids.clone()
    labels[: prompt_ids.shape[0]] = IGNORE_INDEX
    return full_ids, labels


def candidate_sensitivity(model, input_ids, labels, image_tensors, image_sizes):
    """Compute candidate sensitivity with leave-one-out image masking."""
    full_logprob = sequence_logprob(model, input_ids, labels, image_tensors, image_sizes)
    leave_one_out = []
    for candidate_index in range(len(image_sizes)):
        if isinstance(image_tensors, torch.Tensor):
            masked = image_tensors.clone()
            masked[candidate_index] = torch.zeros_like(masked[candidate_index])
        else:
            masked = [image.clone() for image in image_tensors]
            masked[candidate_index] = torch.zeros_like(masked[candidate_index])
        leave_one_out.append(
            sequence_logprob(model, input_ids, labels, masked, image_sizes)
        )
    per_candidate = [full_logprob - value for value in leave_one_out]
    return {
        "full_logprob": full_logprob,
        "leave_one_out_logprob": leave_one_out,
        "candidate_sensitivity": per_candidate,
        "cs": sum(per_candidate) / len(per_candidate),
    }


def process_row(model, tokenizer, image_processor, row, row_index):
    interactions = row["interaction"][:5]
    results = []
    for round_id, interaction in enumerate(interactions):
        history = interactions[:round_id]
        history_questions = [x["question"] for x in history]
        history_answers = [x["answer"] for x in history]
        paths = interaction["candidate"][:4]
        full_ids, labels = build_round_text_inputs(
            tokenizer, row["initial_query"], history_questions, history_answers,
            interaction["question"]
        )

        shuffled = list(paths)
        random.Random(1000003 + row_index * 5 + round_id).shuffle(shuffled)
        mode_paths = {"normal": paths, "shuffle": shuffled, "rank1_only": paths, "no_image": paths}
        mode_values = {}
        for mode in ["normal", "shuffle", "rank1_only", "no_image"]:
            tensors, sizes = prepare_images(model, image_processor, mode_paths[mode], mode)
            mode_values[mode] = {
                "logprob": sequence_logprob(model, full_ids, labels, tensors, sizes),
                **visual_reliance(model, full_ids, labels, tensors, sizes),
            }

        full_tensors, sizes = prepare_images(model, image_processor, paths, "normal")
        blank_tensors, _ = prepare_images(model, image_processor, paths, "no_image")
        blank_lp = sequence_logprob(model, full_ids, labels, blank_tensors, sizes)
        cs_values = candidate_sensitivity(model, full_ids, labels, full_tensors, sizes)

        results.append({
            "index": row_index,
            "round": round_id,
            "initial_query": row["initial_query"],
            "history_questions": history_questions,
            "history_answers": history_answers,
            "question": interaction["question"],
            "candidate": paths,
            "modes": mode_values,
            "ias": cs_values["full_logprob"] - blank_lp,
            **cs_values,
        })
    return results


def main(args):
    random.seed(42 + args.start)
    torch.manual_seed(42 + args.start)
    checkpoint = torch.load(args.conv_log, map_location="cpu")
    conversations = checkpoint["conversation_log"]
    end = min(args.end, len(conversations))
    conversations = conversations[args.start:end]

    logger = type("Logger", (), {"info": print})()
    model, tokenizer = load_question_model(args.model, logger=logger, attn_implementation="eager")
    model.eval()
    image_processor = model.get_vision_tower().image_processor

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for local_index, row in enumerate(conversations):
            absolute_index = args.start + local_index
            for result in process_row(model, tokenizer, image_processor, row, absolute_index):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (local_index + 1) % 10 == 0:
                gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                print(f"gpu={gpu} processed={absolute_index + 1}/{end}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conv-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    main(parser.parse_args())

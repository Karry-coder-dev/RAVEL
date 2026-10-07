import argparse
import json
import math
import os
import random
import re
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn.functional as F
import transformers
import yaml
from PIL import Image
from easydict import EasyDict
from tqdm import tqdm

from prompt import prompt_question_generator_v3, wrap_question_prompt
from model.llava.conversation import conv_templates
from model.llava.mm_utils import process_images, tokenizer_image_token
from model.llava.train.train_utils import pad_sequence
from model.llava_qwen_reid import LlavaQwenForPersonReID
import model.llava_reid_utils as llava_reid_utils
from model.llava_reid_utils import AnswerGeneratorSGLang, delete_prefix
from rl_retriever_utils import build_retriever, encode_texts
from train_llava_reid import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    parse_yaml_file_with_env,
)
from utils.simple_tokenizer import SimpleTokenizer


IGNORE_INDEX = -100
llava_reid_utils.LlavaQwenForPersonReID = LlavaQwenForPersonReID

# Fixed experiment protocol for the current RL line.
# Questioner always sees rank top-4 candidates, answerer uses max_tokens=64
# from config, and retrieval text uses the original LLaVA-ReID prefix cleaner.
FIXED_NUM_CANDIDATES = 4


class PrintLogger:
    def info(self, message):
        print(message)


def load_eval_config(path):
    cfg = yaml.load(open(path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    cfg.update(cfg["stage_config"]["eval"])
    cfg["stage"] = "train_questioner"
    del cfg["stage_config"]
    return EasyDict(cfg)


def load_training_args(llava_config, model_path, output_dir, lr, lora_enable=None):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parse_yaml_file_with_env(parser, llava_config)
    model_args.model_name_or_path = model_path
    model_args.mm_tunable_parts = "mm_language_model"
    training_args.output_dir = output_dir
    training_args.learning_rate = lr
    training_args.deepspeed = None
    training_args.report_to = []
    training_args.torch_compile = False
    training_args.dataloader_drop_last = False
    training_args.group_by_modality_length = False
    training_args.local_rank = -1
    if lora_enable is not None:
        training_args.lora_enable = lora_enable
    return model_args, data_args, training_args


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_states(path, limit=None, seed=42):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Refusing non-train state split={row.get('split')!r}")
            rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def append_answer(collection, answer):
    cleaned = delete_prefix(answer).strip()
    return (collection + " " + cleaned).strip()


def rank_from_feat(qfeat, gfeats, pid, gids):
    sim = qfeat @ gfeats.t()
    order = torch.argsort(sim, descending=True).cpu().numpy()
    matches = gids[order] == pid
    pos = matches.nonzero()[0]
    return int(pos[0]) + 1 if len(pos) else len(gids) + 1


def rank1_shaped_reward(rank_before, rank_after, rank1_entry_bonus=0.35):
    reciprocal_delta = 1.0 / max(1, rank_after) - 1.0 / max(1, rank_before)
    rank1_entry_term = rank1_entry_bonus if rank_before > 1 and rank_after == 1 else 0.0
    return reciprocal_delta + rank1_entry_term


def normalize_question(question):
    return " ".join(str(question).strip().split())


def invalid_question_reason(question):
    q = normalize_question(question)
    q_lower = q.lower()
    if not q:
        return "empty"
    if len(q.split()) < 5:
        return "too_short"
    if "?" not in q:
        return "no_question_mark"
    if q_lower.startswith("candidate person") or q_lower.startswith("[candidate person]"):
        return "candidate_prefix"

    candidate_markers = [
        "similar persons",
        "which candidate",
        "which image",
        "candidate image",
        "potential match",
        "potential matches",
        "top-ranked",
        "top ranked",
        "narrow down",
        "first image",
        "second image",
        "third image",
        "fourth image",
    ]
    if any(marker in q_lower for marker in candidate_markers) or re.search(r"\bimage\s*[1-4]\b", q_lower):
        return "candidate_selection"
    if "conversation:" in q_lower or "answer:" in q_lower:
        return "self_filled_dialogue"

    unconditional_memory_dump_markers = [
        "everything you remember",
        "all details",
        "full description",
        "complete description",
        "describe the person completely",
        "describe the person as completely as possible",
        "as much detail as possible",
        "tell me everything",
    ]
    if any(marker in q_lower for marker in unconditional_memory_dump_markers):
        return "memory_dump"

    local_targets = [
        "backpack", "bag", "handbag", "purse", "umbrella", "suitcase", "luggage",
        "shoes", "boots", "sneakers", "pants", "trousers", "jeans", "skirt", "shorts",
        "coat", "jacket", "shirt", "top", "hoodie", "sweater", "dress",
        "hat", "cap", "hair", "glasses", "mask", "watch", "jewelry", "jewellery",
        "necklace", "accessory", "accessories", "carrying", "holding",
        "upper body", "lower body", "sleeve", "collar", "hood", "pattern", "logo",
        "color", "colour", "environment", "setting", "background",
    ]
    broad_request_markers = [
        "any additional details",
        "any other details",
        "more details",
        "more specific details",
        "additional information",
        "anything else",
        "distinguishing features",
        "distinctive features",
    ]
    broad_target_markers = [
        "person", "man", "woman", "individual", "target", "appearance",
        "surroundings", "location", "where", "overall", "general appearance",
        "appearance or surroundings", "appearance or location", "appearance or clothing",
        "clothing and surroundings",
    ]
    has_broad_request = any(marker in q_lower for marker in broad_request_markers)
    has_broad_target = any(marker in q_lower for marker in broad_target_markers)
    has_local_target = any(marker in q_lower for marker in local_targets)
    if has_broad_request and not has_local_target:
        return "broad_memory_dump"
    if re.search(r"(distinguishing|distinctive) features? (or|and) (characteristics|traits)", q_lower):
        return "broad_memory_dump"
    return None


def repeat_penalty(question, history, group_prefix=None):
    q_tokens = {t.strip(".,?!:;").lower() for t in question.split() if len(t) > 2}
    if not q_tokens:
        return False
    histories = list(history or [])
    if group_prefix:
        histories.extend(group_prefix)
    for old in histories:
        old_tokens = {t.strip(".,?!:;").lower() for t in old.split() if len(t) > 2}
        if not old_tokens:
            continue
        jaccard = len(q_tokens & old_tokens) / max(1, len(q_tokens | old_tokens))
        if jaccard >= 0.70:
            return True
    return False


def build_user_prompt(row, num_candidates, prompt_template):
    return wrap_question_prompt(
        prompt_template,
        row["initial_query"],
        row.get("questions", []),
        row.get("answers", []),
        num_candidates,
        batch=False,
    )


def generate_questions(
    model,
    tokenizer,
    image_processor,
    conv_template,
    state,
    group_size,
    num_candidates,
    prompt_template,
    temperature,
    top_p,
    max_new_tokens,
):
    image_paths = state["candidate_images"][:num_candidates]
    image_all = [Image.open(img).convert("RGB") for _ in range(group_size) for img in image_paths]
    image_tensor = process_images(image_all, image_processor, model.config)
    image_tensor = [image.to(dtype=torch.bfloat16) for image in image_tensor]
    image_sizes = [image.size for image in image_all]

    user_prompt = build_user_prompt(state, num_candidates, prompt_template)
    input_ids_list = []
    for _ in range(group_size):
        conv = deepcopy(conv_template)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        input_ids_list.append(tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt"))

    input_ids = pad_sequence(tokenizer, input_ids_list, True, tokenizer.pad_token_id).to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        gen_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(image_tensor),
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            begin_suppress_tokens=[tokenizer.eos_token_id],
            max_new_tokens=max_new_tokens,
        )
    questions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    cleaned = []
    for question in questions:
        if question.startswith("assistant\n"):
            question = question.replace("assistant\n", "", 1)
        cleaned.append(question.strip())
    model.train()
    return cleaned


def encode_one_example(row, tokenizer, conv_template, num_candidates, prompt_template):
    user_prompt = build_user_prompt(row, num_candidates, prompt_template)

    prompt_conv = deepcopy(conv_template)
    prompt_conv.append_message(prompt_conv.roles[0], user_prompt)
    prompt_conv.append_message(prompt_conv.roles[1], None)
    prompt_ids = tokenizer_image_token(prompt_conv.get_prompt(), tokenizer, return_tensors="pt")

    full_conv = deepcopy(conv_template)
    full_conv.append_message(full_conv.roles[0], user_prompt)
    full_conv.append_message(full_conv.roles[1], row["question"].strip())
    full_ids = tokenizer_image_token(full_conv.get_prompt(), tokenizer, return_tensors="pt")

    labels = full_ids.clone()
    labels[: prompt_ids.shape[0]] = IGNORE_INDEX
    return full_ids, labels


def build_train_batch(row, model, tokenizer, image_processor, conv_template, num_candidates, prompt_template):
    input_ids, labels = encode_one_example(row, tokenizer, conv_template, num_candidates, prompt_template)
    images = row["candidate_images"][:num_candidates]
    image_all = [Image.open(path).convert("RGB") for path in images]
    image_tensor = process_images(image_all, image_processor, model.config)
    image_tensor = [image.to(dtype=torch.bfloat16) for image in image_tensor]

    input_ids = pad_sequence(tokenizer, [input_ids], True, tokenizer.pad_token_id).to(model.device)
    labels = pad_sequence(tokenizer, [labels], True, IGNORE_INDEX).to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": image_tensor,
        "image_sizes": [image.size for image in image_all],
        "modalities": ["image"] * len(image_tensor),
        "selector_grad_c": torch.ones(len(image_tensor), dtype=torch.bfloat16, device=model.device),
    }


def sequence_nll(logits, labels):
    token_logps, mask = token_logps_from_logits(logits, labels)
    denom = mask.sum(dim=1).clamp_min(1)
    return -(token_logps * mask).sum(dim=1) / denom


def token_logps_from_logits(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    return token_logps, mask


def output_nll(outputs, labels):
    if outputs.logits.shape[1] != labels.shape[1]:
        return outputs.loss.reshape(1)
    return sequence_nll(outputs.logits, labels)


def output_token_logps(outputs, labels):
    logits = outputs.logits
    if logits.shape[1] != labels.shape[1]:
        common_len = min(logits.shape[1], labels.shape[1])
        # Multimodal forwards may drop/expand placeholder image tokens. The supervised
        # question tokens are at the right edge, so right-align before gathering logprobs.
        logits = logits[:, -common_len:, :]
        labels = labels[:, -common_len:]
    return token_logps_from_logits(logits, labels)


def adapter_disabled(model):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()


def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/main/config/baseline.yaml")
    parser.add_argument("--llava-config", default="./code/main/config/train_question.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--ref-model-path", default=None)
    parser.add_argument("--states", required=True)
    parser.add_argument("--gallery-preprocessed", default="dataset/interactive-pedes/preprocessed_data.pt")
    parser.add_argument("--ann", default="dataset/interactive-pedes/Interactive-PEDES_interactive_annos.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--samples-out", default=None)
    parser.add_argument("--limit-states", type=int, default=3000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--grad-accum-states", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--repeat-coef", type=float, default=0.02)
    parser.add_argument("--invalid-reward", type=float, default=None)
    parser.add_argument("--rank1-entry-bonus", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--answerer-base-url", default=None)
    parser.add_argument("--answerer-api-key", default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_eval_config(args.config)
    model_path = args.model_path or cfg.question_model_path
    model_args, data_args, training_args = load_training_args(
        args.llava_config,
        model_path,
        args.output_dir,
        args.lr,
    )
    print(f"Loading policy model from {model_path}; lora_enable={training_args.lora_enable}")
    model, tokenizer = llava_reid_utils.load_llava_reid_model_for_training(model_args, data_args, training_args)
    model.train()
    model.config.use_cache = False
    model.get_vision_tower().to(dtype=torch.bfloat16, device=model.device)
    image_processor = model.get_vision_tower().image_processor
    conv_template = conv_templates["qwen_reid"]
    prompt_template = prompt_question_generator_v3[args.prompt_index]

    ref_model = None
    if args.ref_model_path:
        ref_output_dir = os.path.join(args.output_dir, "_ref_model_unused")
        ref_model_args, ref_data_args, ref_training_args = load_training_args(
            args.llava_config,
            args.ref_model_path,
            ref_output_dir,
            args.lr,
            lora_enable=False,
        )
        ref_model_args.mm_tunable_parts = ""
        print(f"Loading KL reference model from {args.ref_model_path}; lora_enable={ref_training_args.lora_enable}")
        ref_model, _ = llava_reid_utils.load_llava_reid_model_for_training(
            ref_model_args, ref_data_args, ref_training_args
        )
        ref_model.eval()
        ref_model.config.use_cache = False
        ref_model.requires_grad_(False)
        ref_model.get_vision_tower().to(dtype=torch.bfloat16, device=ref_model.device)
        ref_lora_params = sum(1 for name, _ in ref_model.named_parameters() if "lora_" in name.lower())
        print(f"KL reference LoRA parameter tensors: {ref_lora_params}")

    states = load_states(args.states, args.limit_states, args.seed)
    if not states:
        raise ValueError("No train states loaded.")

    pt = torch.load(args.gallery_preprocessed, map_location="cpu")
    gfeats = F.normalize(pt["image_feats"].float(), dim=-1)
    ann = json.load(open(args.ann, "r", encoding="utf-8"))
    train_ann = [x for x in ann if x.get("split") == "train"]
    if len(train_ann) != gfeats.shape[0]:
        raise ValueError(f"train annotations {len(train_ann)} != gallery features {gfeats.shape[0]}")
    gids = torch.tensor([int(x["id"]) for x in train_ann], dtype=torch.long).cpu().numpy()

    retriever_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    retriever = build_retriever(cfg, retriever_device)
    simple_tokenizer = SimpleTokenizer()
    answerer = AnswerGeneratorSGLang(
        args.answerer_base_url or getattr(cfg, "answerer_base_url", "http://localhost:10500/v1"),
        args.answerer_api_key or getattr(cfg, "answerer_api_key", "Qwen-7B"),
        args.answer_max_tokens or getattr(cfg, "answer_max_tokens", getattr(cfg, "max_answer_length", 64)),
        direct=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    total_groups = int(math.ceil(len(states) * args.epochs))
    total_updates = max(1, math.ceil(total_groups / max(1, args.grad_accum_states)))
    print(f"Loaded {len(states)} online train states")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.2f}M")
    print(f"Approx optimizer updates: {total_updates}")
    print(f"Online GRPO args: beta={args.beta} clip_eps={args.clip_eps} temp={args.temperature} top_p={args.top_p} group={args.group_size}")
    print(f"Fixed protocol: top-{FIXED_NUM_CANDIDATES}, answer_max_tokens={answerer.max_tokens}")
    print(f"KL reference: {args.ref_model_path if args.ref_model_path else 'adapter_disabled(policy_model)'}")
    invalid_reward = -1.0 if args.invalid_reward is None else args.invalid_reward
    print(f"Validity gate: invalid_reward={invalid_reward} repeat_coef={args.repeat_coef}")

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    metrics_f = open(args.metrics_out, "w", encoding="utf-8")
    samples_f = open(args.samples_out, "w", encoding="utf-8") if args.samples_out else None

    optimizer_step = 0
    running = []
    optimizer.zero_grad(set_to_none=True)
    for group_step in tqdm(range(total_groups), desc="Online GRPO"):
        state = states[group_step % len(states)]
        questions = generate_questions(
            model,
            tokenizer,
            image_processor,
            conv_template,
            state,
            args.group_size,
            FIXED_NUM_CANDIDATES,
            prompt_template,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
        )
        invalid_reasons = []
        history_questions = state.get("questions", [])
        for question in questions:
            reason = invalid_question_reason(question)
            if reason is None and repeat_penalty(question, history_questions):
                reason = "history_repeat"
            invalid_reasons.append(reason)
        valid_indices = [i for i, reason in enumerate(invalid_reasons) if reason is None]
        answers = [""] * len(questions)
        rank_after_by_index = [state["rank_before"]] * len(questions)
        rank_reward_by_index = [invalid_reward] * len(questions)
        if valid_indices:
            valid_questions = [questions[i] for i in valid_indices]
            valid_answers = answerer([state["fine_grained_description"]] * len(valid_questions), valid_questions)
            collections = [append_answer(state["collection"], ans) for ans in valid_answers]
            qfeats = encode_texts(
                retriever,
                simple_tokenizer,
                collections,
                cfg.max_retrieve_length,
                args.text_batch_size,
                retriever_device,
            )
            qfeats = F.normalize(qfeats.float(), dim=-1)
            for local_idx, sample_idx in enumerate(valid_indices):
                answer = valid_answers[local_idx]
                rank_after = rank_from_feat(qfeats[local_idx], gfeats, state["pid"], gids)
                rank_reward = rank1_shaped_reward(
                    state["rank_before"],
                    rank_after,
                    rank1_entry_bonus=args.rank1_entry_bonus,
                )
                answers[sample_idx] = answer
                rank_after_by_index[sample_idx] = rank_after
                rank_reward_by_index[sample_idx] = rank_reward

        sample_rows = []
        rewards = []
        rank_rewards = []
        previous_group_questions = []
        for sample_idx, (question, answer) in enumerate(zip(questions, answers)):
            invalid_reason = invalid_reasons[sample_idx]
            rank_after = rank_after_by_index[sample_idx]
            rank_reward = rank_reward_by_index[sample_idx]
            # Earlier-round repetition is invalid; only group-internal repetition
            # receives the separate small penalty.
            rep = 1.0 if repeat_penalty(question, None, previous_group_questions) else 0.0
            reward = rank_reward - args.repeat_coef * rep
            if invalid_reason is not None:
                reward = invalid_reward
            row = {
                **state,
                "sample_id_in_group": sample_idx,
                "question": question,
                "answer": answer,
                "rank_after": rank_after,
                "rank_reward": rank_reward,
                "invalid_question": invalid_reason is not None,
                "invalid_reason": invalid_reason,
                "repeat": bool(rep),
                "reward": reward,
                "prompt_index": args.prompt_index,
                "prompt_template": prompt_template,
            }
            previous_group_questions.append(question)
            sample_rows.append(row)
            rewards.append(reward)
            rank_rewards.append(rank_reward)

        reward_mean = sum(rewards) / len(rewards)
        reward_var = sum((r - reward_mean) ** 2 for r in rewards) / max(1, len(rewards))
        reward_std = max(reward_var ** 0.5, 1e-6)
        advantages = [(reward - reward_mean) / reward_std for reward in rewards]

        group_loss = torch.zeros([], device=model.device)
        nll_values, ref_nll_values, old_nll_values, kl_values, ratio_values = [], [], [], [], []
        valid_loss_count = 0
        skipped_loss_count = 0
        for row, advantage in zip(sample_rows, advantages):
            if row.get("invalid_question") and row.get("invalid_reason") == "empty":
                skipped_loss_count += 1
                continue
            batch = build_train_batch(
                row,
                model,
                tokenizer,
                image_processor,
                conv_template,
                FIXED_NUM_CANDIDATES,
                prompt_template,
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                old_outputs = model(**batch)
                old_logps, token_mask = output_token_logps(old_outputs, batch["labels"])
                old_logps = old_logps.detach()
            if ref_model is not None:
                ref_batch = move_batch_to_device(batch, ref_model.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    ref_outputs = ref_model(**ref_batch)
                    ref_logps, _ = output_token_logps(ref_outputs, ref_batch["labels"])
                    ref_logps = ref_logps.detach().to(old_logps.device)
            else:
                with torch.no_grad(), adapter_disabled(model):
                    ref_outputs = model(**batch)
                    ref_logps, _ = output_token_logps(ref_outputs, batch["labels"])
                    ref_logps = ref_logps.detach()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = model(**batch)
                logps, token_mask = output_token_logps(outputs, batch["labels"])
                mask = token_mask.to(logps.dtype)
                denom = mask.sum(dim=1).clamp_min(1.0)
                nll = -(logps * mask).sum(dim=1) / denom
                ref_nll = -(ref_logps * mask).sum(dim=1) / denom
                old_nll = -(old_logps * mask).sum(dim=1) / denom

                log_ratio = (logps - old_logps) * mask
                ratio = torch.exp(log_ratio).clamp(0.0, 10.0)
                clipped_ratio = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
                adv = torch.as_tensor(advantage, dtype=logps.dtype, device=logps.device).detach()
                unclipped = ratio * adv
                clipped = clipped_ratio * adv
                policy_loss = -torch.minimum(unclipped, clipped) * mask

                ref_log_ratio = (ref_logps - logps) * mask
                kl_k3 = (torch.exp(ref_log_ratio).clamp(0.0, 10.0) - ref_log_ratio - 1.0) * mask
                loss = (policy_loss + args.beta * kl_k3).sum(dim=1) / denom
            if not torch.isfinite(loss).all():
                skipped_loss_count += 1
                continue
            group_loss = group_loss + loss.mean()
            valid_loss_count += 1
            nll_values.append(float(nll.detach().cpu().mean()))
            ref_nll_values.append(float(ref_nll.detach().cpu().mean()))
            old_nll_values.append(float(old_nll.detach().cpu().mean()))
            kl_values.append(float((kl_k3.sum(dim=1) / denom).detach().cpu().mean()))
            ratio_values.append(float(((ratio * mask).sum(dim=1) / denom).detach().cpu().mean()))

        if valid_loss_count > 0:
            (group_loss / valid_loss_count / args.grad_accum_states).backward()

        improved = sum((not row.get("invalid_question")) and row["rank_after"] < state["rank_before"] for row in sample_rows)
        same = sum((not row.get("invalid_question")) and row["rank_after"] == state["rank_before"] for row in sample_rows)
        worse = sum((not row.get("invalid_question")) and row["rank_after"] > state["rank_before"] for row in sample_rows)
        invalid = sum(row.get("invalid_question", False) for row in sample_rows)
        metric = {
            "group_step": group_step + 1,
            "optimizer_step": optimizer_step,
            "state_id": state["state_id"],
            "round": state["round"],
            "source": state.get("source"),
            "rank_before": state["rank_before"],
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "rank_reward_mean": sum(rank_rewards) / len(rank_rewards),
            "rank_reward_max": max(rank_rewards),
            "rank_reward_min": min(rank_rewards),
            "advantage_mean": sum(advantages) / len(advantages),
            "kl_k3_mean": sum(kl_values) / len(kl_values) if kl_values else 0.0,
            "ratio_mean": sum(ratio_values) / len(ratio_values) if ratio_values else 0.0,
            "nll_mean": sum(nll_values) / len(nll_values) if nll_values else 0.0,
            "old_nll_mean": sum(old_nll_values) / len(old_nll_values) if old_nll_values else 0.0,
            "ref_nll_mean": sum(ref_nll_values) / len(ref_nll_values) if ref_nll_values else 0.0,
            "valid_loss_count": valid_loss_count,
            "skipped_loss_count": skipped_loss_count,
            "improved_ratio": improved / len(sample_rows),
            "same_ratio": same / len(sample_rows),
            "worse_ratio": worse / len(sample_rows),
            "invalid_ratio": invalid / len(sample_rows),
            "valid_ratio": 1.0 - invalid / len(sample_rows),
            "unique_questions": len(set(questions)),
            "repeat_ratio": sum(row["repeat"] for row in sample_rows) / len(sample_rows),
            "loss": float(group_loss.detach().cpu()),
        }
        metrics_f.write(json.dumps(metric, ensure_ascii=False) + "\n")
        metrics_f.flush()
        running.append(metric)

        if samples_f:
            for row, advantage in zip(sample_rows, advantages):
                row["advantage"] = advantage
                row["group_reward_mean"] = reward_mean
                row["group_reward_std"] = reward_std
                samples_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            samples_f.flush()

        if (group_step + 1) % args.grad_accum_states == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step % args.log_steps == 0:
                recent = running[-args.log_steps * args.grad_accum_states :]
                print(
                    "update={} reward={:.6f} rank_reward={:.6f} kl={:.6f} "
                    "improve={:.3f} worse={:.3f} invalid={:.3f} repeat={:.3f} uniq_q={:.2f}".format(
                        optimizer_step,
                        sum(x["reward_mean"] for x in recent) / len(recent),
                        sum(x["rank_reward_mean"] for x in recent) / len(recent),
                        sum(x["kl_k3_mean"] for x in recent) / len(recent),
                        sum(x["improved_ratio"] for x in recent) / len(recent),
                        sum(x["worse_ratio"] for x in recent) / len(recent),
                        sum(x["invalid_ratio"] for x in recent) / len(recent),
                        sum(x["repeat_ratio"] for x in recent) / len(recent),
                        sum(x["unique_questions"] for x in recent) / len(recent),
                    )
                )

    if total_groups % args.grad_accum_states != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    metrics_f.close()
    if samples_f:
        samples_f.close()
    save_model(model, tokenizer, args.output_dir)
    print(f"Saved online GRPO LoRA/model to {args.output_dir}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()

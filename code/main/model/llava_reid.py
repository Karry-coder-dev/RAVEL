import datetime
import base64
import io
import json
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re
import torch
import torch.nn.functional as F
from PIL import Image
from easydict import EasyDict
from torch import nn
from tqdm import tqdm
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError, URLError

from prompt import prompt_question_generator_v3
from prompt import wrap_question_prompt
from reid_datasets.bases import tokenize_simple
from utils.iotools import LoggerX
from utils.metrics import per_sample_ranks, max_discriminative
from model.IRRA import IRRA
from utils import misc
from .clip_model import convert_weights
from utils.simple_tokenizer import SimpleTokenizer

from .llava.conversation import conv_templates
from .llava.mm_utils import tokenizer_image_token, process_images
from .llava.train.train_utils import pad_sequence
from .llava.utils import rank0_print
from .llava_reid_utils import delete_prefix, load_llava_reid_model, \
    AnswerGeneratorSGLang, process_image_train
from .retriever_adapters import load_rde_retriever
from .selector import Selector, SelectorConfig

warnings.filterwarnings("ignore", category=UserWarning)


class ChatIRQuestionerHTTP(nn.Module):
    """Questioner adapter for a local OpenAI-compatible ChatIR service."""

    def __init__(self, base_url, timeout=180):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, payload):
        request = Request(
            self.base_url + "/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # The evaluation host configures an HTTP proxy.  The questioner endpoint
        # is an SSH reverse tunnel on loopback, so it must bypass that proxy.
        local_opener = build_opener(ProxyHandler({}))
        for attempt in range(5):
            try:
                with local_opener.open(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                if attempt == 4:
                    raise RuntimeError(
                        f"ChatIR questioner request failed after retries at {self.base_url}: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)

    def forward(self, initial_query, candidate_images, questions, answers,
                rank1_feedback=None):
        del candidate_images, rank1_feedback
        generated_all = []
        # Keep the API payload moderate while halving sequential HTTP round trips.
        chunk_size = 1000
        for start in range(0, len(initial_query), chunk_size):
            end = start + chunk_size
            payload = {
                "initial_query": initial_query[start:end],
                "questions": questions[start:end],
                "answers": answers[start:end],
            }
            result = self._request(payload)
            generated = result.get("questions")
            expected = len(initial_query[start:end])
            if not isinstance(generated, list) or len(generated) != expected:
                actual = len(generated) if isinstance(generated, list) else type(generated).__name__
                raise RuntimeError(
                    "ChatIR questioner returned an invalid response: "
                    f"expected {expected} questions, got_count={actual}"
                )
            generated_all.extend(generated)
        return generated_all


class OpenAIVisualQuestionerHTTP(ChatIRQuestionerHTTP):
    """Forward the current Top-4 images to a local OpenAI vision service."""

    def __init__(self, base_url, timeout=7200, image_max_side=768, jpeg_quality=90):
        super().__init__(base_url, timeout=timeout)
        self.image_max_side = image_max_side
        self.jpeg_quality = jpeg_quality

    def _encode_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((self.image_max_side, self.image_max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def forward(self, initial_query, candidate_images, questions, answers,
                rank1_feedback=None):
        del rank1_feedback
        if not (len(initial_query) == len(candidate_images) == len(questions) == len(answers)):
            raise ValueError("OpenAI vision questioner batch dimensions must match")
        generated_all = []
        # Image payloads are much larger than text-only ChatIR requests. Small
        # chunks bound reverse-tunnel memory while retaining API concurrency.
        chunk_size = 16
        for start in range(0, len(initial_query), chunk_size):
            end = min(start + chunk_size, len(initial_query))
            image_batch = []
            for paths in candidate_images[start:end]:
                if len(paths) != 4:
                    raise ValueError(f"OpenAI vision questioner expects Top-4 images, got {len(paths)}")
                image_batch.append([self._encode_image(path) for path in paths])
            result = self._request({
                "initial_query": initial_query[start:end],
                "questions": questions[start:end],
                "answers": answers[start:end],
                "candidate_images_b64": image_batch,
            })
            generated = result.get("questions")
            expected = end - start
            if not isinstance(generated, list) or len(generated) != expected:
                actual = len(generated) if isinstance(generated, list) else type(generated).__name__
                raise RuntimeError(
                    "OpenAI vision questioner returned an invalid response: "
                    f"expected {expected} questions, got_count={actual}"
                )
            generated_all.extend(generated)
        return generated_all


class SimRVHeuristicQuestioner(nn.Module):
    """Five-turn ReID adaptation of SimRV's heuristic + Ask Object chain."""

    needs_candidate_pool = False

    def _person_word(self, text):
        lowered = text.lower()
        for word in ("woman", "man", "girl", "boy", "person", "lady", "child"):
            if word in lowered.split():
                return word
        return "person"

    def _answer_phrase(self, answer):
        phrase = (answer or "").strip().rstrip(".")
        for prefix in (
            "the person is ", "the man is ", "the woman is ",
            "the person was ", "the man was ", "the woman was ",
            "he is ", "she is ", "he was ", "she was ",
        ):
            if phrase.lower().startswith(prefix):
                return phrase[len(prefix):]
        return phrase

    def _object_phrase(self, answer, memory):
        """Extract a compact object phrase from a full witness sentence."""
        text = f"{answer or ''} {memory or ''}".lower()
        match = re.search(r"(?:carrying|holding|with)\s+(?:a|an|the)\s+([^,.]+)", text)
        if match:
            phrase = match.group(1).strip()
            phrase = re.sub(r"\b(?:in|on|near|while|and)\b.*$", "", phrase).strip()
            if phrase:
                return phrase
        for candidate in ("backpack", "backpack", "bag", "umbrella", "handbag", "suitcase"):
            if candidate in text:
                return candidate
        return "object"

    def _color_phrase(self, answer, memory):
        colors = ("black", "white", "gray", "grey", "red", "blue", "green", "yellow", "brown", "pink")
        del memory
        text = (answer or "").lower()
        for color in colors:
            if color in text:
                return color
        return ""

    def forward(self, initial_query, candidate_images, questions, answers,
                rank1_feedback=None):
        del candidate_images, rank1_feedback
        generated = []
        for text, history_answers in zip(initial_query, answers):
            person = self._person_word(text)
            round_id = len(history_answers)
            if round_id == 0:
                question = f"What is the {person} doing?"
            elif round_id == 1:
                action = self._answer_phrase(history_answers[-1]) or "walking"
                question = f"Where is the {person} {action}?"
            elif round_id == 2:
                question = f"What object is the {person} carrying or holding?"
            elif round_id == 3:
                obj = self._object_phrase(history_answers[-1], text)
                question = f"What color is the {obj}?"
            else:
                obj = self._object_phrase(history_answers[-2], text)
                color = self._color_phrase(history_answers[-1], text)
                question = f"Where is the {color} {obj}?" if color else f"Where is the {obj} located?"
            generated.append(question)
        return generated


class PlugIRQuestionerHTTP(ChatIRQuestionerHTTP):
    """HTTP adapter for the PlugIR-style candidate-caption questioner."""

    needs_candidate_pool = True

    def __init__(self, base_url, timeout=300, question_options=5):
        super().__init__(base_url, timeout=timeout)
        self.question_options = question_options

    def forward(self, initial_query, candidate_images, questions, answers,
                rank1_feedback=None, candidate_captions=None, retrieval_text=None,
                prior_questions=None, fallback=False):
        del candidate_images, rank1_feedback
        if candidate_captions is None or len(candidate_captions) != len(initial_query):
            raise ValueError("PlugIR questioner requires one caption set per query")
        if retrieval_text is None or len(retrieval_text) != len(initial_query):
            raise ValueError("PlugIR questioner requires one retrieval text per query")
        if prior_questions is None:
            prior_questions = [[] for _ in initial_query]
        if len(prior_questions) != len(initial_query):
            raise ValueError("PlugIR prior question sets must match query batch size")
        generated_all = []
        chunk_size = 128
        for start in range(0, len(initial_query), chunk_size):
            end = start + chunk_size
            result = self._request({
                "initial_query": initial_query[start:end],
                "questions": questions[start:end],
                "answers": answers[start:end],
                "candidate_captions": candidate_captions[start:end],
                "retrieval_text": retrieval_text[start:end],
                "prior_questions": prior_questions[start:end],
                "fallback": fallback,
                "question_options": self.question_options,
            })
            generated = result.get("question_options")
            expected = len(initial_query[start:end])
            if not isinstance(generated, list) or len(generated) != expected:
                actual = len(generated) if isinstance(generated, list) else type(generated).__name__
                raise RuntimeError(
                    "PlugIR questioner returned an invalid response: "
                    f"expected {expected} option sets, got_count={actual}"
                )
            for options in generated:
                if not isinstance(options, list) or not options:
                    raise RuntimeError("PlugIR questioner returned an empty option set")
            generated_all.extend(generated)
        return generated_all


class LlavaForPersonReID(nn.Module):
    def __init__(self, config, llava_config=None, logger: LoggerX = None):
        super(LlavaForPersonReID, self).__init__()
        self.retriever_device = "cuda" if misc.is_dist_avail_and_initialized() else "cuda:0"
        self.interact_round = config.interact_round
        self.num_candidates = config.num_candidates
        self.q_length = config.max_question_length
        self.a_length = config.max_answer_length
        self.r_length = config.max_retrieve_length
        self.num_potential_candidates = 200
        self.stage = config.stage

        self.total_time = 0
        self.inference_count = 0

        if llava_config is not None and "data_args" in llava_config:
            self.image_folder = llava_config["data_args"].image_folder

        match self.stage:
            case "train_retriever":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, logger=logger)
                self.forward = self._train_retriever
            case "prepare_data":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, logger=logger)
                self.forward = self._prepare_data
            case "warmup_selector":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                self.forward = self._warmup_selector
            case "train_questioner":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config)
                if config.selector_model_path is not None:
                    self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                    self.load_selector(config, logger)
                self.forward = self._train_questioner_selector
            case "train_selector":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config)
                self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                self.load_selector(config, logger)
                self.forward = self._train_questioner_selector
            case "eval":
                self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
                questioner_api_url = getattr(config, "questioner_api_url", None)
                if getattr(config, "questioner_type", "") == "simrv_heuristic":
                    self.question_model = SimRVHeuristicQuestioner()
                elif getattr(config, "questioner_type", "") == "openai_vision":
                    if not questioner_api_url:
                        raise ValueError("OpenAI vision questioner requires questioner_api_url")
                    self.question_model = OpenAIVisualQuestionerHTTP(
                        questioner_api_url,
                        timeout=getattr(config, "questioner_api_timeout", 7200),
                        image_max_side=getattr(config, "questioner_image_max_side", 768),
                        jpeg_quality=getattr(config, "questioner_jpeg_quality", 90),
                    )
                elif questioner_api_url:
                    if getattr(config, "questioner_type", "") == "plugir":
                        self.question_model = PlugIRQuestionerHTTP(
                            questioner_api_url,
                            timeout=getattr(config, "questioner_api_timeout", 300),
                            question_options=getattr(config, "plugir_question_options", 5),
                        )
                        caption_cache = getattr(config, "plugir_caption_cache", None)
                        if not caption_cache or not os.path.isfile(caption_cache):
                            raise FileNotFoundError("PlugIR caption cache is missing: " + str(caption_cache))
                        with open(caption_cache) as handle:
                            self.plugir_caption_by_path = {
                                row["image_path"]: row["caption"].strip()
                                for line in handle if line.strip()
                                for row in [json.loads(line)] if row.get("caption", "").strip()
                            }
                        self.plugir_clusters = getattr(config, "plugir_clusters", 10)
                        self.plugir_candidate_pool_size = getattr(config, "plugir_candidate_pool_size", 74)
                    else:
                        self.question_model = ChatIRQuestionerHTTP(
                            questioner_api_url,
                            timeout=getattr(config, "questioner_api_timeout", 180),
                        )
                else:
                    self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config, logger)
                if config.selector_model_path is not None:
                    self.selector = Selector(SelectorConfig(num_candidates=config.num_candidates - 1))
                    self.load_selector(config, logger)

                self.answer_model = AnswerGeneratorSGLang(
                    getattr(config, "answerer_base_url", "http://localhost:10500/v1"),
                    getattr(config, "answerer_api_key", "Qwen-7B"),
                    getattr(config, "answer_max_tokens", getattr(config, "max_answer_length", 40)),
                    direct=True)
                self.round_state_path = getattr(config, "round_state_path", None)

                self.forward = self._inference
            case _:
                raise ValueError("Unknown stage:", self.stage)

    def _plugir_candidate_captions(self, pool_indices, pool_feats, pool_mask):
        """Select one low-entropy representative from each visual candidate cluster."""
        from sklearn.cluster import KMeans

        result = []
        for indices, feats, mask in zip(pool_indices, pool_feats, pool_mask):
            valid = mask.bool()
            indices, feats = indices[valid], feats[valid].float()
            count = min(self.plugir_clusters, len(indices))
            if count == 0:
                result.append([])
                continue
            feats = F.normalize(feats, dim=-1)
            labels = KMeans(n_clusters=count, random_state=42, n_init=10).fit_predict(feats.cpu().numpy())
            similarity = feats @ feats.T
            probability = F.softmax(similarity, dim=-1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            representatives = []
            for label in range(count):
                members = torch.as_tensor((labels == label).nonzero()[0], device=entropy.device)
                representative = members[entropy[members].argmin()].item()
                path = self.gallery_path[indices[representative].item()]
                caption = self.plugir_caption_by_path.get(path)
                if caption:
                    representatives.append(caption)
            result.append(representatives)
        return result

    def _plugir_select_question(self, context, options, pool_feats, pool_mask):
        """Use the frozen retriever to reproduce PlugIR's KL-based question selection."""
        valid_feats = F.normalize(pool_feats[pool_mask.bool()].float(), dim=-1)
        if not len(options) or valid_feats.numel() == 0:
            return options[0] if options else "What additional detail do you remember about the person?"
        base_feat = F.normalize(self.encode_text([context]).float(), dim=-1)[0]
        previous = F.softmax(valid_feats @ base_feat, dim=0)
        scores = []
        for question in options:
            trial_feat = F.normalize(self.encode_text([(context + " " + question).strip()]).float(), dim=-1)[0]
            proposal = F.softmax(valid_feats @ trial_feat, dim=0)
            scores.append((previous * (previous.clamp_min(1e-12).log() - proposal.clamp_min(1e-12).log())).sum())
        return options[int(torch.stack(scores).argmin().item())]

    def _plugir_filter_options(self, contexts, option_sets):
        """Apply PlugIR's answerability filter with the local frozen Qwen model."""
        system_prompt = (
            "Answer the question only according to the given context. "
            "Output exactly one label: ANSWERABLE if the context determines the answer; "
            "otherwise UNCERTAIN. Do not provide an answer or explanation."
        )
        flat = [(sample_idx, context, question)
                for sample_idx, (context, options) in enumerate(zip(contexts, option_sets))
                for question in options]
        keep, rejected = [[] for _ in option_sets], [[] for _ in option_sets]
        if not flat:
            return keep, rejected

        def judge(item):
            sample_idx, context, question = item
            prompt = f"[Context]\n{context.strip()}\n\n[Question]\n{question}\n[Answer]\n"
            response = self.answer_model.processor.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            answer = response.choices[0].message.content or ""
            return sample_idx, question, answer.strip().upper().startswith("UNCERTAIN"), answer

        with ThreadPoolExecutor(max_workers=min(16, len(flat))) as executor:
            futures = [executor.submit(judge, item) for item in flat]
            for future in as_completed(futures):
                sample_idx, question, is_unanswered, _raw_answer = future.result()
                if is_unanswered:
                    keep[sample_idx].append(question)
                else:
                    rejected[sample_idx].append(question)
        return keep, rejected

    # def forward(self, batch):
    #     raise NotImplementedError

    def _train_retriever(self, batch):
        if not self.training:
            batch_images = batch['images'].to(self.retriever_device)
            image_feats = self.retrieval_model.encode_image(batch_images)
            image_feats = image_feats[:, 0, :]
            return EasyDict({'image_feats': image_feats, 'image_id': batch['image_id']})

        f_cap_ids_list, mlm_ids_batch, mlm_labels_batch = [], [], []
        for i in range(len(batch['fine-grained_caption'])):
            f_cap = batch['fine-grained_caption'][i]
            f_cap_ids = tokenize_simple(f_cap, self.clip_tokenizer, self.r_length)
            mlm_ids, mlm_labels = self.retrieval_model.build_random_masked_tokens_and_labels(f_cap_ids.cpu().numpy(),
                                                                                             self.clip_tokenizer)

            f_cap_ids_list.append(f_cap_ids)
            mlm_ids_batch.append(mlm_ids)
            mlm_labels_batch.append(mlm_labels)

        f_cap_ids_list = torch.stack(f_cap_ids_list, dim=0).cuda()
        mlm_ids_batch = torch.stack(mlm_ids_batch, dim=0).cuda()
        mlm_labels_batch = torch.stack(mlm_labels_batch, dim=0).cuda()
        # m_cap_ids = torch.stack(m_cap_ids, dim=0).cuda()

        batch = {
            'pids': batch['pids'],
            'image_ids': batch['image_ids'],
            'images': batch['images'],
            'caption_ids': f_cap_ids_list,
            'mlm_ids': mlm_ids_batch,
            'mlm_labels': mlm_labels_batch
        }
        ret = self.retrieval_model(batch)
        return ret

    @torch.no_grad()
    def _inference(self, batch):
        '''
        Batch size = N
        image encode:
        batch = {
            'image': Tensor
        }
        text interact:
        batch = {
            'coarse_grained_caption': [List] coarse-grained_caption,
            'fine_grained_caption': [List]fine-grained_caption,
        }
        '''
        if "images" in batch:
            batch_images = batch["images"].to(self.retriever_device)
            image_feats = self.retrieval_model.encode_image(batch_images)
            image_feats = image_feats[:, 0, :]
            return EasyDict({"image_feats": image_feats, "image_id": batch["image_id"]})

        B = len(batch["initial_description"])
        initial_description = batch["initial_description"]
        collection = list(deepcopy(batch["initial_description"]))
        questions_log = [[] for _ in range(B)]
        answers_log = [[] for _ in range(B)]
        rank1_feedback_log = ["" for _ in range(B)]
        round_state_path = getattr(self, "round_state_path", None)
        start_round = 0
        round_collections = [list(collection)]
        ret_dict = EasyDict({"text_id": batch["text_id"],
                             "text_feats": [],
                             "conversations": []})
        for i in range(B):
            conversation = {"id": batch['text_id'][i].item(),
                            "initial_query": batch['initial_description'][i],
                            "interaction": []}
            ret_dict.conversations.append(conversation)

        if round_state_path and os.path.exists(round_state_path):
            state = torch.load(round_state_path, map_location="cpu")
            saved_ids = state.get("text_ids")
            current_ids = [int(x.item()) if torch.is_tensor(x) else int(x)
                           for x in batch["text_id"]]
            if (state.get("batch_size") == B
                    and state.get("data_iter") == batch.get("eval_batch_id")
                    and saved_ids == current_ids):
                collection = state["collection"]
                questions_log = state["questions_log"]
                answers_log = state["answers_log"]
                ret_dict.conversations = state["conversations"]
                round_collections = state.get("round_collections", [list(collection)])
                start_round = int(state.get("round_completed", 0))
                for previous_collection in round_collections[:start_round]:
                    ret_dict.text_feats.append(self.encode_text(previous_collection).unsqueeze(0))
                print(f"Resuming interaction from completed round {start_round}/{self.interact_round}")

        for r in tqdm(range(start_round, self.interact_round), desc="Interactive retrieving", file=sys.stdout, ncols=75):
            text_feats = self.encode_text(collection)
            retrieval_text_feats = text_feats
            pool_size = (
                self.plugir_candidate_pool_size
                if getattr(self.question_model, "needs_candidate_pool", False)
                else math.ceil(self.num_potential_candidates / (r + 1))
            )
            k = torch.full([B], fill_value=pool_size,
                           device=self.retriever_device)
            image_indices, image_feats, padding_mask = self.get_topK_images(text_feats, k)
            candidate_pool_indices = image_indices.clone()
            candidate_pool_feats = image_feats.clone()
            candidate_pool_mask = padding_mask.clone()

            # start_time = time.time()
            if hasattr(self, "selector"):
                if hasattr(self, "selector_retrieval_model"):
                    selector_tokens = [tokenize_simple(text, self.clip_tokenizer, self.r_length)
                                       for text in collection]
                    selector_tokens = torch.stack(selector_tokens).to(self.retriever_device)
                    selector_text_feats = F.normalize(
                        self.selector_retrieval_model.encode_text(selector_tokens), dim=-1)
                    selector_image_feats = self.selector_gallery_cls[
                        image_indices.clamp_min(0)]
                    text_feats = selector_text_feats.to(torch.bfloat16)
                    image_feats = selector_image_feats[:, 1:].to(torch.bfloat16)
                else:
                    text_feats = text_feats.to(torch.bfloat16)
                    image_feats = image_feats[:, 1:].to(torch.bfloat16)
                padding_mask = padding_mask[:, 1:]
                khot = self.selector(text_feats, image_feats, padding_mask=padding_mask, temperature=None)
                khot = torch.cat([torch.ones([B, 1], device=khot.device), khot], dim=1)
            else:
                khot = self.get_topk_images_mask(text_feats, image_feats, padding_mask)

            image_indices = image_indices[khot.bool()].view(B, -1)

            image_paths = [[self.gallery_path[i] for i in image_indices[idx]] for idx in range(B)]

            rank1_feedback_log = ["" for _ in range(B)]

            active_indices = list(range(B))
            if len(active_indices) > 0:
                if getattr(self.question_model, "needs_candidate_pool", False):
                    print(f"[PlugIR] round={r}: building representative captions", flush=True)
                    representative_captions = self._plugir_candidate_captions(
                        candidate_pool_indices, candidate_pool_feats, candidate_pool_mask)
                    print(f"[PlugIR] round={r}: requesting question options", flush=True)
                    question_options = self.question_model(
                        [initial_description[i] for i in active_indices],
                        [image_paths[i] for i in active_indices],
                        [questions_log[i] for i in active_indices],
                        [answers_log[i] for i in active_indices],
                        [rank1_feedback_log[i] for i in active_indices],
                        [representative_captions[i] for i in active_indices],
                        [collection[i] for i in active_indices])
                    print(f"[PlugIR] round={r}: filtering answerable options with local Qwen", flush=True)
                    question_options, rejected_options = self._plugir_filter_options(
                        [collection[i] for i in active_indices], question_options)
                    target_options = self.question_model.question_options
                    # PlugIR's referring branch feeds rejected questions back to
                    # the generator. Here each state receives up to three
                    # generation attempts to collect target_options valid
                    # candidates, matching the source loop while batching API
                    # requests across unresolved states.
                    for retry in range(2):
                        unresolved = [
                            local_idx for local_idx, options in enumerate(question_options)
                            if len(options) < target_options
                        ]
                        if not unresolved:
                            break
                        print(f"[PlugIR] round={r}: re-sampling {len(unresolved)} unresolved states", flush=True)
                        retry_options = self.question_model(
                            [initial_description[active_indices[i]] for i in unresolved],
                            [image_paths[active_indices[i]] for i in unresolved],
                            [questions_log[active_indices[i]] for i in unresolved],
                            [answers_log[active_indices[i]] for i in unresolved],
                            [rank1_feedback_log[active_indices[i]] for i in unresolved],
                            [representative_captions[active_indices[i]] for i in unresolved],
                            [collection[active_indices[i]] for i in unresolved],
                            prior_questions=[
                                question_options[i] + rejected_options[i]
                                for i in unresolved
                            ])
                        retry_keep, retry_rejected = self._plugir_filter_options(
                            [collection[active_indices[i]] for i in unresolved], retry_options)
                        for local_idx, kept, rejected in zip(unresolved, retry_keep, retry_rejected):
                            for question in kept:
                                if question not in question_options[local_idx]:
                                    question_options[local_idx].append(question)
                            question_options[local_idx] = question_options[local_idx][:target_options]
                            rejected_options[local_idx].extend(rejected)
                    unresolved = [
                        local_idx for local_idx, options in enumerate(question_options)
                        if len(options) < target_options
                    ]
                    if unresolved:
                        print(f"[PlugIR] round={r}: source fallback for {len(unresolved)} unresolved states", flush=True)
                        fallback_options = self.question_model(
                            [initial_description[active_indices[i]] for i in unresolved],
                            [image_paths[active_indices[i]] for i in unresolved],
                            [questions_log[active_indices[i]] for i in unresolved],
                            [answers_log[active_indices[i]] for i in unresolved],
                            [rank1_feedback_log[active_indices[i]] for i in unresolved],
                            [representative_captions[active_indices[i]] for i in unresolved],
                            [collection[active_indices[i]] for i in unresolved],
                            fallback=True)
                        fallback_keep, _ = self._plugir_filter_options(
                            [collection[active_indices[i]] for i in unresolved], fallback_options)
                        for local_idx, kept in zip(unresolved, fallback_keep):
                            for question in kept:
                                if question not in question_options[local_idx]:
                                    question_options[local_idx].append(question)
                            while len(question_options[local_idx]) < target_options:
                                # PlugIR's exact final fallback when its generic
                                # proposal is also answerable from current text.
                                question_options[local_idx].append(
                                    "What is the other object in the image?"
                                )
                            question_options[local_idx] = question_options[local_idx][:target_options]
                    print(f"[PlugIR] round={r}: selecting question from options", flush=True)
                    questions_active = [
                        self._plugir_select_question(
                            collection[sample_idx], options,
                            candidate_pool_feats[sample_idx], candidate_pool_mask[sample_idx])
                        for sample_idx, options in zip(active_indices, question_options)
                    ]
                else:
                    questions_active = self.question_model(
                        [initial_description[i] for i in active_indices],
                        [image_paths[i] for i in active_indices],
                        [questions_log[i] for i in active_indices],
                        [answers_log[i] for i in active_indices],
                        [rank1_feedback_log[i] for i in active_indices])

                if getattr(self.question_model, "needs_candidate_pool", False):
                    print(f"[PlugIR] round={r}: querying answerer", flush=True)
                answers_active = self.answer_model(
                    [batch['fine-grained_description'][i] for i in active_indices],
                    questions_active)

                for local_idx, sample_idx in enumerate(active_indices):
                    question = questions_active[local_idx]
                    answer = answers_active[local_idx]
                    collection[sample_idx] = (collection[sample_idx] + ' ' + delete_prefix(answer)).strip()
                    questions_log[sample_idx].append(question)
                    answers_log[sample_idx].append(answer)
                    ret_dict.conversations[sample_idx]["interaction"].append({
                        "round": r,
                        "question": question,
                        "answer": answer,
                        "candidate": image_paths[sample_idx]
                    })

            ret_dict.text_feats.append(retrieval_text_feats.unsqueeze(0))
            round_collections.append(list(collection))
            if round_state_path:
                os.makedirs(os.path.dirname(round_state_path), exist_ok=True)
                state_tmp = round_state_path + ".tmp"
                torch.save({
                    "data_iter": batch.get("eval_batch_id"),
                    "batch_size": B,
                    "text_ids": [int(x.item()) if torch.is_tensor(x) else int(x)
                                 for x in batch["text_id"]],
                    "round_completed": r + 1,
                    "collection": collection,
                    "round_collections": round_collections,
                    "questions_log": questions_log,
                    "answers_log": answers_log,
                    "conversations": ret_dict.conversations,
                }, state_tmp)
                os.replace(state_tmp, round_state_path)
                print(f"Saved round checkpoint: batch={batch.get('eval_batch_id')} round={r + 1}", flush=True)

        text_feats = self.encode_text(collection)
        ret_dict.text_feats.append(text_feats.unsqueeze(0))
        ret_dict.text_feats = torch.cat(ret_dict.text_feats, dim=0)

        for i in range(B):
            if i == 0:
                print('------- Case --------')
                print('Fine-grained description', batch['fine-grained_description'][i])
                print('Initial query:', initial_description[i])
                for j in range(len(questions_log[i])):
                    print("Q#{}:".format(j + 1), questions_log[i][j])
                    print("A#{}:".format(j + 1), answers_log[i][j])
                print('Final Query:', collection[i])
                print('')

        return ret_dict

    def _warmup_selector(self, batch):
        descriptions = batch["descriptions"]
        rounds = batch["rounds"]
        text_feats = self.encode_text(descriptions)
        k = torch.ceil(self.num_potential_candidates / (rounds + 1))
        image_indices, image_feats, padding_mask = self.get_topK_images(text_feats, k)
        img_kmeans_mask = self.get_topk_images_mask(text_feats, image_feats, padding_mask)

        text_feats = text_feats.to(torch.bfloat16)
        image_feats = image_feats[:, 1:].to(torch.bfloat16)
        padding_mask = padding_mask[:, 1:]
        khot = self.selector(text_feats, image_feats, padding_mask=padding_mask, temperature=1.0)
        khot = torch.cat([torch.ones([khot.shape[0], 1], device=khot.device), khot], dim=1)

        acc = (khot[:, 1:] == img_kmeans_mask[:, 1:].bool()).float()[img_kmeans_mask[:, 1:].bool()].mean()

        khot = khot[img_kmeans_mask.bool()]
        img_kmeans_mask = img_kmeans_mask[img_kmeans_mask.bool()]

        selector_loss = nn.BCELoss()(khot, img_kmeans_mask.to(torch.float))
        return dict(loss=selector_loss, acc=acc)

    def _train_questioner_selector(self,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   labels: torch.Tensor,
                                   n_rounds: torch.Tensor,
                                   descriptions: List[str],
                                   temperature: float = None):
        B = len(descriptions)
        text_feats = self.encode_text(descriptions)
        k = torch.ceil(self.num_potential_candidates / (n_rounds + 1))
        # k = torch.full([B], fill_value=self.num_potential_candidates, device=self.retriever_device)
        image_indices, image_feats, padding_mask = self.get_topK_images(text_feats, k)
        if temperature is None:
            khot = self.get_topk_images_mask(text_feats, image_feats, padding_mask)
        else:
            text_feats = text_feats.to(torch.bfloat16)
            image_feats = image_feats[:, 1:].to(torch.bfloat16)
            padding_mask = padding_mask[:, 1:]
            khot = self.selector(text_feats, image_feats, padding_mask=padding_mask, temperature=temperature)
            khot = torch.cat([torch.ones([B, 1], device=khot.device), khot], dim=1)

        image_indices = image_indices[khot.bool()].view(B, self.num_candidates)
        khot_grad_c = khot[torch.nonzero(khot, as_tuple=True)]

        image_paths = [[self.gallery_path[i] for i in image_indices[idx]] for idx in range(B)]
        processor = self.question_model.image_processor
        images = [[process_image_train(im, processor, self.image_folder) for im in sl] for sl in image_paths]
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": [im[0].to(torch.bfloat16) for im_list in images for im in im_list],
            "image_sizes": [im[1] for im_list in images for im in im_list],
            "modalities": [im[2] for im_list in images for im in im_list],
            "selector_grad_c": khot_grad_c.to(torch.bfloat16)
        }
        ret = self.question_model(**batch)
        return ret

    @torch.no_grad()
    def get_topK_images(self, text_feats: torch.Tensor, k: torch.Tensor):
        N, max_K = text_feats.shape[0], self.num_potential_candidates
        if text_feats.ndim == 1:
            text_feats = text_feats.unsqueeze(0)
        similarity = text_feats @ self.gallery_cls.t()
        top_k_img_ids = torch.topk(similarity, max_K, dim=1, largest=True).indices
        top_k_img_feats = self.gallery_cls[top_k_img_ids]

        padding_mask = torch.arange(max_K).to(text_feats.device).unsqueeze(0).expand(N, max_K)
        padding_mask = torch.less(padding_mask, k.unsqueeze(1))
        top_k_img_ids[torch.logical_not(padding_mask)] = -1

        return top_k_img_ids, top_k_img_feats, padding_mask

    @torch.no_grad()
    def get_topk_images_mask(self, text_feats: torch.Tensor, image_feats: torch.Tensor, padding_mask: torch.BoolTensor):
        if text_feats.ndim == 1:
            text_feats = text_feats.unsqueeze(0)
        candidate_mask = torch.zeros([image_feats.shape[0], self.num_potential_candidates], dtype=torch.bool,
                                     device=self.retriever_device)
        for idx in range(image_feats.shape[0]):
            top_k_similarity = text_feats[idx].unsqueeze(0) @ image_feats[idx].t()
            candidate_ids = torch.topk(top_k_similarity, self.num_candidates, dim=1).indices
            candidate_mask[idx, candidate_ids] = True
        return candidate_mask

    @torch.no_grad()
    def _prepare_data(self, batch):
        if 'images' in batch:
            batch_images = batch['images'].to(self.retriever_device)
            image_feats = self.retrieval_model.encode_image(batch_images)
            image_feats = image_feats[:, 0, :]
            return EasyDict({'image_feats': image_feats})

        B = len(batch['initial_query'])
        image_ids, pids, answers = batch['image_ids'], batch["pids"], batch['answers']
        dialog_history = batch['initial_query']

        QA_indices = torch.zeros([B, self.interact_round], dtype=torch.int, device='cuda')
        rank_all = {'R1': [], 'R5': [], 'R10': []}
        for r in range(self.interact_round):
            rank_this_round = []
            for i in range(B):
                if r > 0 and QA_indices[i, r - 1] == -100:
                    continue
                dialog = dialog_history[i]
                ans_set = answers[i]
                candidates, cand_index = [], []
                for idx, ans in enumerate(ans_set):
                    ans = delete_prefix(ans)
                    if ans not in dialog:
                        candidates.append(dialog_history[i] + ' ' + ans)
                        cand_index.append(idx)
                if len(candidates) < 2:
                    QA_indices[i, r] = -100
                    continue
                for j in range(len(candidates)):
                    candidates[j] = tokenize_simple(candidates[j], self.clip_tokenizer,
                                                    self.r_length).unsqueeze(0)

                candidates = torch.cat(candidates, dim=0).to(self.retriever_device)
                with torch.no_grad():
                    text_cand_cls = F.normalize(self.retrieval_model.encode_text(candidates), dim=-1, p=2)

                similarity = text_cand_cls @ self.gallery_cls.t()
                query_labels = torch.full((similarity.shape[0],), pids[i], dtype=torch.long,
                                          device=self.retriever_device)

                ranks = per_sample_ranks(similarity, query_labels, self.gallery_pid)
                select_idx = torch.argmin(ranks)

                rank_this_round.append(ranks[select_idx])

                select_idx = cand_index[select_idx]
                QA_indices[i, r] = select_idx
                dialog_history[i] += ' ' + delete_prefix(answers[i][select_idx])

            rank_this_round = torch.tensor(rank_this_round)
            rank_all['R1'].append(round((rank_this_round == 1).float().mean().item() * 100, 1))
            rank_all['R5'].append(round((rank_this_round <= 5).float().mean().item() * 100, 1))
            rank_all['R10'].append(round((rank_this_round <= 10).float().mean().item() * 100, 1))
        print(rank_all)
        ret = EasyDict({
            'image_ids': image_ids,
            'initial_query': batch['initial_query'],
            'QA_indices': QA_indices,
        })
        return ret

    @torch.no_grad()
    def encode_text(self, text):
        # Keep full-test-set round evaluation memory-safe.
        encoded = []
        for start in range(0, len(text), 512):
            chunk = text[start:start + 512]
            with torch.amp.autocast('cuda', enabled=True):
                text_tokens = [tokenize_simple(t, self.clip_tokenizer, self.r_length) for t in chunk]
                text_tokens = torch.stack(text_tokens, dim=0).to(self.retriever_device)
                encoded.append(F.normalize(self.retrieval_model.encode_text(text_tokens), dim=-1))
        return torch.cat(encoded, dim=0)

    @torch.no_grad()
    def set_gallery(self, gallery_cls, gallery_image_path, **kwargs):
        self.gallery_cls = F.normalize(gallery_cls.to(self.retriever_device), dim=-1)
        self.gallery_path = gallery_image_path
        if "gallery_pid" in kwargs.keys():
            self.gallery_pid = kwargs.pop("gallery_pid").to(self.retriever_device)
        rank0_print(">>> set_gallery unused data: ", kwargs.keys())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        return self.question_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _build_retrieval_model(self, cfg, device, training=True, logger=None):
        retriever_type = str(getattr(cfg, "retriever_type", "irra")).lower()
        if retriever_type == "rde":
            if training:
                raise ValueError("RDE retriever is evaluation-only in RAVEL")
            rde_project_root = getattr(cfg, "rde_project_root", None) or os.environ.get(
                "RAVEL_RDE_PROJECT_ROOT"
            )
            rde_checkpoint_dir = getattr(cfg, "rde_checkpoint_dir", None) or os.environ.get(
                "RAVEL_RDE_CHECKPOINT_DIR"
            )
            if not rde_project_root or not rde_checkpoint_dir:
                raise ValueError(
                    "retriever_type=rde requires rde_project_root/rde_checkpoint_dir "
                    "or RAVEL_RDE_PROJECT_ROOT/RAVEL_RDE_CHECKPOINT_DIR"
                )
            retrieval_model, text_length = load_rde_retriever(
                rde_project_root, rde_checkpoint_dir
            )
            self.r_length = text_length
            self.clip_tokenizer = SimpleTokenizer()
            return retrieval_model
        if retriever_type != "irra":
            raise ValueError(f"Unknown retriever_type: {retriever_type}")

        rank0_print("Building retrieval on device: {}".format(device))
        retrieval_model_config = EasyDict({
            'pretrain_choice': cfg.clip_pretrain_model,
            'img_size': cfg.img_size,
            'stride_size': cfg.stride_size,
            'text_length': cfg.max_retrieve_length,
            'temperature': cfg.temperature,
            'vocab_size': cfg.vocab_size,
            'training': cfg.stage == "train_retriever",
            'num_classes': cfg.num_classes
        })
        retrieval_model = IRRA(retrieval_model_config, device)
        ckpt_path = os.path.join(cfg.output_dir, 'retrieval_model_mix_IRRA')
        if self.stage != 'train_retriever':
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError('No checkpoint of retrieval model in {}'.format(ckpt_path))
            else:
                ckpt_path = os.path.join(ckpt_path, 'checkpoint.pth')
                ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
                response = retrieval_model.load_state_dict(ckpt, strict=False)
                if logger is not None:
                    logger.info(f'Load retrieval model from {ckpt_path}. Missing Parameters: {response.missing_keys}')
                else:
                    rank0_print(f'Load retrieval model from {ckpt_path}. Missing Parameters: {response.missing_keys}')

        convert_weights(retrieval_model)
        if not training:
            for name, param in retrieval_model.named_parameters():
                param.requires_grad = False
        retrieval_model.to(device)
        self.clip_tokenizer = SimpleTokenizer()
        return retrieval_model

    def load_selector(self, cfg, logger):
        ckpt_path = os.path.join(cfg.selector_model_path, "selector.pth")
        if not os.path.exists(ckpt_path) or not os.path.isfile(ckpt_path):
            raise FileNotFoundError('No checkpoint of selector model in {}'.format(ckpt_path))
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            response = self.selector.load_state_dict(ckpt, strict=False)
            if logger is not None:
                logger.info(f'Load selector model from {ckpt_path}: {response.missing_keys}')
            else:
                rank0_print(f'Load selector model from {ckpt_path}. Missing Parameters: {response.missing_keys}')
            self.selector.to(self.retriever_device)


class LlavaForPersonReIDQuestionModel(nn.Module):

    def __init__(self, questioner_config, llava_args, logger=None):
        super(LlavaForPersonReIDQuestionModel, self).__init__()
        self.num_candidates = questioner_config.num_candidates
        self.prompt = prompt_question_generator_v3
        self.mini_batch_size = getattr(questioner_config, "questioner_batch_size", 4)
        model, tokenizer = load_llava_reid_model(llava_args, logger)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = self.model.get_vision_tower().image_processor
        self.conv_template = conv_templates['qwen_reid']
        match questioner_config.stage:
            case "train_questioner" | "train_selector":
                self.forward = self.model.forward
            case "eval":
                self.forward = self._eval

    def generate_mini_batch(self, initial_query: list[str], candidate_images: list[list[str]],
                            questions: list[list[str]], answers: list[list[str]],
                            rank1_feedback: list[str] | None = None):
        """
                initial_query: List[str], len = N
                candidate_images: List[List[str]]
                question: List[str] len = N
                labels: List[str] len = N
        """
        B = len(initial_query)
        conv_batch = wrap_question_prompt(self.prompt, initial_query, questions, answers, self.num_candidates, True)
        image_all = [Image.open(img) for sublist in candidate_images for img in sublist]
        image_tensor = process_images(image_all, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16) for _image in image_tensor]
        image_sizes = [_image.size for _image in image_all]

        input_ids_list, answers = [], []
        for idx in range(B):
            conv = deepcopy(self.conv_template)
            conv.append_message(conv.roles[0], conv_batch[idx])
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_text, self.tokenizer, return_tensors='pt')
            input_ids_list.append(input_ids)

        input_ids_batch = pad_sequence(self.tokenizer, input_ids_list, True, self.tokenizer.pad_token_id)
        attention_mask_batch = input_ids_batch.ne(self.tokenizer.pad_token_id).to(self.model.device)
        input_ids_batch = input_ids_batch.to(self.model.device)

        gen_ids = self.model.generate(input_ids_batch, images=image_tensor, image_sizes=image_sizes,
                                      modalities=["image"] * len(image_all),
                                      attention_mask=attention_mask_batch,
                                      do_sample=True, top_p=0.5,
                                      begin_suppress_tokens=[self.tokenizer.eos_token_id],
                                      max_new_tokens=100)
        questions = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for i in range(len(questions)):
            if questions[i].startswith("assistant\n"):
                questions[i] = questions[i].replace("assistant\n", "")
        return questions

    def _eval(self, initial_query: list[str], candidate_images: list[list[str]],
              question: list[list[str]], answer: list[list[str]],
              rank1_feedback: list[str] | None = None):
        question_all = []
        for idx in range(0, len(initial_query), self.mini_batch_size):
            end_idx = min(idx + self.mini_batch_size, len(initial_query))
            feedback_mini_batch = None if rank1_feedback is None else rank1_feedback[idx:end_idx]
            question_mini_batch = self.generate_mini_batch(initial_query[idx:end_idx],
                                                           candidate_images[idx:end_idx],
                                                           question[idx:end_idx],
                                                           answer[idx:end_idx],
                                                           feedback_mini_batch)
            question_all.extend(question_mini_batch)
        return question_all

    def get_model(self):
        return self.model

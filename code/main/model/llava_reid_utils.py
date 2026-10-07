import json
import os
import re
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import transformers
from PIL import Image
from torch import nn
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.llava import conversation as conversation_lib
from utils.openai_processor import format_request, OpenAIBatchProcessor
from utils.iotools import LoggerX

from .llava.mm_utils import expand2square
from .llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
import prompt
from .llava.train.train_utils import find_all_linear_names
from .llava.utils import rank0_print
from .llava_qwen_reid import LlavaQwenReIDConfig


warnings.filterwarnings("ignore", category=UserWarning)


def delete_prefix(text):
    text = text.replace('Yes, ', '').replace('No, ', '')
    text = re.sub(r'^[A-Z]\)\s*', '', text)
    if len(text) > 1:
        text = text[0].upper() + text[1:]
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # If the last sentence ends with a period, question mark, or exclamation mark, it is complete
    sentences = [s for s in sentences if
                 not (s.startswith("I don't know") or s.startswith("I don't") or s.startswith("I do not"))]
    if len(sentences) > 0 and not re.match(r'.*[\.\?\!]$', sentences[-1]):
        sentences = sentences[:-1]
    text = ' '.join(sentences)
    return text


def load_question_model(model_path, logger: LoggerX = None, attn_implementation="sdpa"):
    """加载 LLaVA 模型（微调版或原生版均使用 LlavaQwenForCausalLM）"""
    logger.info("Load question model from {}".format(model_path))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=["mm_projector"],
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = LlavaQwenForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="cuda:0",
        attn_implementation=attn_implementation,
        trust_remote_code=True
    )

    model.config.tokenizer_padding_side = 'left'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    return model, tokenizer

    # config = LlavaQwenConfig.from_pretrained(model_path)
    # bnb_config = BitsAndBytesConfig(**dict(config.quantization_config))
    # logger.debug(bnb_config)
    # with init_empty_weights():
    #     model = LlavaQwenForCausalLM.from_pretrained(model_path,
    #                                                  quantization_config=bnb_config,
    #                                                  device_map="cpu")
    # device_map = infer_auto_device_map(model,
    #                                    max_memory={0: "8GiB", 1: "12GiB"},
    #                                    no_split_module_classes=["Qwen2DecoderLayer", "SigLipVisionTower"])
    # for k in device_map.keys():
    #     device_map[k] = 0 if k == 'model.vision_tower' else 1
    # logger.debug(f"device_map {str(dict(device_map))}")
    #
    # model = LlavaQwenForCausalLM.from_pretrained(model_path, device_map=device_map)
    # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    # image_processor = model.get_vision_tower().image_processor
    # return tokenizer, model, image_processor, 4096


def load_llava_reid_model(kwargs, logger=None):
    if "model_path" in kwargs.keys():
        return load_question_model(**kwargs, logger=logger)
    else:
        return load_llava_reid_model_for_training(**kwargs)


def load_llava_reid_model_for_training(model_args, data_args, training_args):
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],  # [, "mm_projector", "lm_head", "vision_tower"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    config = LlavaQwenReIDConfig.from_pretrained(model_args.model_name_or_path)

    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")
    model = LlavaQwenForPersonReID.from_pretrained(
        # model_args.model_name_or_path,
        config._name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=None,
        **bnb_model_from_pretrained_args,
    )

    # model = PeftModel.from_pretrained(model, "./weight/official-selector-baseline")
    # model = model.merge_and_unload()

    # model = get_model(model_args, training_args, bnb_model_from_pretrained_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config._name_or_path,  # model_args.model_name_or_path,
                                                           cache_dir=training_args.cache_dir,
                                                           model_max_length=training_args.model_max_length,
                                                           padding_side="right")
    rank0_print(f"Prompt version: {model_args.version}")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    rank0_print(tokenizer)

    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32)
        rank0_print("prepare_model_for_kbit_training")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    else:
        model.requires_grad_(True)
        if model_args.freeze_backbone:
            print("Freezing backbone")
            model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    if training_args.lora_enable:
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        from peft import LoraConfig, get_peft_model
        if os.path.exists(os.path.join(model_args.model_name_or_path, "adapter_config.json")):
            rank0_print(f"Loading Pretrained LoRA adapters from {model_args.model_name_or_path}...")
            joint_training = "mm_language_model" in model_args.mm_tunable_parts
            if joint_training:
                rank0_print("Continue training LoRA weights.")
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=joint_training)
        else:
            rank0_print("Adding LoRA adapters...")
            tunable_parts = model_args.mm_tunable_parts.split(",")
            skip_keywords = []
            if "mm_language_model" not in tunable_parts:
                skip_keywords.append("model.layers")
            if "mm_mlp_adapter" not in tunable_parts:
                skip_keywords.append("mm_projector")
            if "mm_vision_resampler" not in tunable_parts:
                skip_keywords.append("vision_resampler")
            if "mm_vision_tower" not in tunable_parts:
                skip_keywords.append("vision_tower")
            rank0_print("skipping model from LoRA: ", skip_keywords)
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model, skip_keywords=skip_keywords),
                modules_to_save=["model.mm_projector"],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
    else:
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(False)
        model.get_model().vision_resampler.requires_grad_(False)
        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_mlp_adapter" in tunable_parts:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if "mm_vision_resampler" in tunable_parts:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    try:
                        param.requires_grad_(True)
                    except RuntimeError:
                        print(name, param.data.dtype)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                    param.requires_grad_(True)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # rank0_print(model)
    return model, tokenizer


def process_image_train(image_file, image_processor, image_folder):
    try:
        image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
    except Exception as exn:
        print(f"Failed to open image {image_file}. Exception:", exn)
        raise exn
    image_size = image.size
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image, image_size, "image"


class AnswerGeneratorSGLang(nn.Module):
    def __init__(self, base_url, api_key="EMPTY", max_tokens=40, direct=False):
        super(AnswerGeneratorSGLang, self).__init__()
        self.processor = OpenAIBatchProcessor(base_url, api_key)
        self.end_point = "/v1/chat/completions"
        self.input_file_path = f"SGLang_storage/batch_request_{uuid.uuid4()}.jsonl"
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.prompt = prompt.prompt_answer_generator
        self.max_tokens = max_tokens
        self.direct = direct
        os.makedirs("SGLang_storage", exist_ok=True)
        rank0_print(
            f"Loading answer generator from {base_url} with key {api_key}, "
            f"max_tokens={max_tokens}, direct={direct}"
        )

    @torch.no_grad()
    def forward(self, context, question):
        if len(context) > 256:
            answers = []
            for start in range(0, len(context), 256):
                answers.extend(self.forward(context[start:start + 256], question[start:start + 256]))
            return answers
        jsonl_file, answers = [], []
        for idx, (ctx, q) in enumerate(zip(context, question)):
            message = self.prompt.replace('<|context|>', ctx.replace("I saw ", "You saw ")).replace('<|question|>', q)
            request = format_request(idx, self.system_prompt,
                                     user_prompt=message, image_path=None,
                                     end_point=self.end_point,
                                     temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=self.max_tokens)
            jsonl_file.append(request)

        ans_response = None
        if not self.direct:
            with open(self.input_file_path, 'w') as f:
                for entry in jsonl_file:
                    f.write(json.dumps(entry) + '\n')
            ans_response = self.processor.process_batch(self.input_file_path, self.end_point, interval=3)
        if ans_response is None:
            if not self.direct:
                rank0_print("Answer generator batch request failed; falling back to parallel direct chat requests.")
            max_workers = max(1, min(len(jsonl_file), int(os.environ.get("ANSWER_DIRECT_WORKERS", "16"))))

            def direct_request(request):
                response = self.processor.client.chat.completions.create(**request["body"])
                return {
                    "custom_id": request["custom_id"],
                    "response": {
                        "body": {
                            "choices": {
                                "message": {
                                    "content": response.choices[0].message.content
                                }
                            }
                        }
                    }
                }

            if max_workers == 1:
                ans_response = [direct_request(request) for request in jsonl_file]
            else:
                ans_response = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(direct_request, request) for request in jsonl_file]
                    for future in as_completed(futures):
                        ans_response.append(future.result())
        ans_response = sorted(ans_response, key=lambda x: int(x["custom_id"]))
        for ans in ans_response:
            ans = ans["response"]["body"]["choices"]["message"]["content"]
            answers.append(ans)
        for idx, q in enumerate(question):
            if len(q) < 15:
                answers[idx] = "I don't know."
        invalid_question = sum(1 for q in question if len(q) < 15)
        if invalid_question > 0:
            print("Invalid question:", invalid_question)
        return answers


class AnswerVisualGeneratorSGLang(nn.Module):
    def __init__(self, base_url, api_key="EMPTY"):
        super(AnswerVisualGeneratorSGLang, self).__init__()
        self.processor = OpenAIBatchProcessor(base_url, api_key)
        self.end_point = "/v1/chat/completions"
        self.input_file_path = f"SGLang_storage/batch_request_{uuid.uuid4()}.jsonl"
        self.system_prompt = "You are a helpful assistant."
        self.prompt = prompt.prompt_answer_visual_generator
        os.makedirs("SGLang_storage", exist_ok=True)
        rank0_print(f"Loading answer visual generator from {base_url} with key {api_key}")

    @torch.no_grad()
    def forward(self, context, question):
        jsonl_file, answers = [], []
        for idx, (ctx, q) in enumerate(zip(context, question)):
            message = self.prompt.replace('<|question|>', q)
            # print(ctx, q)
            request = format_request(idx, self.system_prompt,
                                     user_prompt=message, image_path=ctx,
                                     end_point=self.end_point,
                                     temperature=0.01,
                                     max_tokens=40)
            jsonl_file.append(request)

        with open(self.input_file_path, 'w') as f:
            for entry in jsonl_file:
                f.write(json.dumps(entry) + '\n')
        ans_response = self.processor.process_batch(self.input_file_path, self.end_point, interval=3)
        for ans in ans_response:
            ans = ans["response"]["body"]["choices"]["message"]["content"]
            answers.append(ans)
        for idx, q in enumerate(question):
            if len(q) < 15:
                answers[idx] = "I don't know."
        invalid_question = sum(1 for q in question if len(q) < 10)
        if invalid_question > 0:
            print("Invalid question:", invalid_question)
        return answers



class Rank1VerifierSGLang(nn.Module):
    def __init__(self, base_url, api_key="EMPTY"):
        super(Rank1VerifierSGLang, self).__init__()
        self.processor = OpenAIBatchProcessor(base_url, api_key)
        self.end_point = "/v1/chat/completions"
        self.input_file_path = f"SGLang_storage/batch_request_{uuid.uuid4()}.jsonl"
        self.system_prompt = "You are a careful eyewitness verifier."
        self.prompt = (
            "Imagine you are a witness helping police identify a person.\n\n"
            "This is what you remember:\n"
            "<|context|>\n\n"
            "You are shown one candidate image.\n\n"
            "Decide whether this candidate is likely to be the person you remember.\n\n"
            "Be careful and realistic:\n"
            "- Clothing color, clothing type, backpack, shoes, hair, posture, body build, and surroundings can all be useful evidence.\n"
            "- The candidate image may or may not be the exact same frame as your memory; the same person may appear from a different angle, at a nearby location, or with some details less visible.\n"
            "- Choose STOP if the visible evidence, taken together, makes you reasonably confident this is the same person.\n"
            "- Choose CONTINUE if the candidate looks like a different person, if there is a clear mismatch, or if the evidence is too weak to identify the person.\n"
            "- Do not infer details that are not visible.\n\n"
            "Output exactly one line:\n"
            "Final decision: STOP\n"
            "or\n"
            "Final decision: CONTINUE"
        )
        os.makedirs("SGLang_storage", exist_ok=True)
        rank0_print(f"Loading rank1 verifier from {base_url} with key {api_key}")

    @staticmethod
    def parse_stop(response: str) -> bool:
        response = response.strip().upper()
        if "FINAL DECISION:" in response:
            decision = response.rsplit("FINAL DECISION:", 1)[-1].strip()
            if decision.startswith("CONTINUE"):
                return False
            if decision.startswith("STOP"):
                return True
        if response.startswith("CONTINUE"):
            return False
        if response.startswith("STOP"):
            return True
        return False

    @torch.no_grad()
    def forward(self, context, image_path):
        raw_answers = []
        for idx, (ctx, img) in enumerate(zip(context, image_path)):
            message = self.prompt.replace('<|context|>', ctx.replace("I saw ", "You saw "))
            request = format_request(idx, self.system_prompt,
                                     user_prompt=message, image_path=img,
                                     end_point=self.end_point,
                                     temperature=0.01,
                                     max_tokens=8)
            try:
                response = self.processor.client.chat.completions.create(**request["body"])
                raw_answers.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Rank1 verifier request failed at index {idx}. Exception: {e}")
                raw_answers.append("CONTINUE")

        stop_flags = [self.parse_stop(ans) for ans in raw_answers]
        return stop_flags, raw_answers

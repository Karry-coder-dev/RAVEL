import argparse
import os
import pathlib
import tempfile
import warnings
from dataclasses import dataclass, asdict

import wandb

import tokenizers

# import model.llava.model.language_model.llava_qwen
from model.llava.train.llava_trainer import LLaVATrainer
from model.llava.train.train_utils import *
from model.llava.utils import rank0_print
from model.llava_reid import LlavaForPersonReID
from utils.iotools import save_model
from utils.misc import is_main_process
from utils.args_parser import get_args_parser

torch.multiprocessing.set_sharing_strategy("file_system")
from packaging import version

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={
        "help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={
            "help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_spatial_pool_ratio: Optional[float] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    mm_newline_position: Optional[str] = field(default="one_token")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True,
                               metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",
                            metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    mm_selector_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)

    language_model_warm_up_steps: int = field(default=None)
    gumbel_temperature: float = field(default=None)
    gumbel_temperature_end: float = field(default=None)

    attn_implementation: str = field(default="flash_attention_2",
                                     metadata={
                                         "help": "Use transformers attention implementation. 'flash_attention_2'"})


@dataclass
class ModelConfig:
    stage: str = field(default="train_questioner")
    interact_round: int = field(default=5)
    num_candidates: int = field(default=10)
    max_answer_length: int = field(default=64)
    max_question_length: int = field(default=128)
    max_retrieve_length: int = field(default=192)
    clip_pretrain_model: str = field(default="ViT-B/16")
    vocab_size: int = field(default=49408)
    temperature: float = field(default=0.02)
    img_aug: bool = field(default=True)
    img_size: tuple = field(default=(384, 128))
    stride_size: int = field(default=16)
    num_classes: int = field(default=-1)
    output_dir: str = field(default="./")
    question_model_path: str = field(default=None)
    selector_model_path: str = field(default=None)
    retriever_type: str = field(default="irra")
    rde_project_root: str = field(default=None)
    rde_checkpoint_dir: str = field(default=None)


def parse_yaml_file_with_env(parser, path):
    """Parse a YAML config after expanding environment variables in its text."""
    config_path = pathlib.Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    expanded_text = os.path.expandvars(raw_text)
    if "${" in expanded_text:
        raise RuntimeError(
            f"Unresolved environment variable in {config_path}. "
            "Set the required variable before launching training."
        )
    if expanded_text == raw_text:
        return parser.parse_yaml_file(str(config_path))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", delete=False
        ) as handle:
            handle.write(expanded_text)
            temporary_path = handle.name
        return parser.parse_yaml_file(temporary_path)
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def train(args):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=config)
    rank0_print(f"Loading config from {args.llava_config_file}")
    model_args, data_args, training_args = parse_yaml_file_with_env(
        parser, args.llava_config_file
    )

    parser = transformers.HfArgumentParser(ModelConfig)
    model_cfg = parser.parse_dict(vars(args), allow_extra_keys=True)[0]
    model_cfg.output_dir = os.path.join(args.output_dir, args.dataset_name + '_' + args.run_name)

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    model = LlavaForPersonReID(config=model_cfg,
                               llava_config={"model_args": model_args,
                                             "data_args": data_args,
                                             "training_args": training_args})

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(
        p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params / 1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params / 1e6:.2f} MB)")

    rank0_print('Loading gallery features')
    pt = torch.load(os.path.join(model_cfg.output_dir, "preprocessed_data.pt"), map_location='cpu')
    model.set_gallery(gallery_cls=pt["image_feats"], gallery_image_path=pt["image_paths"])

    tokenizer = model.question_model.tokenizer
    data_args.stage = model_cfg.stage
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # for n, p in model.named_parameters():
    #     rank0_print(n, p.requires_grad)
    if "wandb" in training_args.report_to and is_main_process():
        if os.path.exists(os.path.join(training_args.output_dir, "wandb_run_id.log")):
            with open(os.path.join(training_args.output_dir, "wandb_run_id.log"), 'r') as f:
                wandb_run_id = f.readline()
        else:
            wandb_run_id = None
        wandb.init(project="LlavaReid",
                   name=training_args.run_name,
                   config=asdict(model_args) | asdict(training_args) | asdict(data_args),
                   settings=wandb.Settings(_disable_stats=True, ),
                   id=wandb_run_id,
                   resume="must" if wandb_run_id is not None else "never"
                   )
        if wandb_run_id is None:
            wandb_run_id = wandb.run.id
            with open(os.path.join(training_args.output_dir, "wandb_run_id.log"), 'w') as f:
                f.write(str(wandb_run_id))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.question_model.model.config.use_cache = True

    if "mm_selector" in model.question_model.model.config.mm_tunable_parts:
        save_model(name="selector",
                   model=model.selector, save_path=training_args.output_dir,
                   logger=None, args=None)
        rank0_print("Selector model saved to {}".format(training_args.output_dir))
    if "mm_language_model" in model.question_model.model.config.mm_tunable_parts:
        if training_args.lora_enable:
            # state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
            # non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                if hasattr(model.question_model.model, "config"):
                    model.question_model.model.config.save_pretrained(training_args.output_dir)
            #     if hasattr(model, "generation_config"):
            #         model.generation_config.save_pretrained(training_args.output_dir)
            #     model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            #     torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            model.question_model.model.save_pretrained(training_args.output_dir, safe_serialization=False)
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    # Parse arguments
    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        # override with mode specified config
        assert configs['stage'] in ["train_questioner", "train_selector"]
        if configs["stage"] in configs["stage_config"]:
            configs.update(configs['stage_config'][configs['stage']])
            rank0_print(f"Update stage config {configs['stage']}: {configs['stage_config'][configs['stage']]}")
        del configs['stage_config']

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)
    train(args)

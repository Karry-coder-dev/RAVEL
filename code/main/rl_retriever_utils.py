import os

import torch
import torch.nn.functional as F
from easydict import EasyDict

from model.IRRA import IRRA
from model.clip_model import convert_weights
from reid_datasets.bases import tokenize_simple


def _infer_num_classes(ckpt):
    for key, value in ckpt.items():
        if value.ndim == 2 and key.endswith("classifier.weight"):
            return value.shape[0]
    return 0


def build_retriever(cfg, device):
    output_dir = getattr(cfg, "output_dir", f"{cfg.dataset_name}_{cfg.run_name}")
    ckpt_path = os.path.join(output_dir, "retrieval_model_mix_IRRA", "checkpoint.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = EasyDict({
        "pretrain_choice": getattr(cfg, "clip_pretrain_model", "ViT-B/16"),
        "img_size": getattr(cfg, "img_size", (384, 128)),
        "stride_size": getattr(cfg, "stride_size", 16),
        "text_length": cfg.max_retrieve_length,
        "temperature": getattr(cfg, "temperature", 0.02),
        "vocab_size": getattr(cfg, "vocab_size", 49408),
        "training": False,
        "num_classes": _infer_num_classes(ckpt),
    })
    model = IRRA(model_cfg, device)
    model.load_state_dict(ckpt, strict=False)
    convert_weights(model)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def encode_texts(model, tokenizer, texts, max_length, batch_size, device):
    feats = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = [tokenize_simple(text, tokenizer, max_length) for text in batch]
        tokens = torch.stack(tokens, dim=0).to(device)
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            feat = model.encode_text(tokens)
            feat = F.normalize(feat.float(), dim=-1)
        feats.append(feat.cpu())
    return torch.cat(feats, dim=0)

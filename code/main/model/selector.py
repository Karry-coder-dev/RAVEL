from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

from model.clip_model import convert_weights
from model.llava.utils import rank0_print


@dataclass
class SelectorConfig:
    n_layers: int = field(default=5)
    n_heads: int = field(default=8)
    d_input: int = field(default=512)
    d_model: int = field(default=512)
    num_candidates: int = field(default=10)
    mlp_ratio: float = field(default=4.0)


class Selector(nn.Module):
    def __init__(self, config: SelectorConfig):
        super(Selector, self).__init__()
        self.config = config
        self.num_candidates = config.num_candidates
        self.modality_embed = nn.Embedding(2, config.d_model)
        # self.projector = nn.Linear(config.d_input, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=int(config.d_model * config.mlp_ratio),
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.selector = nn.Linear(config.d_model, 1, bias=False)

        self.sampler = SubsetGumbelSampler(self.num_candidates)

    def get_modality_embedding(self, text, image):
        text_embed = self.modality_embed(torch.zeros(text.size(0), dtype=torch.long, device=text.device))
        image_embed = self.modality_embed(torch.ones(image.size(0), dtype=torch.long, device=image.device)).unsqueeze(1)
        # text_embed = 0
        # image_embed = 0
        return text + text_embed, image + image_embed

    def forward(self, text: torch.Tensor, image: torch.Tensor, padding_mask: torch.Tensor, temperature=None):
        '''
        Batch size: B, Sequence Length: L, Hidden Size: D
        text: [B,D]
        image: [B, L, D]
        padding_mask: [B, L]
        Note: we use huggingface style padding mask, where True means attend and False means padding tokens.
        '''
        assert text.size(0) == image.size(0) and text.size(1) == image.size(2)
        assert padding_mask.float().sum(dim=-1).min() >= self.num_candidates
        B, L, D = image.shape
        text, image = text * math.sqrt(self.config.d_model), image * math.sqrt(self.config.d_model)
        text, image = self.get_modality_embedding(text, image)
        x = torch.cat([text.unsqueeze(1), image], dim=1).permute(1, 0, 2)
        src_key_padding_mask = torch.cat([torch.zeros([B, 1], device=image.device, dtype=torch.bool),
                                          torch.logical_not(padding_mask)], dim=1)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x[1:].permute(1, 0, 2)[padding_mask, :].reshape(-1, D)
        x = self.selector(x)

        logits = torch.full([B, L], fill_value=-10000, dtype=x.dtype, device=x.device)
        logits[padding_mask] = x.squeeze()
        logits = torch.log_softmax(logits, dim=1)
        if self.training:
            khot = self.sampler(logits, hard=True, temperature=temperature)
        else:
            top_k_indices = torch.topk(logits, self.num_candidates, dim=1, sorted=True).indices
            khot = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, top_k_indices, True)
        return khot


class SubsetGumbelSampler(nn.Module):
    EPSILON = torch.finfo(torch.bfloat16).tiny

    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def __call__(self, scores, hard=True, temperature=1.0):
        scores = scores.float()
        g = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores)).sample()
        scores = scores + g

        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([self.EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = F.softmax(scores / temperature, dim=1)
            khot = khot + onehot_approx

        if hard:
            ids = torch.topk(khot, self.k, dim=1).indices
            khot_hard = torch.zeros_like(khot).scatter_(1, ids, 1)
            ret = khot_hard - khot.detach() + khot
        else:
            ret = khot
        return ret

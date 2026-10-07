import random
from math import ceil

from easydict import EasyDict

from model import objectives
from model.clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class FilterLayer(nn.Module):
    def __init__(self, ratio=0.5):
        super(FilterLayer, self).__init__()
        self.ratio = ratio

    def forward(self, x, cls_attn, attn_mask, cls_indices):
        '''

        :param x: [N,L,D]
        :param cls_attn: [N,L]
        :param cls_indices: [N,L]
        :return:
        '''

        N, L, D = x.shape
        n_tokens = ceil(L * self.ratio)
        if attn_mask is not None:
            cls_attn[attn_mask] = -1000
        cls_attn[torch.arange(N), cls_indices] = -1000

        topK_indices = cls_attn.topk(dim=-1, k=n_tokens).indices
        print('topK_indices', topK_indices.shape)
        topK_indices = topK_indices.unsqueeze(-1).expand(N, n_tokens, D)
        topK_tokens = torch.gather(input=x, dim=1, index=topK_indices)

        return topK_tokens


class IRRA(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size, args.text_length)
        self.embed_dim = base_cfg['embed_dim']
        self.vocab_size = base_cfg['vocab_size']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if args.training:
            # id_loss classifier
            self.classifier = nn.Linear(self.embed_dim, args.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

            # mlm_loss cross-attn
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=4,
                                                       heads=self.embed_dim // 64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn = self.cross_modal_transformer([x])
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model(image=image).img_feats
        return x.float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model(text=text).text_feats
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()
        N = batch['images'].shape[0]
        images = batch['images'].to(self.device)
        caption_ids = batch['caption_ids'].to(self.device)
        x = self.base_model(image=images, text=caption_ids)
        img_feats, img_attn, text_feats, text_attn = x.img_feats, x.img_attn, x.text_feats, x.text_attn
        image_cls = img_feats[:, 0, :].float()
        text_cls = text_feats[torch.arange(N), caption_ids.argmax(dim=-1)].float()
        logit_scale = self.logit_scale

        # SDM Loss
        ret.update({'sdm_loss': objectives.compute_sdm(image_cls, text_cls, batch['pids'], logit_scale)})
        # ID Loss
        image_logits = self.classifier(image_cls.half()).float()
        text_logits = self.classifier(text_cls.half()).float()
        ret.update({'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids'])})
        # MLM_Loss
        mlm_ids = batch['mlm_ids']
        mlm_feats = self.base_model(text=mlm_ids).text_feats
        x = self.cross_former(mlm_feats, img_feats, img_feats)
        x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
        scores = x.float().reshape(-1, self.args.vocab_size)
        mlm_labels = batch['mlm_labels'].reshape(-1)
        ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)})

        return ret

    def build_random_masked_tokens_and_labels(self, tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

    def __repr__(self):
        return f"IRRA model text length: {self.args.text_length}"


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model, dtype=torch.bfloat16)
    return model

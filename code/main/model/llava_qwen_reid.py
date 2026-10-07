import random
import math
import re
from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from transformers import Qwen2Config, AutoConfig, AutoModelForCausalLM, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.llava.model import LlavaMetaForCausalLM
from model.llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
from model.llava.model.llava_arch import unpad_image


class LlavaQwenReIDConfig(LlavaQwenConfig):
    model_type = "llava_qwen_reid"


class LlavaQwenForPersonReID(LlavaQwenForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenReIDConfig

    def __init__(self, config):
        # super(LlavaQwenForPersonReID, self).__init__(config)
        LlavaQwenForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen_reid"
        config.rope_scaling = None
        self.get_model().vision_resampler = AdaptiveSpatialPool(self.get_vision_tower().num_patches_per_side,
                                                                config.mm_spatial_pool_ratio)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().vision_resampler(image_features)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            selector_grad_c: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            modalities: Optional[List[str]] = ["image"],
            dpo_forward: Optional[bool] = False,
            cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(">>>> inputs ids: ", input_ids.shape)
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds,
             labels) = self.prepare_inputs_labels_for_multimodal_with_selector_grads(input_ids, position_ids,
                                                                                     attention_mask,
                                                                                     past_key_values, labels, images,
                                                                                     selector_grad_c,
                                                                                     modalities,
                                                                                     image_sizes)
        # print("<<<< input embeds:", inputs_embeds.shape)
        return Qwen2ForCausalLM.forward(self, input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds,
                                        labels=labels,
                                        use_cache=use_cache,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict, )

    # overwrite prepare_inputs_labels_for_multimodal
    def prepare_inputs_labels_for_multimodal_with_selector_grads(self, input_ids, position_ids, attention_mask,
                                                                 past_key_values, labels,
                                                                 images, selector_grad_c,
                                                                 modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)

            selector_grad_c = selector_grad_c.unsqueeze(1).unsqueeze(2)
            encoded_image_features = encoded_image_features * selector_grad_c

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for image_feat in encoded_image_features:
                image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

            # if mm_patch_merge_type == "flat":
            #     image_features = [x.flatten(0, 1) for x in image_features]

            if mm_patch_merge_type.startswith("spatial"):
                assert "pad" in image_aspect_ratio
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # assert "unpad" in mm_patch_merge_type and "pad" in image_aspect_ratio, f"{mm_patch_merge_type} {image_aspect_ratio}"
                    image_feature = image_feature[0]
                    height = width = int(math.sqrt(image_feature.shape[0]))
                    # assert height * width == image_feature.shape[0]
                    image_feature = image_feature.view(height, width, -1).permute(2, 0, 1)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(
                        *image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        tune_image_boundaries = getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False)
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                if tune_image_boundaries:
                    cur_input_embeds_1 = cur_input_embeds_1.detach()
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                text_embeds = cur_input_embeds_no_im[i]
                if tune_image_boundaries:
                    # Keep the adjacent image delimiters trainable during adapter pretraining.
                    prefix_len = int(i > 0)
                    suffix_len = int(i < num_images)
                    if text_embeds.shape[0] < prefix_len + suffix_len:
                        raise ValueError("Each image token must be surrounded by image start/end tokens")
                    middle_end = text_embeds.shape[0] - suffix_len
                    text_embeds = torch.cat([
                        text_embeds[:prefix_len],
                        text_embeds[prefix_len:middle_end].detach(),
                        text_embeds[middle_end:],
                    ], dim=0)
                cur_new_input_embeds.append(text_embeds)
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                                                      dtype=cur_new_embed.dtype,
                                                                      device=cur_new_embed.device), cur_new_embed),
                                                         dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed,
                                                          torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                                                      dtype=cur_new_embed.dtype,
                                                                      device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(
                new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


AutoConfig.register("llava_qwen_reid", LlavaQwenReIDConfig)
AutoModelForCausalLM.register(LlavaQwenReIDConfig, LlavaQwenForPersonReID)


class AdaptiveSpatialPool(nn.Module):
    def __init__(self, image_size, pool_ratio):
        super().__init__()
        self.image_size = image_size
        pool_size = math.ceil(image_size / pool_ratio)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, image_features):
        # ori_W = int(math.sqrt(image_features.shape[1] * images.shape[3] // images.shape[2]))
        # ori_H = int(ori_W * images.shape[2] // images.shape[3])

        B, _, F = image_features.shape

        image_features_spatial = image_features.view(B, self.image_size, self.image_size, F).permute(0, 3, 1, 2)
        image_features_spatial_pool = self.pool(image_features_spatial)

        image_features_spatial_pool = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()
        return image_features_spatial_pool

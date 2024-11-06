# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import json
import logging
import os
import re
import pathlib
import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from pycocotools import mask as mask_util
from functools import partial
from copy import deepcopy

import tokenizers
import torch
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ferretui import conversation as conversation_lib
from ferretui.constants import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
    IMAGE_TOKEN_INDEX, DEFAULT_REGION_FEA_TOKEN,
    VOCAB_IMAGE_H, VOCAB_IMAGE_W, GROUNDING_TEMPLATES
)
from ferretui.mm_utils import process_anyres_image, tokenizer_image_token
from ferretui.model import *
from ferretui.train.ferret_trainer import FerretTrainer
from .misc import regulate_box, format_bytes, extract_coors, unfreeze_vit
from .utils.peft import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    get_mm_adapter_state_maybe_zero_3,
    get_vision_tower_state_maybe_zero_3,
    find_all_linear_names
)

from PIL import Image

import torch.distributed as dist

# local_rank = None
local_rank = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
global_rank = int(os.getenv('RANK', '0'))
world_size = int(os.environ["WORLD_SIZE"])

def init_distributed_mode():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank
    )
    print(f"dist.is_initialized(): {dist.is_initialized()}")


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    add_region_feature: bool = False
    region_geo_sampler: bool = False
    sampler_pooler_mode: str = field(default='mean')    # Support 'mean' or 'max'
    no_coor: bool = False


@dataclass
class DataArguments:
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    data_multiple: List[float] = field(default=None, metadata={"help": "Data mutliplier for each dataset when mixed. None means direct concat."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: List[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    resized_image_h: int = 336  #  224
    resized_image_w: int = 336  #  224
    point_input_sample: str = 'segment_mask-uniform'  # 'segment_mask-uniform', 'segment_mask-center', 'segment_mask-gaussian', 'gaussian', 'center'
    refer_previous_point: bool = False
    use_shard_datasets: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_qv_proj_only: bool = False
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    use_safetensors: Optional[bool] = None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN,
                        '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>'
                    )
            replace_token = DEFAULT_IMAGE_TOKEN

            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


""" preprocess """
def preprocess_llama_2(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    对话数据格式化并进行标注，使其可以输入到模型中进行训练，尤其是适用于 “Llama3” 的对话模型。
    这个预处理过程分为几个关键步骤：格式化对话、对话模板填充、分段标记和掩码处理
    """
    # 1. 准备对话模板和角色
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 2. Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 3. Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') 
                for prompt in conversations], 
            dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # 4. Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # 5. 对各轮对话进行 Token 掩码处理
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i > 0:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        # 6. 异常情况警告
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask targets
    sep = "<start_of_turn>" + conv.sep + conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1 # exclude <bos>
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1 # exclude <bos>

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi3(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(
                rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(sources: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + \
            conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
                    for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version.startswith("gemma"):
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_phi3(sources, tokenizer, has_image=has_image)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def extend_list(original_list, multiplier):
    # Calculate how many elements to replicate and how many to select randomly
    replicate_elements = math.floor(multiplier)
    random_elements = multiplier - replicate_elements

    # Replicate the list
    replicated_list = original_list * replicate_elements

    # Calculate how many elements to randomly select
    select_elements = math.ceil(len(original_list) * random_elements)

    # Randomly select elements and append to the replicated list
    for _ in range(select_elements):
        random_element = random.choice(original_list)
        replicated_list.append(random_element)

    return replicated_list


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def shard_data(self, datas, data_path, ori_counts):
        no_shard_per_worker = int(math.ceil(len(datas) / self.world_size))
        datas = datas[no_shard_per_worker*self.global_rank: no_shard_per_worker*(self.global_rank+1)]
        print(f"Shard {data_path}: ori: {ori_counts}, now: {len(datas)}")
        return datas

    def load_pretrain(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "pretrain"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_llava_mixed(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'llava_v1_5_mixed'
            # may contain sharegpt data
            if "image" in data_i:
                data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_git_instruction(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'git_instruction'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_vg_element(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'vg_element'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_llava_grounded(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "llava_grounded"
            data_i["image"] = os.path.join(image_folder, data_i["image"])
            data_i["location_instruction"] = True
        return datas

    def load_flickr(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "flickr_entities"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas

    def load_refexp(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "refexp"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas

    def load_obj365(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "objects365"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas

    def load_sharegpt4v(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'sharegpt4v'
            if "image" in data_i:
                data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_lvisinstruct4v(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'lvisinstruct4v'
            if "image" in data_i:
                data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas

    # plain multimodal data without bbox from LLaVA v1.5
    def load_vqa(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "vqa"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_swit(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "vqa"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_sharegpt(self, data_path):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "sharegpt"
        return datas

    def load_screen2words(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = 'screen2words'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_widgetcaptions(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "widgetcaptions"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_taperception(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "taperception"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_widget_listing(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "widget_listing"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_ocr(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "ocr"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_find_text(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "find_text"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_icon_recognition(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "icon_recognition"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_find_icons(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "find_icons"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_widget_classification(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "widget_classification"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_find_widget(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "find_widget"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_detailed_description(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "detailed_description"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_conversation_perception(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "conversation_perception"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_conversation_interaction(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "conversation_interaction"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas

    def load_function(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "function"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def __init__(self, data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: DataArguments,
                 ):
        super(LazySupervisedDataset, self).__init__()
        data_multiple = data_args.data_multiple
        if not isinstance(data_path, list):
            data_path = [data_path]

        image_folders = data_args.image_folder
        if not isinstance(data_args.image_folder, list):
            image_folders = [data_args.image_folder]

        self.data_args = data_args

        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.global_rank = int(os.getenv('RANK', '0'))

        rank0_print(f"world size: {self.world_size}")
        print(f"global rank: {self.global_rank}")

        list_data_dict = []
        for data_path_i, image_folder_i in zip(data_path, image_folders):
            if 'blip_laion_cc_sbu' in data_path_i:
                rank0_print(f"Loading Pretrain Blip_Laion_CC_SBU data")
                list_data_dict.append(self.load_pretrain(data_path_i, image_folder_i))
            elif 'minigemini' in data_path_i:
                rank0_print(f"Loading Pretrain ALLaVA data")
                list_data_dict.append(self.load_pretrain(data_path_i, image_folder_i))
            elif 'llava_v1_5_mix' in data_path_i:
                rank0_print(f"Loading LLaVA v1.5 mixed data")
                list_data_dict.append(self.load_llava_mixed(data_path_i, image_folder_i))
            elif 'svit_v1_mix' in data_path_i:
                rank0_print(f"Loading SVIT v1 mixed data")
                list_data_dict.append(self.load_llava_mixed(data_path_i, image_folder_i))
            elif 'git_instruction' in data_path_i:
                rank0_print(f"Loading GIT instruct data")
                if data_multiple is None:
                    rank0_print(f"Multiplying GIT instruct by 3 times to make it around 100k")
                    list_data_dict.append(self.load_git_instruction(data_path_i, image_folder_i) * 3)
                else:
                    list_data_dict.append(self.load_git_instruction(data_path_i, image_folder_i))
            elif 'vg_objects' in data_path_i:
                rank0_print(f"Loading VG objects data")
                list_data_dict.append(self.load_vg_element(data_path_i, image_folder_i))
            elif 'vg_relations' in data_path_i:
                rank0_print(f"Loading VG relations data")
                list_data_dict.append(self.load_vg_element(data_path_i, image_folder_i))
            elif 'vg_regions' in data_path_i:
                rank0_print(f"Loading VG regions data")
                list_data_dict.append(self.load_vg_element(data_path_i, image_folder_i))
            elif 'grounded_llava_box' in data_path_i:
                rank0_print(f"Loading LLaVA grounded data")
                list_data_dict.append(self.load_llava_grounded(data_path_i, image_folder_i))
            elif 'flickr' in data_path_i:
                rank0_print(f"Loading Flickr30k entities data")
                list_data_dict.append(self.load_flickr(data_path_i, image_folder_i))
            elif 'refexp' in data_path_i:
                rank0_print(f"Loading Ref expression data")
                list_data_dict.append(self.load_refexp(data_path_i, image_folder_i))
            elif 'objects365' in data_path_i:
                rank0_print(f"Loading O365 data")
                list_data_dict.append(self.load_obj365(data_path_i, image_folder_i))
            elif 'sharegpt4v' in data_path_i.lower():
                rank0_print(f"Loading sharegpt4v data")
                list_data_dict.append(self.load_sharegpt4v(data_path_i, image_folder_i))
            elif 'lvis-instruct4v' in data_path_i.lower():
                rank0_print(f"Loading lvisinstruct4v data")
                list_data_dict.append(self.load_lvisinstruct4v(data_path_i, image_folder_i))

            elif 'okvqa' in data_path_i:
                rank0_print(f"Loading COCO OKVQA data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'vqav2' in data_path_i:
                rank0_print(f"Loading COCO VQAv2 data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'ocr_vqa' in data_path_i:
                rank0_print(f"Loading OCRVQA data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'textvqa_textcaps' in data_path_i:
                rank0_print(f"Loading TextVQA TextCaps data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'gqa_vqa' in data_path_i:
                rank0_print(f"Loading GQA data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'svit_v1' in data_path_i:
                rank0_print(f"Loading SWIT complex data")
                list_data_dict.append(self.load_swit(data_path_i, image_folder_i))
            elif 'sharegpt' in data_path_i or 'wizardlm' in data_path_i:
                rank0_print(f"Loading ShareGPT/WizardLM text only data")
                list_data_dict.append(self.load_sharegpt(data_path_i))
            elif 'screen2words' in data_path_i:
                logging.warning(f"Loading screen2words data")
                list_data_dict.append(self.load_screen2words(data_path_i, image_folder_i))
            elif 'widgetcaptions' in data_path_i:
                logging.warning(f"Loading widgetcaptions data")
                list_data_dict.append(self.load_widgetcaptions(data_path_i, image_folder_i))
            elif 'taperception' in data_path_i:
                logging.warning(f"Loading taperception data")
                list_data_dict.append(self.load_taperception(data_path_i, image_folder_i))
            elif 'widget_listing' in data_path_i:
                logging.warning(f"Loading widget_listing data")
                list_data_dict.append(self.load_widget_listing(data_path_i, image_folder_i))
            elif 'ocr' in data_path_i:
                logging.warning(f"Loading ocr data")
                list_data_dict.append(self.load_ocr(data_path_i, image_folder_i))
            elif 'find_text' in data_path_i:
                logging.warning(f"Loading find_text data")
                list_data_dict.append(self.load_find_text(data_path_i, image_folder_i))
            elif 'icon_recognition' in data_path_i:
                logging.warning(f"Loading icon_recognition data")
                list_data_dict.append(self.load_icon_recognition(data_path_i, image_folder_i))
            elif 'find_icons' in data_path_i:
                logging.warning(f"Loading find_icons data")
                list_data_dict.append(self.load_find_icons(data_path_i, image_folder_i))
            elif 'widget_classification' in data_path_i:
                logging.warning(f"Loading widget_classification data")
                list_data_dict.append(self.load_widget_classification(data_path_i, image_folder_i))
            elif 'find_widget' in data_path_i:
                logging.warning(f"Loading find_widget data")
                list_data_dict.append(self.load_find_widget(data_path_i, image_folder_i))
            elif 'detailed_description' in data_path_i:
                logging.warning(f"Loading detailed_description data")
                list_data_dict.append(self.load_detailed_description(data_path_i, image_folder_i))
            elif 'conversation_perception' in data_path_i:
                logging.warning(f"Loading conversation_perception data")
                list_data_dict.append(self.load_conversation_perception(data_path_i, image_folder_i))
            elif 'conversation_interaction' in data_path_i:
                logging.warning(f"Loading conversation_interaction data")
                list_data_dict.append(self.load_conversation_interaction(data_path_i, image_folder_i))
            elif 'function' in data_path_i:
                logging.warning(f"Loading function data")
                list_data_dict.append(self.load_function(data_path_i, image_folder_i))
            else:
                rank0_print("Loading {} not supported".format(data_path_i))

        if data_multiple is None:
            # Concat all data directly and shuffle.
            list_data_dict = [item for dataset_i in list_data_dict for item in dataset_i]
            random.shuffle(list_data_dict)
        else:
            new_list_data_dict = []
            for data_scaler_i, dataset_i in zip(data_multiple, list_data_dict):
                dataset_name_i = dataset_i[0]['dataset']
                rank0_print(f"Multiplying {dataset_name_i} by {data_scaler_i} times")
                new_dataset_i = extend_list(dataset_i, data_scaler_i)
                new_list_data_dict.extend(new_dataset_i)
            list_data_dict = new_list_data_dict
            random.shuffle(list_data_dict)

        print(f"R{self.global_rank} number of samples: {len(list_data_dict)}")
        rank0_print(f"The total training set contains {len(list_data_dict)} samples.")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.model_args = model_args
        self.point_input_sample = self.data_args.point_input_sample
        self.add_region_feature = self.model_args.add_region_feature
        self.no_coor = self.model_args.no_coor
        self.refer_previous_point = self.data_args.refer_previous_point

        if self.data_args.use_shard_datasets:
            self.sync_iter_counts()

    def __len__(self):
        return len(self.list_data_dict)

    def sync_iter_counts(self):
        # Sync the total sample counts on each worker
        # Calculate the number of samples for this worker
        num_samples = len(self)
        rank0_print(f"sync iter counts num_samples: {num_samples}")
        # Gather the number of samples from all workers
        min_num_samples_tensor = torch.tensor(num_samples, dtype=torch.int64).cuda()
        dist.all_reduce(min_num_samples_tensor, op=dist.ReduceOp.MIN)
        min_num_samples = min_num_samples_tensor.item()
        print(f"min_num_sample: {min_num_samples}")
        # Create a subset of the dataset based on min_num_samples
        # indices = list(range(num_samples))
        # np.random.shuffle(indices)
        # subset_indices = indices[:min_num_samples]
        self.list_data_dict = self.list_data_dict[:min_num_samples]
        print(f"[R{self.global_rank}] sync_iter_counts at {os.path.basename(__file__)}: ori dataset counts: {num_samples}, now mmx list dataset len: {len(self.list_data_dict)}.")

    def get_obj_center(self, box, ratio_w, ratio_h, std_dev_weight=0.15):
        box_center_w = ratio_w * (box[0]+box[2])/2.0
        box_center_h = ratio_h * (box[1]+box[3])/2.0

        box_min_w = ratio_w * box[0]
        box_max_w = ratio_w * box[2]

        box_min_h = ratio_h * box[1]
        box_max_h = ratio_h * box[3]

        # Set std of gaussian sampling, 68%/95% is sampled within +- 1/2 std_dev.
        gau_std_w = (box_max_w - box_min_w)*std_dev_weight
        gau_std_h = (box_max_h - box_min_h)*std_dev_weight

        def sample_float_within_range(mean, std_dev, min_val, max_val):
            while True:
                x = random.gauss(mean[0], std_dev[0])
                y = random.gauss(mean[1], std_dev[1])
                if min_val[0] <= x <= max_val[0] and min_val[1] <= y <= max_val[1]:
                    return x, y

        jit_x, jit_y = sample_float_within_range(
            mean=[box_center_w, box_center_h],
            std_dev=[gau_std_w, gau_std_h],
            min_val=[box_min_w, box_min_h],
            max_val=[box_max_w, box_max_h]
        )

        return jit_x, jit_y

    def sample_point_in_segment(self, mask, ratio_w, ratio_h, box=None, sampling='uniform'):
        mask['counts'] = mask['counts'].encode('ascii')
        bin_mask = mask_util.decode(mask)
        # Get the indices of True elements in the mask
        # Note here the size of bin_mask is [h, w].
        indices = np.transpose(np.nonzero(bin_mask))
        if sampling == 'center' or sampling == 'gaussian':
            if sampling == 'center':
                box_anchor_w = int((box[0]+box[2])/2.0)
                box_anchor_h = int((box[1]+box[3])/2.0)
            elif sampling == 'gaussian':
                # Sample a point based on centered gaussian distribution. ratio_w and ratio_h is set to 1 to keep original wh.
                box_anchor_w, box_anchor_h = self.get_obj_center(box, 1, 1, std_dev_weight=0.15)
            # get 1000 random items from the indices
            sampled_list = random.sample(list(range(len(indices))), min(1000, len(indices)))
            min_dis = 1e6
            min_point = None
            for sampled_i in sampled_list:
                point_i = indices[sampled_i]
                dis_i = (point_i[0] - box_anchor_h)**2 + (point_i[1] - box_anchor_w)**2
                if dis_i <= min_dis or min_point is None:
                    min_dis = dis_i
                    min_point = point_i
            point = min_point
        elif sampling == 'uniform':
            # Randomly select an index
            random_index = np.random.choice(len(indices))
            # Get the randomly selected point
            point = indices[random_index]
        else:
            raise NotImplementedError(f'Not support {sampling}.')
        # Note here point is in original image size and its order is [h, w].
        cor_x = point[1] * ratio_w
        cor_y = point[0] * ratio_h
        return cor_x, cor_y

    def get_bbox_coor(self, box, ratio_w, ratio_h):
        return box[0] * ratio_w, box[1] * ratio_h, box[2] * ratio_w, box[3] * ratio_h

    def generate_mask_for_feature(self, coor, box, mask, raw_w, raw_h):
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        # Build SAM mask
        if mask is not None:
            mask['counts'] = mask['counts'].encode('ascii')
            sam_mask = mask_util.decode(mask)
            # Note [h, w] -> [w, h].
            sam_mask = np.transpose(sam_mask)
        else:
            sam_mask = None
        # Build box mask
        box_mask = np.zeros((raw_w, raw_h))
        box_mask[box[0]:box[2]+1, box[1]:box[3]+1] = 1

        coor_mask = np.zeros((raw_w, raw_h))
        # Assume it samples a point.
        if len(coor) == 2:
            # Define window size
            span = 5
            # Make sure the window does not exceed array bounds
            x_min = max(0, coor[0] - span)
            x_max = min(raw_w, coor[0] + span + 1)
            y_min = max(0, coor[1] - span)
            y_max = min(raw_h, coor[1] + span + 1)
            coor_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
            # SAM mask might be out of bounding box, so don't use sam_mask * box_mask.
            coor_mask = coor_mask * sam_mask if sam_mask is not None else coor_mask * box_mask
            assert (coor_mask==1).any(), f"coor: {coor}, box: {box}, raw_w: {raw_w}, raw_h: {raw_h}"
        elif len(coor) == 4:
            coor_mask = box_mask * sam_mask if sam_mask is not None else box_mask
            if (coor_mask==0).all():
                rank0_print('Find one sample sam mask and box has no overlap, use box mask only')
                coor_mask = box_mask
            assert (coor_mask==1).any(), f"coor: {coor}, box: {box}, raw_w: {raw_w}, raw_h: {raw_h}"
        else:
            raise NotImplementedError('Coordinates must be 2d or 4d.')
        coor_mask = torch.from_numpy(coor_mask)
        assert len(coor_mask.nonzero()) != 0

        return coor_mask

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @staticmethod
    def format_unicode_filenames(filename):
        # replace unicode characters
        # i.e. wikiart/images/arnold-b#U00e3#U00b6cklin_soldiers-amount-towards-a-mountain-fortress.jpg
        #   -> wikiart/images/arnold-bã¶cklin_soldiers-amount-towards-a-mountain-fortress.jpg
            return re.subn(r"(#U[0-9a-f]{4})", lambda cp: chr(int(cp.groups()[0][2:], 16)), filename)[0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = deepcopy(self.list_data_dict[i])
        cache_region_masks = []
        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if not os.path.isfile(image_file):
                image_file = self.format_unicode_filenames(image_file)
                possible_exts = ['.gif', '.jpg', '.jpeg', '.png']
                for ext_ in possible_exts:
                    filename_ = os.path.splitext(image_file)[0]
                    if os.path.isfile(filename_ + ext_):
                        image_file = filename_ + ext_
                        break
            image = Image.open(image_file).convert('RGB')
            image_size = image.size
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255)
                                      for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] #(640, 480)

            elif self.data_args.image_aspect_ratio == 'square_nocrop':
                resized_image_h = self.data_args.resized_image_h
                resized_image_w = self.data_args.resized_image_w
                image_size = image.size
                image = processor.preprocess(
                    image,
                    return_tensors='pt',
                    do_resize=True,
                    do_center_crop=False,
                    size=[resized_image_h, resized_image_w]
                )['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == 'anyres':
                resized_image_h = self.data_args.resized_image_h
                resized_image_w = self.data_args.resized_image_w
                image_size = image.size
                image_process_func = partial(
                    processor.preprocess,
                    return_tensors='pt',
                    do_resize=True,
                    do_center_crop=False,
                    size=[resized_image_h, resized_image_w]
                )
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, \
                                             image_process_func=image_process_func) # torch.Size([5, 3, 336, 336])
                # image_size = image.size
                # image = process_anyres_image(       # torch.Size([5, 3, 336, 336])
                #     image, processor, self.data_args.image_grid_pinpoints)
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # Process Locations/Coordinations.
            if 'location_instruction' in sources[0]:
                assert sources[0]['dataset'] in ['git_instruction', 'vg_element', 'llava_grounded',\
                                                 'refexp', 'flickr_entities', 'objects365', 'widgetcaptions', 'taperception', \
                                                 'widget_listing', 'ocr', 'find_text', 'icon_recognition', 'find_icons', \
                                                 'widget_classification',  'find_widget', 'conversation_interaction']
                ratio_w = VOCAB_IMAGE_W * 1.0 / sources[0]['image_w']
                ratio_h = VOCAB_IMAGE_H * 1.0 / sources[0]['image_h']
                conversation = deepcopy(sources[0]['conversations'])
                assert len(sources) == 1

                # add GROUNDING_PROMPT to grounding dataset at its first human round
                if sources[0]['dataset'] in ['llava_grounded', 'refexp', 'flickr_entities', 'objects365']:
                    # conversation[0]['value'] = conversation[0]['value'] + random.choice(GROUNDING_TEMPLATES)
                    conversation[0]['value'] = conversation[0]['value'] + '\nProvide the bounding boxes of the mentioned objects.'

                for box_list_idx, box_list_i in enumerate(sources[0]['box_x1y1x2y2']):
                    # For human input, always build a cache to save sampled point in this round of human input.
                    if box_list_idx % 2 == 0 and self.refer_previous_point:
                        point_box_cache = {}

                    # Replace location placeholders with points or boxes.
                    if len(box_list_i) == 0:
                        # No location mentioned in this round of conversation.
                        continue

                    if box_list_idx % 2 == 0:
                        # Randomly choose point or box as coordination format in human round.
                        location_format = random.choice(['point', 'box'])
                    else:
                        # Always output box in model reponse.
                        location_format = 'box'
                    cur_conv = conversation[box_list_idx]['value']
                    # Iteratively replace <bbox_location> in current conv with real box/point coordinate.
                    for box_idx, box_i in enumerate(box_list_i):
                        box_i = regulate_box(box_i, sources[0]['image_w'], sources[0]['image_h'])
                        if location_format == 'box':
                            # If this box is mentioned in last human input, use the same coordinates as human mentioned.
                            if 'point_box_cache' in locals() and tuple(box_i) in point_box_cache:
                                raw_coor_i = point_box_cache[tuple(box_i)]
                                coor_i = f'[{int(raw_coor_i[0])}, {int(raw_coor_i[1])}]'
                            else:
                                raw_coor_i = self.get_bbox_coor(box=box_i, ratio_w=ratio_w, ratio_h=ratio_h)
                                coor_i = f'[{int(raw_coor_i[0])}, {int(raw_coor_i[1])}, {int(raw_coor_i[2])}, {int(raw_coor_i[3])}]'
                        elif location_format == 'point':
                            # Assert it's human input.
                            assert box_list_idx % 2 == 0
                            # If this box is mentioned previously in this round of human input, use the same coordinates as previously mentioned.
                            if 'point_box_cache' in locals() and tuple(box_i) in point_box_cache:
                                raw_coor_i = point_box_cache[tuple(box_i)]
                            else:
                                if 'segment_mask' in self.point_input_sample:
                                    if 'masks' in sources[0]:
                                        cur_mask = deepcopy(sources[0]['masks'][box_list_idx][box_idx])
                                        assert cur_mask['size'][0] == sources[0]['image_h']
                                        assert cur_mask['size'][1] == sources[0]['image_w']
                                        if 'uniform' in self.point_input_sample.split('-')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h)
                                        elif 'center' in self.point_input_sample.split('-')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h, box=box_i, sampling='center')
                                        elif 'gaussian' in self.point_input_sample.split('-')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h, box=box_i, sampling='gaussian')
                                    else:
                                        # Not all data have/need segment masks.
                                        obj_center_x, obj_center_y = self.get_obj_center(box=box_i, ratio_w=ratio_w, ratio_h=ratio_h, std_dev_weight=0.15)
                                elif self.point_input_sample == 'gaussian':
                                    obj_center_x, obj_center_y = self.get_obj_center(box=box_i, ratio_w=ratio_w, ratio_h=ratio_h, std_dev_weight=0.15)
                                elif self.point_input_sample == 'center':
                                    obj_center_x = ratio_w * (box_i[0]+box_i[2])/2.0
                                    obj_center_y = ratio_h * (box_i[1]+box_i[3])/2.0
                                else:
                                    raise NotImplementedError(f'Not support {self.point_input_sample} in data sampling')
                                raw_coor_i = [obj_center_x, obj_center_y]
                                if 'point_box_cache' in locals() and self.refer_previous_point:
                                    point_box_cache[tuple(box_i)] = raw_coor_i
                            coor_i = f'[{int(raw_coor_i[0])}, {int(raw_coor_i[1])}]'
                        assert f'<bbox_location{box_idx}>' in cur_conv, f"String '<bbox_location{box_idx}>' not found in {cur_conv}"
                        if self.add_region_feature and box_list_idx % 2 == 0:
                            if self.no_coor:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', f'{DEFAULT_REGION_FEA_TOKEN}')
                            else:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', coor_i + f' {DEFAULT_REGION_FEA_TOKEN}')
                            cur_box = box_i
                            cur_mask = deepcopy(sources[0]['masks'][box_list_idx][box_idx]) if 'masks' in sources[0] else None
                            ori_size_raw_coor_i = [
                                raw_coor_i[0]/ratio_w,
                                raw_coor_i[1]/ratio_h,
                                raw_coor_i[2]/ratio_w,
                                raw_coor_i[3]/ratio_h
                            ] if len(raw_coor_i) == 4 \
                                else [raw_coor_i[0]/ratio_w, raw_coor_i[1]/ratio_h]

                            cur_region_mask = self.generate_mask_for_feature(
                                ori_size_raw_coor_i, cur_box, cur_mask, raw_w=sources[0]['image_w'], raw_h=sources[0]['image_h'])
                            cache_region_masks.append(cur_region_mask)
                            # print('cur_conv:', cur_conv)
                            # print('cur_region_mask:', cur_region_mask.nonzero())
                            # raise NotImplementedError()
                            # pdb.set_trace()
                        else:
                            if self.no_coor:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', '')
                            else:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', coor_i)
                    # Assign this round of conv back.
                    conversation[box_list_idx]['value'] = cur_conv
                sources[0]['conversations'] = conversation
                # print(conversation)
                # exit(0)

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        
        if self.add_region_feature:
            data_dict['region_masks'] = cache_region_masks
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes

        if 'region_masks' in instances[0]:
            region_masks = [instance['region_masks'] for instance in instances]
            batch['region_masks'] = region_masks

        return batch


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        model_args=model_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

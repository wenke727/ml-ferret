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
import os
import pathlib
from typing import Dict

import wandb # type: ignore
import torch
import torch.distributed as dist
import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel

from ferretui import conversation as conversation_lib
from ferretui.model import *
from ferretui.train.ferret_trainer import FerretTrainer
from .utils.peft import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    get_mm_adapter_state_maybe_zero_3,
    find_all_linear_names
)
from .data_processing import DataArguments, make_supervised_data_module
from .misc import format_bytes, unfreeze_vit, rank0_print
from .arguments import ModelArguments, TrainingArguments

wandb.init(mode='offline')

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


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
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


def train(attn_implementation=None):
    """
    参数解析 -> 模型量化配置 -> 初始化模型 -> 配置冻结和微调 -> 分布式和量化设置 -> 加载数据和训练器 -> 启动训练 -> 保存模型
    """
    global local_rank

    # 1.参数解析与预处理
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.resized_image_w = data_args.resized_image_w
    model_args.resized_image_h = data_args.resized_image_h

    if model_args.no_coor:
        assert model_args.add_region_feature

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 2. 模型量化与 dtype 配置：
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        # 启用 BitsAndBytesConfig 配置，从而在内存和计算效率上优化模型
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    model_max_length_args = {}
    if 'llava-v1.6-8b' not in model_args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True)
        if config.max_position_embeddings < training_args.model_max_length:
            rank0_print(
                f'Set the max_position_embeddings from {config.max_position_embeddings} to {training_args.model_max_length}')
            model_max_length_args.update({'max_position_embeddings': training_args.model_max_length})

    # 3. 初始化模型： 根据是否包含视觉模型的 vision_tower 字段，判断是加载多模态还是单模态的模型
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = FerretMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,

                **bnb_model_from_pretrained_args
            )
        elif "gemma" in model_args.model_name_or_path:
            model = FerretGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                use_safetensors=training_args.use_safetensors,
                **bnb_model_from_pretrained_args
            )
        elif "phi3" in model_args.model_name_or_path:
            model = LlavaPhiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = FerretLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                use_safetensors=training_args.use_safetensors,
                **bnb_model_from_pretrained_args,
                **model_max_length_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    # 4. 冻结和微调配置
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 5. 配置分布式、量化和 LoRA 训练
    # 5.1 梯度检查点配置
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 5.2. LoRA 微调设置
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args.lora_qv_proj_only),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # 5.3. 加载 tokenizer
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # 5.4. 设置 tokenizer 特殊标记
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="[PAD]"), tokenizer=tokenizer, model=model)
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif "gemma" in model_args.version:
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["gemma"]
    else:
        if tokenizer.pad_token is None:
            rank0_print("Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="<pad>"), tokenizer=tokenizer, model=model)

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 6. 加载与配置分布式模型
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
            add_region_feature=model_args.add_region_feature,
            region_geo_sampler=model_args.region_geo_sampler,
            sampler_pooler_mode=model_args.sampler_pooler_mode,
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [[g[0]*base_size, g[1]*base_size] for g in grids]

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        if training_args.unfreeze_mm_vision_tower:
            lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            assert lr_of_vit > 0.0 and lr_of_mlp > 0.0
            training_args.mm_projector_lr = lr_of_mlp
            unfreeze_vit(vision_tower)
            rank0_print(
                f'Tune the entire model! The LR of ViT is {lr_of_vit}. The LR of MLP is {lr_of_mlp}. The LR of LLM is {training_args.learning_rate}')

        if model_args.add_region_feature:
            if model_args.region_geo_sampler:
                for p in model.get_model().region_geo_sampler.parameters():
                    p.requires_grad = True
            else:
                for p in model.get_model().region_fea_adapter.parameters():
                    p.requires_grad = True

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        rank0_print(f"Total parameters: {format_bytes(total_params)}")
        rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.mm_patch_merge_type = model_args.mm_patch_merge_type
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer, add_region_feature=model_args.add_region_feature)
        model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 7. 准备数据模块与训练器
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    trainer = FerretTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # 8. 启动训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # 9. 保存模型
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    init_distributed_mode()
    train()

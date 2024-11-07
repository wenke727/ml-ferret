from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from transformers import PreTrainedTokenizer

from ferretui.constants import IGNORE_INDEX

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")    # 基础语言模型的名称或路径
    version: Optional[str] = field(default="v0")                              # 模型版本号
    freeze_backbone: bool = field(default=False)                              # 是否冻结主干网络参数
    tune_mm_mlp_adapter: bool = field(default=False)                         # 是否微调多模态 MLP 适配器
    vision_tower: Optional[str] = field(default=None)                        # 视觉编码器的名称或路径
    mm_vision_select_layer: Optional[int] = field(default=-1)                # 选择视觉特征的层数，-1表示最后一层
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)             # 预训练的多模态 MLP 适配器路径
    mm_projector_type: Optional[str] = field(default='linear')               # 多模态投影器类型
    mm_use_im_start_end: bool = field(default=False)                        # 是否使用图像序列的起始和结束标记
    mm_use_im_patch_token: bool = field(default=True)                       # 是否在序列中使用图像块标记
    mm_patch_merge_type: Optional[str] = field(default='flat')              # 图像块特征的合并方式
    mm_vision_select_feature: Optional[str] = field(default="patch")         # 视觉特征的选择类型
    add_region_feature: bool = False                                         # 是否添加区域特征
    region_geo_sampler: bool = False                                        # 是否使用区域几何采样器
    sampler_pooler_mode: str = field(default='mean')                        # 采样器的池化模式
    no_coor: bool = False                                                   # 是否禁用坐标信息

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
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        }
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        }
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
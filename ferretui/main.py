#%%
import os

# todo cuda visible
import re
import json
import math
import argparse
from copy import deepcopy
from functools import partial

import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from loguru import logger

from ferretui.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_REGION_FEA_TOKEN, VOCAB_IMAGE_W, VOCAB_IMAGE_H # type: ignore
from ferretui.conversation import conv_templates, SeparatorStyle # type: ignore
from ferretui.eval.model_UI import UIData, get_chunk
from ferretui.model.builder import load_pretrained_model # type: ignore
from ferretui.utils import disable_torch_init # type: ignore
from ferretui.mm_utils import tokenizer_image_token, process_images # type: ignore



class FerretUI:
    def __init__(self, args):
        # 禁用 PyTorch 初始化以节省内存
        self.args = args
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = self.get_model_name_from_path(model_path)

        # 加载预训练模型、分词器和图像处理器
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(model_path, args.model_base, model_name, use_safetensors=True)

    def get_model_name_from_path(self, model_path):
        if 'gemma' in model_path:
            return 'ferret_gemma'
        elif 'llama' or 'vicuna' in model_path:
            return 'ferret_llama'
        else:
            raise ValueError(f"No model matched for {model_path}")

    def generate(self, img: Image.Image, ann: dict, image_size:list=None, verbose:bool=False):
        args = self.args
        qs = ann["question"]
        cur_prompt = qs

        if image_size is None:
            image_size = img.size

        # 处理问题中的图像标记
        if "<image>" in qs:
            qs = qs.split('\n')[1]

        if self.model.config.mm_use_im_start_end: # False
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # 创建对话模板
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # TODO multiturn conversation
        if verbose:
            logger.debug(f"prompt: {prompt}")

        # 对输入进行编码
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).cuda()

        # 处理图像
        if self.model.config.image_aspect_ratio == "square_nocrop":
            image_tensor = self.image_processor.preprocess(
                img,
                return_tensors='pt',
                do_resize=True,
                do_center_crop=False,
                size=[args.image_h, args.image_w]
            )['pixel_values'][0]
        elif self.model.config.image_aspect_ratio == "anyres":
            image_process_func = partial(
                self.image_processor.preprocess,
                return_tensors='pt',
                do_resize=True,
                do_center_crop=False,
                size=[args.image_h, args.image_w]
            )
            image_tensor = process_images(
                [img],
                self.image_processor,
                self.model.config,
                image_process_func=image_process_func
            )[0]
        else:
            image_tensor = process_images([img], self.image_processor, self.model.config)[0]

        images = image_tensor.unsqueeze(0).to(args.data_type).cuda()

        # 处理区域掩码
        region_masks = ann['region_masks']
        if region_masks is not None:
            region_masks = [[region_mask_i.cuda().half() for region_mask_i in region_masks]]
        else:
            region_masks = None

        # 模型推理
        with torch.inference_mode():
            self.model.orig_forward = self.model.forward
            self.model.forward = partial(
                self.model.orig_forward,
                region_masks=region_masks
            )
            output_ids = self.model.generate(
                input_ids,
                images=images,
                region_masks=region_masks,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
            self.model.forward = self.model.orig_forward

        # 解码输出
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if verbose:
            logger.debug(f"outputs: {outputs}")

        return outputs, cur_prompt


def eval_model(args):
    # 数据集
    dataset = UIData(data_path=args.data_path, image_path=args.image_path, args=args)
    data_ids = dataset.ids

    # 初始化模型包装器
    model = FerretUI(args)

    chunk_data_ids = get_chunk(data_ids, args.num_chunks, args.chunk_idx)
    answers_folder = os.path.expanduser(args.answers_file)
    os.makedirs(answers_folder, exist_ok=True)
    answers_file = os.path.join(answers_folder, f'{args.chunk_idx}_of_{args.num_chunks}.jsonl')
    ans_file = open(answers_file, "w")

    for i, id in enumerate(tqdm(chunk_data_ids)):
        img, ann, image_size = dataset[id]

        # 使用模型生成输出
        outputs, cur_prompt = model.generate(img, ann, image_size)

        # 获取标签（如果有）
        if 'label' in ann:
            label = ann['label']
        elif len(ann['conversations']) > 1:
            label = ann['conversations'][1]['value']
        else:
            label = None

        # 将结果写入答案文件
        ans_file.write(json.dumps({
            "id": ann['id'],
            "image_path": ann['image'],
            "prompt": cur_prompt,
            "text": outputs,
            "label": label,
        }) + "\n")
        ans_file.flush()
    ans_file.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./ckpt/llama8b-anyres")
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="./playground/images")
    parser.add_argument("--data_path", type=str, default="./playground/sample_data/eval_data_example_1_no_box_in.json")
    parser.add_argument("--answers_file", type=str, default="./eval_output/data_no_box_in_eval.jsonl")
    parser.add_argument("--conv_mode", type=str, default="ferret_llama_3", help="[ferret_gemma_instruct,ferret_llama_3,ferret_vicuna_v1]")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--region_format", type=str, default="point", choices=["point", "box", "segment", "free_shape"])
    parser.add_argument("--no_coor", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--data_type", type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    args = parser.parse_args()

    if args.data_type == 'fp16':
        args.data_type = torch.float16
    elif args.data_type == 'bf16':
        args.data_type = torch.bfloat16
    else:
        args.data_type = torch.float32

    return args


if __name__ == "__main__":
    args = get_args()
    args.add_region_feature = True
    args.region_format = True
    # eval_model(args)

    # 输入格式
    ann = {
        "id": 1,
        "image": "appstore_reminders.png",
        "image_h": 2532,
        "image_w": 1170,
        "conversations": [{
            "from": "human",
            "value": "<image>\nIdentify the icon type of the widget <bbox_location0>?"
        }],
        "box_x1y1x2y2": [[[1035, 540, 1114, 622]]],
        # "question": "Identify the icon type of the widget [884, 213, 952, 245]?",
        "question": "widget listing",
        "region_masks": None
    }

    fn = './playground/images/appstore_reminders.png'
    img = Image.open(fn).convert("RGB")

    model = FerretUI(args)
    output, cur_prompt = model.generate(img, ann, verbose=True)


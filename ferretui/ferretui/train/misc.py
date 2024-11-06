import os
import re
import torch

local_rank = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
global_rank = int(os.getenv('RANK', '0'))
world_size = int(os.environ["WORLD_SIZE"])


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def format_bytes(size):
    billion = 10**9
    million = 10**6

    if size >= billion:
        return f"{size / billion:.2f}B"
    elif size >= million:
        return f"{size / million:.2f}M"
    else:
        return f"{size} bytes"

def regulate_box(box, img_w, img_h):
    return [
        max(0, min(box[0], img_w-1)),
        max(0, min(box[1], img_h-1)),
        max(0, min(box[2], img_w-1)),
        max(0, min(box[3], img_h-1))
    ]

def extract_coors(s):
    # Regex pattern to match brackets content
    brackets_pattern = r'\[(.*?)\]'

    # Regex pattern to match values
    values_pattern = r'=\s*([^,\]]+)'

    # Find all bracketed strings
    brackets = re.findall(brackets_pattern, s)

    # Define a list to hold the list of values
    values_list = []

    # Extract values from each bracketed string
    for bracket in brackets:
        # Find all matches in the string
        matches = re.findall(values_pattern, bracket)
        # Convert matches to integers and add to values_list
        values_list.append([int(match) for match in matches])

    return values_list


def unfreeze_vit(vision_tower):
    for _, p in vision_tower.named_parameters():
        p.requires_grad = True

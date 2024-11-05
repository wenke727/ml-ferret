import pandas as pd
import ast
import cv2

def get_image_size(img_fn):
    """
    从图像文件中获取宽度和高度。

    参数:
    - img_fn (str): 图像文件路径

    返回:
    - tuple: (height, width) 图像的高度和宽度
    """
    img = cv2.imread(img_fn)
    if img is None:
        raise ValueError(f"无法读取图像文件: {img_fn}")
    height, width = img.shape[:2]
    return height, width

def normalize_single_bbox(bbox, height, width):
    scale_x, scale_y = 1000 / width, 1000 / height
    return [int(coord * scale_x) if i % 2 == 0 else int(coord * scale_y) for i, coord in enumerate(bbox)]

def denormalize_single_bbox(bbox, height, width):
    scale_x, scale_y = width / 1000, height / 1000
    return [int(coord * scale_x) if i % 2 == 0 else int(coord * scale_y) for i, coord in enumerate(bbox)]

def normalize_bbox(df, height=None, width=None, img_fn=None):
    if img_fn:
        height, width = get_image_size(img_fn)

    if not (height and width):
        raise ValueError("必须提供图像的 height 和 width。")

    df['bbox'] = df['bbox'].apply(lambda bbox: normalize_single_bbox(bbox if isinstance(bbox, list) else ast.literal_eval(bbox), height, width))
    return df

def denormalize_bbox(df, height=None, width=None, img_fn=None):
    if img_fn:
        height, width = get_image_size(img_fn)

    if not (height and width):
        raise ValueError("必须提供图像的 height 和 width。")

    df['bbox'] = df['bbox'].apply(
        lambda bbox: denormalize_single_bbox(
            bbox if isinstance(bbox, list) else ast.literal_eval(bbox), height, width)
    )

    return df


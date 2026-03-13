#!/usr/bin/env python3
"""
将分割 mask 图像降采样并整理为 (N, H, W, 1) 格式的 .npy 文件。

输入: 每张图像对应一个二值 mask PNG（0=背景, 255=前景）
输出: masks.npy，shape=(N, H, W, 1)，dtype=float32，值为 0.0 或 1.0

用法:
    python tools/prepare_seg_masks.py \
        --mask_dir lab9_data/batch_converSeg_outputs_heatthing \
        --image_dir lab9_data/images \
        --output    lab9_data/seg_masks/masks.npy \
        --height 336 --width 504
"""

import argparse
import os
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir",  required=True, help="包含 *_mask.png 的目录")
    parser.add_argument("--image_dir", required=True, help="训练图像目录，用于确定帧顺序")
    parser.add_argument("--output",    required=True, help="输出 .npy 文件路径")
    parser.add_argument("--height",    type=int, default=336)
    parser.add_argument("--width",     type=int, default=504)
    args = parser.parse_args()

    # 按训练图像的排序顺序确定帧顺序
    image_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_names = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(args.image_dir)
        if f.endswith(image_exts)
    ])
    print(f"Found {len(image_names)} images: {image_names}")

    masks = []
    for name in image_names:
        mask_path = os.path.join(args.mask_dir, f"{name}_mask.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask not found for image '{name}': expected {mask_path}"
            )
        mask = Image.open(mask_path).convert("L")  # grayscale
        mask = mask.resize((args.width, args.height), Image.NEAREST)
        arr = (np.array(mask) > 127).astype(np.float32)  # (H, W), 0.0 or 1.0
        masks.append(arr)

    masks = np.stack(masks, axis=0)          # (N, H, W)
    masks = masks[..., np.newaxis]           # (N, H, W, 1)
    print(f"Output shape: {masks.shape}")
    print(f"Foreground ratio: {masks.mean():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, masks)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

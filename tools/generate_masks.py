#!/usr/bin/env python3
"""
DDGS 预处理 Mask 生成脚本
用于 DAFE (Distance-Aware Fidelity Enhancement) 机制

用法:
    python tools/generate_masks.py --data_path <数据目录> --mask_param <阈值> --resolution <分辨率>

示例:
    python tools/generate_masks.py --data_path lab9_data --mask_param 10 --resolution 1
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import glob


def generate_far_mask(depth_path, mask_param, target_size=None):
    """
    从深度图生成远距离区域的 mask

    Args:
        depth_path: 深度图路径 (支持逆深度图)
        mask_param: 阈值参数，表示远距离区域的百分比
                    例如 mask_param=10 表示最远的 10% 区域
        target_size: 目标尺寸 (width, height)，用于缩放

    Returns:
        far_mask: torch.Tensor (H, W)，远距离区域为 1，其他为 0
    """
    # 读取深度图 (假设是 16-bit 逆深度图)
    depth_img = Image.open(depth_path)
    inv_depth = np.array(depth_img, dtype=np.float32)

    # 归一化
    if inv_depth.max() > 1:
        inv_depth = inv_depth / 65535.0

    # 逆深度转深度 (避免除零)
    # 逆深度小的地方 = 实际深度大 = 远距离
    valid_mask = inv_depth > 1e-6
    depth = np.zeros_like(inv_depth)
    depth[valid_mask] = 1.0 / inv_depth[valid_mask]
    depth[~valid_mask] = depth[valid_mask].max() if valid_mask.any() else 1.0

    # 计算远距离阈值 (最远的 mask_param% 区域)
    threshold = np.percentile(depth[valid_mask], 100 - mask_param)

    # 生成 mask: 远距离区域为 1
    far_mask = (depth >= threshold).astype(np.float32)

    # 如果需要缩放
    if target_size is not None:
        far_mask_pil = Image.fromarray((far_mask * 255).astype(np.uint8))
        far_mask_pil = far_mask_pil.resize(target_size, Image.NEAREST)
        far_mask = np.array(far_mask_pil).astype(np.float32) / 255.0

    return torch.from_numpy(far_mask)


def main():
    parser = argparse.ArgumentParser(description='DDGS 预处理 Mask 生成脚本')
    parser.add_argument('--data_path', type=str, required=True, help='数据目录路径')
    parser.add_argument('--mask_param', type=int, default=10, help='远距离阈值百分比 (默认: 10)')
    parser.add_argument('--resolution', type=int, default=1, help='分辨率因子 (默认: 1)')
    parser.add_argument('--depth_dir', type=str, default='depths_invdepth', help='深度图目录名 (默认: depths_invdepth)')
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    scene_name = os.path.basename(data_path)

    # 输入目录
    images_dir = os.path.join(data_path, 'images')
    depth_dir = os.path.join(data_path, args.depth_dir)

    # 输出目录
    output_dir = os.path.join(
        os.path.dirname(data_path),  # 父目录
        f'preprocessed_masks_{args.mask_param}',
        scene_name,
        f'r{args.resolution}'
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"DDGS 预处理 Mask 生成")
    print(f"=" * 60)
    print(f"数据目录: {data_path}")
    print(f"场景名称: {scene_name}")
    print(f"深度图目录: {depth_dir}")
    print(f"Mask 参数: {args.mask_param} (最远 {args.mask_param}% 区域)")
    print(f"分辨率因子: {args.resolution}")
    print(f"输出目录: {output_dir}")
    print(f"=" * 60)

    # 检查目录
    if not os.path.exists(images_dir):
        print(f"[错误] 图像目录不存在: {images_dir}")
        return
    if not os.path.exists(depth_dir):
        print(f"[错误] 深度图目录不存在: {depth_dir}")
        return

    # 获取图像列表
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))])

    if not image_files:
        print(f"[错误] 未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张图像")
    print()

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]

        # 读取原始图像获取尺寸
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path)
        orig_w, orig_h = img.size

        # 计算目标尺寸
        target_w = orig_w // args.resolution
        target_h = orig_h // args.resolution
        target_size = (target_w, target_h)

        # 查找对应的深度图
        depth_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG']:
            candidate = os.path.join(depth_dir, base_name + ext)
            if os.path.exists(candidate):
                depth_path = candidate
                break

        if depth_path is None:
            print(f"  [警告] 未找到深度图: {base_name}, 跳过")
            continue

        # 生成 mask
        far_mask = generate_far_mask(depth_path, args.mask_param, target_size)

        # 保存
        output_path = os.path.join(output_dir, f'{base_name}.pt')
        torch.save(far_mask, output_path)

        # 统计
        far_ratio = far_mask.sum().item() / far_mask.numel() * 100
        print(f"  {base_name}: size={target_w}x{target_h}, far_ratio={far_ratio:.1f}%")

    print()
    print(f"=" * 60)
    print(f"Mask 生成完成!")
    print(f"输出目录: {output_dir}")
    print(f"=" * 60)
    print()
    print(f"现在可以运行训练:")
    print(f"  python train.py -s {data_path} -m output/{scene_name} \\")
    print(f"      --n_views 9 -r {args.resolution} \\")
    print(f"      --mask_param {args.mask_param} --lambda_far 0.5")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
将包含instance_id的PLY文件转换为可视化PLY文件。
流程: instance_id -> matplotlib颜色映射 -> RGB2SH -> 新PLY

用法:
    python tools/convert_instance_id_to_rgb_ply.py input.ply output.ply
    python tools/convert_instance_id_to_rgb_ply.py input.ply output.ply --colormap tab20
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt


C0 = 0.28209479177387814


def RGB2SH(rgb):
    """RGB [0,1] -> SH DC系数"""
    return (rgb - 0.5) / C0


def load_ply_with_instance_id(path):
    """加载PLY文件，提取所有属性"""
    plydata = PlyData.read(path)
    vertex = plydata.elements[0]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1)
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    f_rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1])
    )
    f_rest = np.stack([vertex[name] for name in f_rest_names], axis=1) if f_rest_names else None

    opacity = vertex["opacity"]

    scale_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1])
    )
    scales = np.stack([vertex[name] for name in scale_names], axis=1)

    rot_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1])
    )
    rotations = np.stack([vertex[name] for name in rot_names], axis=1)

    instance_ids = np.asarray(vertex["instance_id"]) if "instance_id" in [p.name for p in vertex.properties] else None

    return {
        "xyz": xyz,
        "normals": normals,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "instance_ids": instance_ids,
        "n_f_rest": len(f_rest_names)
    }


def generate_distinct_colors(n):
    """生成n个高区分度颜色 (HSV空间均匀采样+打乱顺序)"""
    import colorsys
    colors = []
    # 使用黄金比例来分散色相，避免相邻颜色太接近
    golden_ratio = 0.618033988749895
    h = 0.0
    for i in range(n):
        h = (h + golden_ratio) % 1.0
        # 变化饱和度和明度增加区分度
        s = 0.6 + 0.4 * ((i % 3) / 2)       # 0.6, 0.8, 1.0
        v = 0.7 + 0.3 * ((i // 3) % 3) / 2  # 0.7, 0.85, 1.0
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    return np.array(colors, dtype=np.float32)


def instance_id_to_rgb(instance_ids, colormap='distinct'):
    """将instance_id映射为RGB颜色"""
    unique_ids = np.unique(instance_ids)
    n_instances = len(unique_ids)
    print(f"Found {n_instances} unique instance IDs: min={unique_ids.min()}, max={unique_ids.max()}")

    # 创建ID到索引的映射
    id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}

    # 生成颜色
    if colormap == 'distinct':
        # 使用黄金比例生成高区分度颜色
        colors = generate_distinct_colors(n_instances)
    else:
        # 使用matplotlib colormap
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_instances))[:, :3]

    # 映射每个点的颜色
    indices = np.array([id_to_idx[id_val] for id_val in instance_ids])
    rgb = colors[indices]

    return rgb


def save_ply(path, data, new_f_dc):
    """保存新的PLY文件"""
    xyz = data["xyz"]
    normals = data["normals"]
    opacity = data["opacity"]
    scales = data["scales"]
    rotations = data["rotations"]
    n_f_rest = data["n_f_rest"]

    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attr_names += ['f_dc_0', 'f_dc_1', 'f_dc_2']
    attr_names += [f'f_rest_{i}' for i in range(n_f_rest)]
    attr_names += ['opacity']
    attr_names += [f'scale_{i}' for i in range(scales.shape[1])]
    attr_names += [f'rot_{i}' for i in range(rotations.shape[1])]

    dtype_full = [(name, 'f4') for name in attr_names]
    f_rest_zeros = np.zeros((xyz.shape[0], n_f_rest), dtype=np.float32)

    attributes = np.concatenate([
        xyz, normals, new_f_dc, f_rest_zeros,
        opacity[:, np.newaxis] if opacity.ndim == 1 else opacity,
        scales, rotations
    ], axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def main():
    parser = argparse.ArgumentParser(description="转换instance_id为可视化RGB颜色的PLY")
    parser.add_argument("input_ply", type=str, help="输入PLY文件路径")
    parser.add_argument("output_ply", type=str, help="输出PLY文件路径")
    parser.add_argument("--colormap", type=str, default="distinct",
                        help="颜色映射: distinct(默认,黄金比例高区分度), 或matplotlib的hsv/turbo/rainbow等")
    args = parser.parse_args()

    print(f"Loading: {args.input_ply}")
    data = load_ply_with_instance_id(args.input_ply)

    if data["instance_ids"] is None:
        print("Error: PLY文件中没有找到instance_id属性")
        return

    print(f"Loaded {data['xyz'].shape[0]} points")

    print(f"Mapping instance_id to RGB (colormap={args.colormap})...")
    rgb = instance_id_to_rgb(data["instance_ids"], args.colormap)

    print("Converting RGB -> SH...")
    new_f_dc = RGB2SH(rgb).astype(np.float32)

    print(f"Saving: {args.output_ply}")
    save_ply(args.output_ply, data, new_f_dc)

    print("Done!")


if __name__ == "__main__":
    main()

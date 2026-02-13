#!/usr/bin/env python3
"""
将包含8D特征的PLY文件转换为可视化PLY文件。
流程: 8D features -> PCA降维到3D -> 归一化 -> RGB2SH -> 新PLY

用法:
    python tools/convert_8d_to_rgb_ply.py input.ply output.ply
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA


C0 = 0.28209479177387814


def RGB2SH(rgb):
    """RGB [0,1] -> SH DC系数"""
    return (rgb - 0.5) / C0


def load_ply_with_8d(path):
    """加载PLY文件，提取所有属性"""
    plydata = PlyData.read(path)
    vertex = plydata.elements[0]

    # 基础属性
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    # 法线
    normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1)

    # 原始SH DC
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    # 原始SH rest
    f_rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1])
    )
    if f_rest_names:
        f_rest = np.stack([vertex[name] for name in f_rest_names], axis=1)
    else:
        f_rest = None

    # opacity, scale, rotation
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

    # 8D特征
    f_8d_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_8d_")],
        key=lambda x: int(x.split("_")[-1])
    )
    if f_8d_names:
        features_8d = np.stack([vertex[name] for name in f_8d_names], axis=1)
    else:
        features_8d = None

    return {
        "xyz": xyz,
        "normals": normals,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "features_8d": features_8d,
        "n_f_rest": len(f_rest_names)
    }


def pca_8d_to_3d(features_8d):
    """PCA降维: 8D -> 3D，并归一化到[0,1]"""
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features_8d)

    # 打印方差解释率
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    # Min-max归一化到[0,1]
    f_min = features_3d.min(axis=0, keepdims=True)
    f_max = features_3d.max(axis=0, keepdims=True)
    features_3d_norm = (features_3d - f_min) / (f_max - f_min + 1e-8)

    return features_3d_norm


def save_ply(path, data, new_f_dc):
    """保存新的PLY文件，用新的f_dc替换原来的，清空f_rest"""
    xyz = data["xyz"]
    normals = data["normals"]
    opacity = data["opacity"]
    scales = data["scales"]
    rotations = data["rotations"]
    n_f_rest = data["n_f_rest"]

    # 构建属性列表
    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attr_names += ['f_dc_0', 'f_dc_1', 'f_dc_2']
    attr_names += [f'f_rest_{i}' for i in range(n_f_rest)]
    attr_names += ['opacity']
    attr_names += [f'scale_{i}' for i in range(scales.shape[1])]
    attr_names += [f'rot_{i}' for i in range(rotations.shape[1])]

    dtype_full = [(name, 'f4') for name in attr_names]

    # 创建f_rest全零（只用DC项）
    f_rest_zeros = np.zeros((xyz.shape[0], n_f_rest), dtype=np.float32)

    # 组合所有属性
    attributes = np.concatenate([
        xyz,
        normals,
        new_f_dc,
        f_rest_zeros,
        opacity[:, np.newaxis] if opacity.ndim == 1 else opacity,
        scales,
        rotations
    ], axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def main():
    parser = argparse.ArgumentParser(description="转换8D特征PLY为可视化PLY")
    parser.add_argument("input_ply", type=str, help="输入PLY文件路径")
    parser.add_argument("output_ply", type=str, help="输出PLY文件路径")
    args = parser.parse_args()

    print(f"Loading: {args.input_ply}")
    data = load_ply_with_8d(args.input_ply)

    if data["features_8d"] is None:
        print("Error: PLY文件中没有找到8D特征 (f_8d_*)")
        return

    print(f"Loaded {data['xyz'].shape[0]} points")
    print(f"8D features shape: {data['features_8d'].shape}")

    # PCA降维
    print("Applying PCA: 8D -> 3D...")
    features_3d = pca_8d_to_3d(data["features_8d"])
    print(f"3D features range: [{features_3d.min():.4f}, {features_3d.max():.4f}]")

    # RGB2SH转换
    print("Converting RGB -> SH...")
    new_f_dc = RGB2SH(features_3d).astype(np.float32)

    # 保存
    print(f"Saving: {args.output_ply}")
    save_ply(args.output_ply, data, new_f_dc)

    print("Done!")


if __name__ == "__main__":
    main()

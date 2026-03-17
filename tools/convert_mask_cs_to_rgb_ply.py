#!/usr/bin/env python3
"""
将包含 mask_cs 的PLY文件转换为可视化PLY：mask_cs==1 的高斯染成红色，其余保持原色。

用法:
    python tools/convert_mask_cs_to_rgb_ply.py input.ply output.ply
    python tools/convert_mask_cs_to_rgb_ply.py input.ply output.ply --bg_color 0.5 0.5 0.5
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def RGB2SH(rgb):
    """RGB [0,1] -> SH DC系数"""
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """SH DC系数 -> RGB [0,1]"""
    return sh * C0 + 0.5


def load_ply(path):
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

    prop_names = [p.name for p in vertex.properties]
    mask_cs = np.asarray(vertex["mask_cs"]).reshape(-1) if "mask_cs" in prop_names else None

    return {
        "xyz": xyz,
        "normals": normals,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "mask_cs": mask_cs,
        "n_f_rest": len(f_rest_names),
    }


def save_ply(path, data, new_f_dc):
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
    parser = argparse.ArgumentParser(description="mask_cs==1 染红色，其余保持原色")
    parser.add_argument("input_ply", type=str, help="输入PLY文件路径")
    parser.add_argument("output_ply", type=str, help="输出PLY文件路径")
    parser.add_argument("--bg_color", type=float, nargs=3, default=None,
                        metavar=("R", "G", "B"),
                        help="mask_cs==0 的覆盖颜色 [0,1]，不指定则保持原始SH颜色")
    args = parser.parse_args()

    print(f"Loading: {args.input_ply}")
    data = load_ply(args.input_ply)
    n = data["xyz"].shape[0]
    print(f"Loaded {n} Gaussians")

    if data["mask_cs"] is None:
        print("Error: PLY文件中没有找到 mask_cs 属性")
        return

    mask = data["mask_cs"] > 0.5  # (N,) bool
    n_fg = mask.sum()
    print(f"mask_cs==1 (foreground): {n_fg} / {n}  ({100*n_fg/n:.1f}%)")

    # 从现有 f_dc 还原当前颜色，或由 bg_color 覆盖
    new_f_dc = data["f_dc"].copy().astype(np.float32)  # (N, 3)

    if args.bg_color is not None:
        bg_rgb = np.array(args.bg_color, dtype=np.float32)
        new_f_dc[~mask] = RGB2SH(bg_rgb)

    # 前景染红色 RGB=(1, 0, 0)
    red_sh = RGB2SH(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    new_f_dc[mask] = red_sh

    print(f"Saving: {args.output_ply}")
    save_ply(args.output_ply, data, new_f_dc)
    print("Done!")


if __name__ == "__main__":
    main()

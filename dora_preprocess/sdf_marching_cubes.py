#!/usr/bin/env python3
#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import os
from typing import Tuple

import numpy as np
import trimesh
from pysdf import SDF
from skimage import measure
from tqdm import tqdm


def build_sdf_grid(mesh: trimesh.Trimesh,
                   resolution: int = 256,
                   padding: float = 0.05,
                   chunk_size: int = 1_000_000,
                   dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SDF on a uniform 3D grid covering mesh bounding box with padding.

    Returns
    - sdf: (Z, Y, X) float32 grid of signed distances (in world units)
    - origin: (3,) world-space origin (min corner) of the grid
    - spacing: (3,) voxel spacing along x,y,z (dx, dy, dz)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh must be a trimesh.Trimesh instance")

    bounds = mesh.bounds  # (2,3): min, max
    size = bounds[1] - bounds[0]
    pad = np.linalg.norm(size) * padding
    bb_min = bounds[0] - pad
    bb_max = bounds[1] + pad

    # Uniform resolution across axes; keep anisotropic spacing if bbox not cubic
    nx = ny = nz = int(resolution)
    xs = np.linspace(bb_min[0], bb_max[0], nx, dtype=dtype)
    ys = np.linspace(bb_min[1], bb_max[1], ny, dtype=dtype)
    zs = np.linspace(bb_min[2], bb_max[2], nz, dtype=dtype)

    dx = (xs[-1] - xs[0]) / (nx - 1) if nx > 1 else dtype(1.0)
    dy = (ys[-1] - ys[0]) / (ny - 1) if ny > 1 else dtype(1.0)
    dz = (zs[-1] - zs[0]) / (nz - 1) if nz > 1 else dtype(1.0)

    # Generate grid points (Z, Y, X) ordering for memory locality
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    sdf_eval = SDF(mesh.vertices, mesh.faces)

    sdf_vals = np.empty((pts.shape[0],), dtype=dtype)
    for start in tqdm(range(0, pts.shape[0], chunk_size), desc="Evaluating SDF", leave=False):
        end = min(start + chunk_size, pts.shape[0])
        sdf_vals[start:end] = sdf_eval(pts[start:end]).astype(dtype, copy=False)

    sdf_grid = sdf_vals.reshape((nz, ny, nx))

    origin = np.array([xs[0], ys[0], zs[0]], dtype=dtype)
    spacing = np.array([dx, dy, dz], dtype=dtype)
    return sdf_grid, origin, spacing


def marching_cubes_from_sdf(sdf_grid: np.ndarray,
                            origin: np.ndarray,
                            spacing: np.ndarray,
                            level: float = 0.0):
    """
    Extract surface mesh at given SDF level using marching cubes.
    Returns vertices (world coords) and faces (int32).
    """
    # skimage expects array order (z, y, x); spacing order should match axes
    verts, faces, normals, _ = measure.marching_cubes(
        sdf_grid, level=level, spacing=(spacing[2], spacing[1], spacing[0])
    )
    # marching_cubes returns coordinates with (z,y,x) axes; convert to (x,y,z)
    verts_xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)
    # Shift by origin (origin is (x0, y0, z0))
    verts_world = verts_xyz + origin[None, :]
    return verts_world.astype(np.float32), faces.astype(np.int32)


def save_mesh(verts: np.ndarray, faces: np.ndarray, out_path: str):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.is_watertight:
        print("输出网格为 watertight。")
    else:
        print("输出网格非 watertight。")
    mesh.export(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compute SDF on a uniform grid and reconstruct mesh via Marching Cubes.")
    parser.add_argument("--obj_path", type=str, required=True, help="输入OBJ文件路径")
    parser.add_argument("--grid_resolution", type=int, default=256, help="每轴体素数，比如256表示256^3体素")
    parser.add_argument("--padding", type=float, default=0.05, help="相对包围盒对角线的额外边界比例")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="SDF批量评估点数，避免内存峰值过高")
    parser.add_argument("--out_mesh", type=str, required=True, help="输出网格路径（.obj或. ply）")
    parser.add_argument("--out_sdf_npz", type=str, default="", help="可选：保存SDF体素网格(npz)，含origin/spacing")
    args = parser.parse_args()

    if not os.path.exists(args.obj_path):
        raise FileNotFoundError(f"OBJ 不存在: {args.obj_path}")

    mesh_or_scene = trimesh.load(args.obj_path, process=False)

    if isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    elif isinstance(mesh_or_scene, trimesh.Scene):
        # 将 Scene 中的几何体合并为一个 Trimesh
        if hasattr(mesh_or_scene, "geometry") and len(mesh_or_scene.geometry) > 0:
            mesh = trimesh.util.concatenate(
                [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        else:
            raise RuntimeError("Scene 中没有可用的 Trimesh 几何体")
    else:
        # 某些版本的 trimesh.load 可能返回 list[Trimesh]
        if isinstance(mesh_or_scene, (list, tuple)):
            meshes = [m for m in mesh_or_scene if isinstance(m, trimesh.Trimesh)]
            if not meshes:
                raise RuntimeError("无法从OBJ构建Trimesh网格（列表中无 Trimesh）")
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise RuntimeError(f"无法从OBJ构建Trimesh网格，类型: {type(mesh_or_scene)}")

    if not mesh.is_watertight:
        print("[警告] 网格非watertight，SDF可能不准确；将继续计算。")

    print("[1/3] 构建均匀SDF网格...")
    sdf_grid, origin, spacing = build_sdf_grid(
        mesh, resolution=args.grid_resolution, padding=args.padding, chunk_size=args.chunk_size
    )

    if args.out_sdf_npz:
        os.makedirs(os.path.dirname(args.out_sdf_npz), exist_ok=True)
        np.savez_compressed(args.out_sdf_npz, sdf=sdf_grid.astype(np.float32), origin=origin, spacing=spacing)
        print(f"已保存 SDF 网格: {args.out_sdf_npz}")

    print("[2/3] Marching Cubes 重建表面...")
    verts, faces = marching_cubes_from_sdf(sdf_grid, origin, spacing, level=0.0)

    print("[3/3] 保存重建网格...")
    os.makedirs(os.path.dirname(args.out_mesh), exist_ok=True)
    save_mesh(verts, faces, args.out_mesh)
    print(f"完成，输出网格: {args.out_mesh}")


if __name__ == "__main__":
    main()

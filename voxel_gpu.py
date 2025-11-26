import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import trimesh
from tqdm import tqdm
import multiprocessing as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated voxel ray fill with batching and progress feedback",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing OBJ mesh files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./voxelized_outputs"),
        help="Output directory for the voxelized OBJ meshes",
    )
    parser.add_argument("--pitch", type=float, default=1.0, help="Voxel size")
    parser.add_argument(
        "--ground-layer",
        type=int,
        default=0,
        help="Y index used to seal the base layer",
    )
    parser.add_argument(
        "--ray-batch",
        type=int,
        default=65536,
        help="Number of rays traced per batch to avoid host memory spikes",
    )
    parser.add_argument(
        "--line-batch",
        type=int,
        default=16384,
        help="Number of axis-aligned lines rasterized per batch when painting voxels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (cuda or cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行处理的进程数（每个进程绑定一个GPU或CPU）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="逗号分隔的GPU编号列表，例如 '0,1,2'；为空则使用默认device",
    )
    return parser.parse_args()


def ray_mesh_intersection_batched(
    mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    total = ray_origins.shape[0]

    point_chunks = []
    face_chunks = []
    distance_chunks = []
    ray_index_chunks = []

    iterator = range(0, total, batch_size)
    for start in tqdm(iterator, desc="Tracing rays", unit="batch"):
        end = min(start + batch_size, total)
        chunk_origins = ray_origins[start:end]
        chunk_dirs = ray_directions[start:end]

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=chunk_origins,
            ray_directions=chunk_dirs,
            multiple_hits=False,
        )

        if len(locations) == 0:
            continue

        distances = np.linalg.norm(
            locations - chunk_origins[index_ray],
            axis=1,
        )

        point_chunks.append(locations)
        face_chunks.append(index_tri)
        distance_chunks.append(distances)
        ray_index_chunks.append(index_ray + start)

    if not point_chunks:
        return {
            "face_index": np.empty(0, dtype=np.int32),
            "point": np.empty((0, 3), dtype=np.float32),
            "distance": np.empty(0, dtype=np.float32),
            "ray_index": np.empty(0, dtype=np.int32),
        }

    return {
        "face_index": np.concatenate(face_chunks),
        "point": np.concatenate(point_chunks),
        "distance": np.concatenate(distance_chunks),
        "ray_index": np.concatenate(ray_index_chunks),
    }


def filter_forward_hits(
    mesh: trimesh.Trimesh,
    intersections: Dict[str, np.ndarray],
    ray_directions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if intersections["point"].shape[0] == 0:
        return np.zeros(0, dtype=bool)

    normals = torch.from_numpy(mesh.face_normals[intersections["face_index"]]).to(device)
    directions = torch.from_numpy(ray_directions[intersections["ray_index"]]).to(device)

    mask = torch.sum(normals * directions, dim=1) > 0
    return mask.cpu().numpy()


def build_base_voxel_layer(voxels: trimesh.voxel.VoxelGrid, layer_index: int) -> trimesh.voxel.VoxelGrid:
    matrix = voxels.matrix.copy()
    y_index = int(np.clip(layer_index, 0, matrix.shape[1] - 1))
    xz_footprint = matrix.any(axis=1)
    matrix[:, y_index, :] |= xz_footprint
    return trimesh.voxel.VoxelGrid(
        encoding=matrix,
        transform=voxels.transform,
        metadata=voxels.metadata,
    )


def _rasterize_axis_aligned_line(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff_axes = np.nonzero(a != b)[0]
    if diff_axes.size > 1:
        raise ValueError("Only axis-aligned lines are supported")

    if diff_axes.size == 0:
        pts = a.reshape(1, 3)
        return pts[:, 0], pts[:, 1], pts[:, 2]

    axis = diff_axes[0]
    delta = b[axis] - a[axis]
    step = 1 if delta >= 0 else -1
    axis_values = np.arange(a[axis], b[axis] + step, step, dtype=np.int32)

    repeated = np.repeat(a.reshape(1, 3), axis_values.shape[0], axis=0)
    repeated[:, axis] = axis_values
    return repeated[:, 0], repeated[:, 1], repeated[:, 2]


def paint_voxels_axis_lines(
    start_points: np.ndarray,
    end_points: np.ndarray,
    voxels: trimesh.voxel.VoxelGrid,
    batch_size: int,
) -> int:
    if start_points.shape[0] == 0:
        return 0

    stacked = np.vstack([start_points, end_points])
    indices = voxels.points_to_indices(stacked)
    start_idx = indices[: start_points.shape[0]].astype(np.int32)
    end_idx = indices[start_points.shape[0] :].astype(np.int32)

    grid_max = np.array(voxels.shape) - 1
    start_idx = np.clip(start_idx, 0, grid_max)
    end_idx = np.clip(end_idx, 0, grid_max)

    drawn_voxels = 0
    iterator = range(0, start_idx.shape[0], batch_size)
    for begin in tqdm(iterator, desc="Rasterizing lines", unit="batch"):
        finish = min(begin + batch_size, start_idx.shape[0])
        chunk_start = start_idx[begin:finish]
        chunk_end = end_idx[begin:finish]

        for s, e in zip(chunk_start, chunk_end):
            ix, iy, iz = _rasterize_axis_aligned_line(s, e)
            voxels.matrix[(ix, iy, iz)] = True
            drawn_voxels += ix.shape[0]

    return drawn_voxels


def export_voxel_mesh(voxels: trimesh.voxel.VoxelGrid, output_path: Path) -> None:
    voxel_mesh = voxels.marching_cubes
    voxel_mesh.apply_scale(voxels.pitch)
    voxel_mesh.apply_translation(voxels.transform[:3, 3])
    voxel_mesh.export(output_path)


def process_one_mesh(
    mesh_path: Path,
    output_dir: Path,
    pitch: float,
    ground_layer: int,
    ray_batch: int,
    line_batch: int,
    device_str: str,
) -> None:
    out_name = mesh_path.stem + ".obj"
    out_path = output_dir / out_name
    if out_path.exists():
        print(f"[PID {mp.current_process().pid}] Output {out_path} already exists, skipping.")
        return

    print(f"[PID {mp.current_process().pid}] Processing {mesh_path} on device {device_str} ...")

    device = torch.device(device_str)

    mesh = trimesh.load_mesh(mesh_path)
    # 如果 Y 轴高度（bounds 的 max_y - min_y）大于 70，则跳过
    height_y = float(mesh.bounds[1][1] - mesh.bounds[0][1])
    if height_y > 70.0:
        print(
            f"[PID {mp.current_process().pid}] Skip {mesh_path} because height_y={height_y:.3f} > 70.0",
        )
        return

    print(f"[PID {mp.current_process().pid}] Loaded mesh with extents {mesh.extents} and bounds {mesh.bounds}")

    voxels = mesh.voxelized(pitch=pitch)
    print(f"[PID {mp.current_process().pid}] Created VoxelGrid with shape {voxels.matrix.shape}")

    sealed_voxels = build_base_voxel_layer(voxels, layer_index=ground_layer)

    centers = sealed_voxels.points
    directions = np.tile(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        (centers.shape[0], 1),
    )

    intersections = ray_mesh_intersection_batched(
        mesh,
        centers,
        directions,
        batch_size=max(1, ray_batch),
    )

    forward_mask = filter_forward_hits(mesh, intersections, directions, device)

    valid_ray_indices = intersections["ray_index"][forward_mask]
    start_points = centers[valid_ray_indices]
    end_points = intersections["point"][forward_mask]

    painted = paint_voxels_axis_lines(
        start_points,
        end_points,
        voxels,
        batch_size=max(1, line_batch),
    )
    print(
        f"[PID {mp.current_process().pid}] Painted {painted} voxels across {start_points.shape[0]} "
        f"axis-aligned segments",
    )

    export_voxel_mesh(voxels, out_path)
    print(f"[PID {mp.current_process().pid}] Exported voxel mesh to {out_path}")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(input_dir.glob("*.obj"))
    if not obj_files:
        print(f"No OBJ files found in {input_dir}")
        return

    print(f"Found {len(obj_files)} OBJ files in {input_dir}")

    # 解析 GPU 列表
    gpu_list: List[str] = []
    if args.gpus.strip():
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.num_workers <= 1 and not gpu_list:
        # 单进程、单设备（保持原有逻辑，便于调试）
        device = torch.device(args.device)
        for mesh_path in obj_files:
            process_one_mesh(
                mesh_path,
                output_dir,
                pitch=args.pitch,
                ground_layer=args.ground_layer,
                ray_batch=args.ray_batch,
                line_batch=args.line_batch,
                device_str=str(device),
            )
        return

    # 多进程 / 多 GPU 模式
    if gpu_list:
        print(f"Using GPUs: {gpu_list}")
    else:
        print(f"Using device string for all workers: {args.device}")

    def _worker_init(device_str: str) -> str:
        """简单返回设备字符串，方便 starmap 传参。"""
        return device_str

    tasks = []
    for idx, mesh_path in enumerate(obj_files):
        if gpu_list:
            dev = f"cuda:{gpu_list[idx % len(gpu_list)]}"
        else:
            dev = args.device
        tasks.append(
            (
                mesh_path,
                output_dir,
                args.pitch,
                args.ground_layer,
                args.ray_batch,
                args.line_batch,
                dev,
            )
        )

    num_workers = min(args.num_workers, len(tasks)) if tasks else 0
    if num_workers <= 1:
        for t in tasks:
            process_one_mesh(*t)
        return

    print(f"Launching {num_workers} worker processes ...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.starmap(process_one_mesh, tasks)


if __name__ == "__main__":
    main()

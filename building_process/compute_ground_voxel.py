from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import trimesh
from trimesh.voxel import VoxelGrid

def get_ground_voxel(mesh: trimesh.Trimesh, target_y_index: int = 0, pitch: float = 1.0) -> VoxelGrid:
    voxels = mesh.voxelized(pitch=pitch)

    xz_footprint = voxels.matrix.any(axis=1)

    new_matrix = np.zeros(voxels.matrix.shape, dtype=bool)

    new_matrix[:, target_y_index, :] |= xz_footprint

    new_voxels = trimesh.voxel.VoxelGrid(
        encoding=new_matrix,
        transform=voxels.transform,
        metadata=voxels.metadata
    )

    return new_voxels


def _process_obj_file(task):
    input_path, output_path, pitch, target_y_index = task
    try:
        mesh = trimesh.load_mesh(input_path)
        ground_voxels = get_ground_voxel(mesh, target_y_index=target_y_index, pitch=pitch)

        voxel_mesh = ground_voxels.marching_cubes
        voxel_mesh.apply_scale(ground_voxels.pitch)
        voxel_mesh.apply_translation(ground_voxels.transform[:3, 3])
        voxel_mesh.export(output_path)

        return input_path, True, None
    except Exception as exc:  # noqa: BLE001 - just bubble details to main process
        return input_path, False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Compute ground voxel layer from a mesh")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing building OBJ files")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save ground OBJ files")
    parser.add_argument("--pitch", type=float, default=1.0, help="Voxel size")
    parser.add_argument("--target_y_index", type=int, default=0, help="Y index for the ground layer")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes (default: CPU count)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    obj_files = [file for file in os.listdir(args.input_folder) if file.endswith(".obj")]
    if not obj_files:
        print("No OBJ files found in input folder.")
        return

    tasks = []
    for file in obj_files:
        input_path = os.path.join(args.input_folder, file)
        output_path = os.path.join(args.output_folder, f'{file.replace("building", "ground")}')
        tasks.append((input_path, output_path, args.pitch, args.target_y_index))

    successes = 0
    failures = 0
    with mp.Pool(processes=args.processes) as pool:
        for input_path, ok, error in pool.imap_unordered(_process_obj_file, tasks):
            if ok:
                successes += 1
            else:
                failures += 1
                print(f"Failed to process {input_path}: {error}")

    print(f"Completed {successes} files; {failures} failures.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
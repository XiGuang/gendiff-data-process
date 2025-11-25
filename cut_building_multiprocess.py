"""Multiprocessing variant of cut_building_only.

This script mirrors the single-process logic but distributes chunk handling across
multiple Blender background subprocesses. Each worker receives a contiguous list
of grid cells to process, builds/loads its own scene, and writes results in an
output subfolder under the shared destination.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import platform
import yaml

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - this file runs inside Blender
    bpy = None  # type: ignore

from utils import (
    clear_scene,
    create_cutting_cube,
    export_object_as_obj,
    get_object_bounds,
    import_obj_files,
    merge_objects,
    perform_boolean_intersection,
)


def parse_steps(step_str: str) -> Tuple[float, float]:
    values = [float(x.strip()) for x in step_str.split(",") if x.strip()]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("steps must contain two comma separated numbers")
    return values[0], values[1]


def get_combinations(indices: Sequence[int], start: int, end: int) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    for i in range(start, end + 1):
        combos.extend(itertools.combinations(indices, i))
    return combos


def copy_model(model_obj):
    bpy.ops.object.select_all(action="DESELECT")
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.duplicate()
    return bpy.context.active_object


def max_z_in_xy_range(obj, x_min, x_max, y_min, y_max):
    world_matrix = obj.matrix_world
    max_z_in_range = None
    count = 0
    for vertex in obj.data.vertices:
        world_coord = world_matrix @ vertex.co
        if x_min <= world_coord.x <= x_max and y_min <= world_coord.y <= y_max:
            count += 1
            if max_z_in_range is None or world_coord.z > max_z_in_range:
                max_z_in_range = world_coord.z
    return max_z_in_range, count


@dataclass
class ChunkTask:
    i: int
    j: int
    cube_x: float
    cube_y: float
    cube_z: float
    step_x: float
    step_y: float
    cube_size: float
    threshold_z: float
    z_offset: float


@dataclass
class WorkerArgs:
    chunk_list: List[ChunkTask]
    cube_size: float
    output_folder: str
    building_model_folder: str
    threshold: int


def compute_scene_bounds(building_models) -> Tuple[List[float], List[float]]:
    overall_min_coords = None
    overall_max_coords = None
    for model in building_models:
        min_coords, max_coords = get_object_bounds(model)
        if overall_min_coords is None:
            overall_min_coords = min_coords
            overall_max_coords = max_coords
        else:
            overall_min_coords = [min(overall_min_coords[i], min_coords[i]) for i in range(3)]
            overall_max_coords = [max(overall_max_coords[i], max_coords[i]) for i in range(3)]
    return overall_min_coords, overall_max_coords


def discover_chunks(building_models, step_x, step_y, cube_size, z_offset):
    min_coords, max_coords = compute_scene_bounds(building_models)
    size_x = max_coords[0] - min_coords[0]
    size_y = max_coords[1] - min_coords[1]
    grid_divisions = [
        math.ceil(size_x / step_x) if step_x > 0 else 1,
        math.ceil(size_y / step_y) if step_y > 0 else 1,
    ]
    chunks: List[ChunkTask] = []
    for i in range(grid_divisions[0]):
        for j in range(grid_divisions[1]):
            cube_x = min_coords[0] + (i + 0.5) * step_x
            cube_y = min_coords[1] + (j + 0.5) * step_y
            cube_z = min_coords[2] + z_offset
            threshold_z = min_coords[2] + cube_size/2 + z_offset
            chunks.append(
                ChunkTask(
                    i=i,
                    j=j,
                    cube_x=cube_x,
                    cube_y=cube_y,
                    cube_z=cube_z,
                    step_x=step_x,
                    step_y=step_y,
                    cube_size=cube_size,
                    threshold_z=threshold_z,
                    z_offset=z_offset,
                )
            )
    return chunks


def process_chunk(chunk: ChunkTask, building_models, output_folder, cube_size, threshold):
    indices = list(range(len(building_models)))
    x_min_cell = chunk.cube_x - cube_size / 2
    x_max_cell = chunk.cube_x + cube_size / 2
    y_min_cell = chunk.cube_y - cube_size / 2
    y_max_cell = chunk.cube_y + cube_size / 2
    cube_z = chunk.cube_z

    intersected_indices = []
    skip = False
    max_z_building = None
    for index in indices:
        max_z_building, count = max_z_in_xy_range(
            building_models[index], x_min_cell, x_max_cell, y_min_cell, y_max_cell
        )
        if max_z_building is not None:
            if max_z_building > chunk.threshold_z:
                skip = True
                break
            else:
                if count < threshold:
                    print(
                        f"Skip building {index}, vertex count {count} lower than threshold {threshold}"
                    )
                else:
                    intersected_indices.append(index)

    if skip:
        print(
            f"Skip chunk {chunk.i}_{chunk.j}, building max z {max_z_building:.4f} > threshold {chunk.threshold_z:.4f}"
        )
        return

    if len(intersected_indices) == 0:
        print(f"No intersected building in chunk {chunk.i}_{chunk.j}")
        return

    cube = create_cutting_cube(cube_size, (chunk.cube_x, chunk.cube_y, cube_z))
    building_indices_combinations = get_combinations(intersected_indices, 0, len(intersected_indices))
    for k, building_indices in enumerate(building_indices_combinations):
        chunk_folder = os.path.join(output_folder, f"{chunk.i}_{chunk.j}_{k}")
        os.makedirs(chunk_folder, exist_ok=True)
        ground_indices = set(intersected_indices) - set(building_indices)
        pieces = [copy_model(building_models[idx]) for idx in building_indices]
        if pieces:
            merged_piece = merge_objects(pieces, f"bs_{chunk.i}_{chunk.j}_{k}")
            b_piece = perform_boolean_intersection(merged_piece, cube, epsilon=1e-2)
            export_object_as_obj(b_piece, chunk_folder, use_coordinates=False)
        data = {
            "position": [chunk.cube_x, cube_z, -chunk.cube_y],
            "size": cube_size,
            "building_indices": list(building_indices),
            "building_names": [building_models[idx].name for idx in building_indices],
            "ground_indices": list(ground_indices),
            "index": [chunk.i, chunk.j, k],
            "z_offset": chunk.z_offset,
        }
        with open(os.path.join(chunk_folder, "data.yaml"), "w") as f:
            yaml.dump(data, f)
    bpy.data.objects.remove(cube, do_unlink=True)


def worker_main(args: WorkerArgs):
    clear_scene()
    building_models = import_obj_files(args.building_model_folder)
    building_models.sort(key=lambda x: tuple(map(int, re.findall(r"\d+", x.name))))
    for chunk in args.chunk_list:
        process_chunk(chunk, building_models, args.output_folder, args.cube_size, args.threshold)


def split_chunks(chunks: List[ChunkTask], workers: int) -> List[List[ChunkTask]]:
    partitioned = [[] for _ in range(max(workers, 1))]
    for idx, chunk in enumerate(chunks):
        partitioned[idx % len(partitioned)].append(chunk)
    # Drop empty assignments to avoid spinning up idle Blender workers
    return [group for group in partitioned if group]


def parse_args():
    parser = argparse.ArgumentParser(description="Multiprocessing cut building script")
    parser.add_argument("-out", "--output_folder", required=True, type=str)
    parser.add_argument("-s", "--steps", type=parse_steps, default=parse_steps("100,100"))
    parser.add_argument("-c", "--cube_size", type=float, default=100.0)
    parser.add_argument("-z", "--z_offset", type=float, default=0.0)
    parser.add_argument("-bu", "--building_model_folder", type=str, required=True)
    parser.add_argument("-t", "--threshold", type=int, default=200)
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = args.output_folder
    cube_size = args.cube_size
    z_offset = args.z_offset
    building_model_folder = args.building_model_folder
    threshold = args.threshold

    clear_scene()
    building_models = import_obj_files(building_model_folder)
    building_models.sort(key=lambda x: tuple(map(int, re.findall(r"\d+", x.name))))

    step_x, step_y = args.steps
    chunks = discover_chunks(building_models, step_x, step_y, cube_size, z_offset)
    clear_scene()  # cleanup parent scene before forking workers
    chunk_groups = split_chunks(chunks, args.workers)
    import multiprocessing as mp

    worker_args = [
        WorkerArgs(chunk_group, cube_size, output_folder, building_model_folder, threshold)
        for chunk_group in chunk_groups
    ]

    if platform.system().lower() == 'linux':
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            print("Warning: could not set multiprocessing start method to 'fork'")

    with mp.Pool(processes=len(chunk_groups) or 1) as pool:
        pool.map(worker_main, worker_args)


if __name__ == "__main__":
    main()

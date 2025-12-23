"""Rebuild chunk outputs from existing chunk metadata.

Instead of deriving chunks via grid sampling like cut_building_multiprocess.py,
this script consumes precomputed chunk descriptors (data.yaml) and regenerates
the corresponding OBJ outputs in batch with optional multiprocessing.
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - this file runs inside Blender
    bpy = None  # type: ignore

from utils import (
    clear_scene,
    create_cutting_cube,
    export_object_as_obj,
    import_obj_files,
    merge_objects,
    perform_boolean_intersection,
)


def copy_model(model_obj):
    bpy.ops.object.select_all(action="DESELECT")
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.duplicate()
    return bpy.context.active_object


@dataclass
class YamlChunkTask:
    chunk_id: str
    cube_center: Tuple[float, float, float]
    cube_size: float
    building_indices: List[int]
    payload: Dict
    relative_path: str


@dataclass
class WorkerArgs:
    tasks: List[YamlChunkTask]
    output_folder: str
    building_model_folder: str


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate chunk outputs from data.yaml files")
    parser.add_argument("-out", "--output_folder", required=True, type=str)
    parser.add_argument("-df", "--data_folder", required=True, type=str)
    parser.add_argument("-bu", "--building_model_folder", required=True, type=str)
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def load_yaml_tasks(data_folder: str) -> List[YamlChunkTask]:
    tasks: List[YamlChunkTask] = []
    for root, _, files in os.walk(data_folder):
        if "data.yaml" not in files:
            continue
        data_path = os.path.join(root, "data.yaml")
        with open(data_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        position = payload.get("position", [0.0, 0.0, 0.0])
        if len(position) != 3:
            raise ValueError(f"Invalid position in {data_path}: expected 3 values")
        cube_x = float(position[0])
        cube_z = float(position[1])
        cube_y = -float(position[2])
        cube_size = float(payload.get("size", 0.0))
        building_indices = [int(idx) for idx in payload.get("building_indices", [])]
        rel_path = os.path.relpath(root, data_folder)
        if rel_path == os.curdir:
            rel_path = os.path.basename(root.rstrip(os.sep)) or f"chunk_{len(tasks)}"
        index_triplet = payload.get("index")
        if isinstance(index_triplet, (list, tuple)) and len(index_triplet) == 3:
            chunk_id = f"{int(index_triplet[0])}_{int(index_triplet[1])}_{int(index_triplet[2])}"
        else:
            chunk_id = rel_path.replace(os.sep, "_")
        tasks.append(
            YamlChunkTask(
                chunk_id=chunk_id,
                cube_center=(cube_x, cube_y, cube_z),
                cube_size=cube_size,
                building_indices=building_indices,
                payload=payload,
                relative_path=rel_path,
            )
        )
    return tasks


def split_tasks(tasks: List[YamlChunkTask], workers: int) -> List[List[YamlChunkTask]]:
    partitioned = [[] for _ in range(max(workers, 1))]
    for idx, task in enumerate(tasks):
        partitioned[idx % len(partitioned)].append(task)
    return [group for group in partitioned if group]


def process_task(task: YamlChunkTask, building_models, output_folder: str):
    cube = create_cutting_cube(task.cube_size, task.cube_center)
    valid_indices: List[int] = []
    for idx in task.building_indices:
        if 0 <= idx < len(building_models):
            valid_indices.append(idx)
        else:
            print(f"Skip non-existent building index {idx} in {task.chunk_id}")
    chunk_folder = os.path.join(output_folder, task.relative_path)
    os.makedirs(chunk_folder, exist_ok=True)
    if not valid_indices:
        print(f"No valid buildings for {task.chunk_id}")
        with open(os.path.join(chunk_folder, "data.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(task.payload, f, sort_keys=False, allow_unicode=True)
        bpy.data.objects.remove(cube, do_unlink=True)
        return
    pieces = [copy_model(building_models[idx]) for idx in valid_indices]
    merged_piece = merge_objects(pieces, f"bs_{task.chunk_id}")
    b_piece = perform_boolean_intersection(merged_piece, cube, epsilon=1e-2)
    export_object_as_obj(b_piece, chunk_folder, use_coordinates=False)
    payload = copy.deepcopy(task.payload)
    payload["building_indices"] = valid_indices
    payload["building_names"] = [building_models[idx].name for idx in valid_indices]
    payload["position"] = [task.cube_center[0], task.cube_center[2], -task.cube_center[1]]
    payload["size"] = task.cube_size
    with open(os.path.join(chunk_folder, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    bpy.data.objects.remove(cube, do_unlink=True)
    bpy.data.objects.remove(b_piece, do_unlink=True)


def worker_main(args: WorkerArgs):
    clear_scene()
    building_models = import_obj_files(args.building_model_folder)
    building_models.sort(key=lambda x: tuple(map(int, re.findall(r"\d+", x.name)) or [0, 0, 0]))
    for task in args.tasks:
        process_task(task, building_models, args.output_folder)


def main():
    if bpy is None:
        raise RuntimeError("This script must be executed inside Blender with bpy available")
    args = parse_args()
    tasks = load_yaml_tasks(args.data_folder)
    if not tasks:
        raise RuntimeError(f"No data.yaml files found under {args.data_folder}")
    chunk_groups = split_tasks(tasks, args.workers)
    worker_args = [
        WorkerArgs(group, args.output_folder, args.building_model_folder)
        for group in chunk_groups
    ]
    import multiprocessing as mp

    if platform.system().lower() == "linux":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            print("Warning: could not set multiprocessing start method to 'fork'")

    with mp.Pool(processes=len(chunk_groups) or 1) as pool:
        pool.map(worker_main, worker_args)


if __name__ == "__main__":
    main()

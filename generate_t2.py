'''
    用于将模型切分成多个小块，并立即处理和保存
'''


import argparse
import multiprocessing as mp
import os
import platform
import re
import sys
import yaml
from dataclasses import dataclass
from typing import List, Tuple

import bpy
from utils import (
    clear_scene, import_obj_files, merge_objects, retain_faces_in_cube,create_cutting_cube, perform_boolean_intersection,
    export_object_as_obj
)
import importlib


@dataclass
class ChunkTask:
    subfolder: str
    cube_x: float
    cube_y: float
    cube_z: float
    cube_size: float
    z_offset: float
    building_indices: List[int]
    ground_indices: List[int]
    index: Tuple[int, int, int]


@dataclass
class WorkerArgs:
    chunks: List[ChunkTask]
    output_folder: str
    building_model_folder: str
    ground_model_folder: str
    proc_id: int


def load_chunk_tasks(input_folder: str) -> List[ChunkTask]:
    tasks: List[ChunkTask] = []
    for entry in os.scandir(input_folder):
        if not entry.is_dir():
            continue
        data_file = os.path.join(entry.path, "data.yaml")
        if not os.path.exists(data_file):
            print(f"跳过 {entry.path}, 缺少 data.yaml")
            continue
        with open(data_file, "r") as f:
            data = yaml.safe_load(f)
        position = data['position']
        building_indices = list(data['building_indices'])
        ground_indices = list(data['ground_indices'])
        cube_size = data['size']
        z_offset = data.get('z_offset', 0)
        index = tuple(data['index'])
        i, j, k = index
        cube_x = position[0]
        cube_y = -position[2]
        cube_z = position[1]
        tasks.append(
            ChunkTask(
                subfolder=entry.path,
                cube_x=cube_x,
                cube_y=cube_y,
                cube_z=cube_z,
                cube_size=cube_size,
                z_offset=z_offset,
                building_indices=building_indices,
                ground_indices=ground_indices,
                index=(i, j, k)
            )
        )
    tasks.sort(key=lambda t: t.index)
    return tasks


def process_chunk(task: ChunkTask, building_models, ground_models, output_folder: str):
    i, j, k = task.index
    print(f"处理小块 {i}_{j}_{k}，输出到 {output_folder}")
    chunk_out = os.path.join(output_folder, f"{i}_{j}_{k}")
    os.makedirs(chunk_out, exist_ok=True)

    pieces = [copy_model(building_models[idx]) for idx in task.building_indices]
    for idx in task.ground_indices:
        print(f"添加地面模型 {idx}, {ground_models[idx].name}")
        pieces.append(copy_model(ground_models[idx]))

    if pieces:
        cube = create_cutting_cube(task.cube_size, (task.cube_x, task.cube_y, task.cube_z))
        merged_piece = merge_objects(pieces, f"bs_{i}_{j}_{k}")
        # b_piece = retain_faces_in_cube(merged_piece, (task.cube_x, task.cube_y, task.cube_z), task.cube_size, 'all')
        b_piece = perform_boolean_intersection(merged_piece, cube, epsilon=1e-2)
        export_object_as_obj(b_piece, chunk_out, use_coordinates=False)
        bpy.data.objects.remove(cube, do_unlink=True)
    data = {
        "position": [task.cube_x, task.cube_z, -task.cube_y],
        "size": task.cube_size,
        "building_indices": list(task.building_indices),
        "building_names": [building_models[idx].name for idx in task.building_indices],
        "ground_indices": list(task.ground_indices),
        "z_offset": task.z_offset,
        "index": list(task.index),
    }
    with open(os.path.join(chunk_out, "data.yaml"), "w") as f:
        yaml.dump(data, f)


def split_chunks(chunks: List[ChunkTask], workers: int) -> List[List[ChunkTask]]:
    if workers <= 1:
        return [chunks]
    groups = [[] for _ in range(min(workers, len(chunks)) or 1)]
    for idx, chunk in enumerate(chunks):
        groups[idx % len(groups)].append(chunk)
    return [group for group in groups if group]


def worker_main(args: WorkerArgs):
    clear_scene()
    building_models = import_obj_files(args.building_model_folder)
    building_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
    ground_models = import_obj_files(args.ground_model_folder)
    ground_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
    for chunk in args.chunks:
        print(f"[Worker {args.proc_id} | PID {os.getpid()}] 开始处理小块 {chunk.index}")
        process_chunk(chunk, building_models, ground_models, args.output_folder)

def copy_model(model_obj):
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.duplicate()
    return bpy.context.active_object

def cut_model(input_folder: str, output_folder: str, building_model_folder: str, ground_model_folder: str, workers: int):
    """将模型切分成多个小块，并立即处理和保存"""
    chunks = load_chunk_tasks(input_folder)
    if not chunks:
        print("未发现可处理的小块")
        return 0

    if workers <= 1:
        clear_scene()
        building_models = import_obj_files(building_model_folder)
        building_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
        ground_models = import_obj_files(ground_model_folder)
        ground_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
        for chunk in chunks:
            process_chunk(chunk, building_models, ground_models, output_folder)
        return len(chunks)

    chunk_groups = split_chunks(chunks, workers)
    print(f"启用 {len(chunk_groups)} 个进程处理 {len(chunks)} 个小块")
    if platform.system().lower() == 'linux':
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            print("提示: multiprocessing 启动方式已设定，保持默认")

    worker_args = [
        WorkerArgs(chunks=group, output_folder=output_folder, building_model_folder=building_model_folder, ground_model_folder=ground_model_folder, proc_id=idx)
        for idx, group in enumerate(chunk_groups)
    ]
    with mp.Pool(processes=len(chunk_groups)) as pool:
        pool.map(worker_main, worker_args)
    return len(chunks)

def main():
    print("开始Blender自动化处理...")

    # 在Blender中运行脚本时，我们自己的参数在'--'之后
    try:
        argv_start = sys.argv.index("--") + 1
        _argv = sys.argv[argv_start:]
    except ValueError:
        _argv = []  # 如果没有'--'，则使用空列表

    parser = argparse.ArgumentParser(description="将OBJ模型切割成t1小块")
    parser.add_argument("-out", "--output_folder", required=True, type=str, help="保存切割后模型的输出文件夹路径")
    parser.add_argument("-in", "--input_folder", required=True, type=str, help="输入文件夹路径")
    parser.add_argument("-bu","--building_model_folder",type=str,default="D:/Projects/python/blender_python/building",help="建筑模型")
    parser.add_argument("-g", "--ground_model_folder",type=str,default="D:/Projects/python/blender_python/ground",help="地面模型")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 1, help="并行进程数")
    args = parser.parse_args()

    output_folder = args.output_folder
    input_folder = args.input_folder

    building_model_folder = args.building_model_folder
    ground_model_folder = args.ground_model_folder


    processed = cut_model(input_folder, output_folder, building_model_folder,ground_model_folder, args.workers)

    print(f"处理完成，共处理 {processed} 个小块！")

if __name__ == "__main__":
    main()
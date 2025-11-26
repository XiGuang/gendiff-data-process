import bpy
import yaml
import re
import os
import argparse
import trimesh
import fpsample
import numpy as np
from multiprocessing import Pool, cpu_count

from process_change.sampling_worker import process_obj_to_points

from utils import import_obj_files,retain_faces_in_cube,merge_objects,export_object_as_obj,clear_scene

def copy_model(model_obj):
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.duplicate()
    return bpy.context.active_object

def get_yaml_data(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data

def get_may_change_models(t1_yaml:str):
    with open(t1_yaml, "r") as f:
        data1 = yaml.safe_load(f)

    new_building_models = set(data1['building_names'])
    new_ground_models = set([f'ground{n}' for n in data1['ground_indices']])

    position=data1['position']

    position[1],position[2]=-position[2],position[1]  # 交换y和z轴

    size=data1['size']

    return new_building_models, new_ground_models, position, size

def export_clipped_obj(mesh, save_dir, cube_center, cube_size):
    """在给定立方体内保留mesh并导出为OBJ，返回(名称, OBJ路径)。若为空则返回None。"""
    os.makedirs(save_dir, exist_ok=True)
    copied_obj = copy_model(mesh)
    retained_mesh = retain_faces_in_cube(copied_obj, cube_center, cube_size, 'all')
    if retained_mesh.data.vertices.__len__() == 0:
        print(f"模型 {retained_mesh.name} 在立方体内无顶点，跳过导出")
        bpy.data.objects.remove(retained_mesh, do_unlink=True)
        return None
    name = retained_mesh.name.split('.')[0]

    # 归一化到[-1,1]^3
    scale = 2.0 / cube_size
    retained_mesh.scale = (scale, scale, scale)
    retained_mesh.location = (
        -cube_center[0] * scale,
        -cube_center[1] * scale,
        -cube_center[2] * scale,
    )

    # 导出OBJ
    export_object_as_obj(retained_mesh, save_dir, False)
    obj_path = os.path.join(save_dir, f"{name}.obj")
    bpy.data.objects.remove(retained_mesh, do_unlink=True)
    return name, obj_path

def process_change_model(yaml: str, save_dir: str, building_models, ground_models, sample_points_num: int = 1000, workers: int | None = None):
    new_building_models, new_ground_models, position, size = get_may_change_models(yaml)

    building_model_objs = [obj for obj in building_models if obj.name in new_building_models]
    # ground_model_objs = [obj for obj in ground_models if obj.name in new_ground_models]

    building_out = os.path.join(save_dir, "building")
    ground_out = os.path.join(save_dir, "ground")
    os.makedirs(building_out, exist_ok=True)
    os.makedirs(ground_out, exist_ok=True)

    # 1) 先在Blender中裁剪并导出OBJ（串行，保证bpy安全）
    tasks = []
    for obj in building_model_objs:
        exported = export_clipped_obj(obj, building_out, position, size)
        if exported is None:
            continue
        name, obj_path = exported
        tasks.append((obj_path, building_out, name, sample_points_num))

    # for obj in ground_model_objs:
    #     exported = export_clipped_obj(obj, ground_out, position, size)
    #     if exported is None:
    #         continue
    #     name, obj_path = exported
    #     tasks.append((obj_path, ground_out, name, sample_points_num))

    if not tasks:
        print(f"无可处理任务，跳过{os.path.basename(save_dir)}")
        return

    # 2) 使用多进程并行执行trimesh采样与FPS（CPU密集）
    if workers is None or workers <= 0:
        workers = cpu_count() or 1
    print(f"并行采样进程数: {workers}, 任务数: {len(tasks)}")

    with Pool(processes=workers) as pool:
        for name, ok, msg in pool.imap_unordered(process_obj_to_points, tasks, chunksize=1):
            status = "成功" if ok else "失败"
            if not ok:
                print(f"[{status}] {name}: {msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process change models based on YAML configuration.")
    parser.add_argument("--folder", type=str, required=True, help="包含YAML文件的文件夹的父文件夹路径")
    parser.add_argument("--building_model_folder", type=str, required=True, help="建筑模型OBJ文件夹路径")
    parser.add_argument("--ground_model_folder", type=str, required=True, help="地面模型OBJ文件夹路径")
    parser.add_argument("--sample_points_num", type=int, default=65536, help="每个模型采样的点数")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数（<=0 使用CPU核心数）")
    parser.add_argument("--only_subdirs", type=str, default="", help="仅处理这些子目录，逗号分隔。为空则处理全部")
    parser.add_argument("--subdirs_file", type=str, default="", help="包含要处理的子目录名称（一行一个）的文件路径。优先于only_subdirs")

    args = parser.parse_args()

    folder = args.folder
    building_model_folder = args.building_model_folder
    ground_model_folder = args.ground_model_folder
    sample_points_num = args.sample_points_num
    workers = args.workers

    clear_scene()

    building_models = import_obj_files(building_model_folder)
    for obj in building_models:
        obj.name = obj.name.split("/")[-1].split(".")[0]  # 仅保留文件名作为对象名
    building_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
    # ground_models = import_obj_files(ground_model_folder)
    # for obj in ground_models:
    #     obj.name = obj.name.split("/")[-1].split(".")[0]  # 仅保留文件名作为对象名
    # ground_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
    ground_models = []

    # 计算要处理的子目录列表
    target_names = None
    if args.subdirs_file:
        if os.path.exists(args.subdirs_file):
            with open(args.subdirs_file, 'r', encoding='utf-8') as f:
                target_names = [line.strip() for line in f if line.strip()]
        else:
            print(f"subdirs_file 不存在: {args.subdirs_file}，将回退到 only_subdirs/全部")
    if target_names is None and args.only_subdirs:
        target_names = [x.strip() for x in args.only_subdirs.split(',') if x.strip()]

    if target_names:
        folders = [os.path.join(folder, d) for d in target_names if os.path.isdir(os.path.join(folder, d))]
    else:
        folders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    for f in folders:
        yaml_file = os.path.join(f, "data.yaml")
        if not os.path.exists(yaml_file):
            print(f"文件夹 {f} 中未找到 data.yaml 文件，跳过")
            continue
        process_change_model(yaml_file, f, building_models, ground_models, sample_points_num, workers)
    print("处理完成")
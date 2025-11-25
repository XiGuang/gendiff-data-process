'''
    用于将模型切分成多个小块，并立即处理和保存
'''


import bpy
import math
import argparse
import sys
from utils import (
    clear_scene, import_obj_files, merge_objects, retain_faces_in_cube,
    get_object_bounds, export_object_as_obj,create_cutting_cube, perform_boolean_intersection
)
import re
import itertools
import os
import yaml
import importlib

def get_combinations(indices,start,end):
    all_combinations = []
    for i in range(start,end+1):
        all_combinations.extend(list(itertools.combinations(indices,i)))
    return all_combinations

def copy_model(model_obj):
    bpy.ops.object.select_all(action='DESELECT')
    model_obj.select_set(True)
    bpy.context.view_layer.objects.active = model_obj
    bpy.ops.object.duplicate()
    return bpy.context.active_object

def max_z_in_xy_range(obj, x_min, x_max, y_min, y_max):
    """计算对象在指定 x,y 范围内的最大 z 值（世界坐标系）"""
    world_matrix = obj.matrix_world
    max_z_in_range = None
    # 范围内点数
    count = 0
    for vertex in obj.data.vertices:
        world_coord = world_matrix @ vertex.co
        if x_min <= world_coord.x <= x_max and y_min <= world_coord.y <= y_max:
            count += 1
            if max_z_in_range is None or world_coord.z > max_z_in_range:
                max_z_in_range = world_coord.z
    return max_z_in_range, count

def has_vertices_in_bbox(obj, x_min, x_max, y_min, y_max, z_min, z_max, epsilon: float = 1e-5):
    """
    检查对象是否与指定的轴对齐边界框 (AABB) 相交（世界坐标系）。

    说明：原先按“顶点是否落入框内”的方式容易漏判——当三角面或边穿过框，
    但顶点都在框外时会返回 False。这里改为使用对象世界空间 AABB 与目标 AABB 的
    相交测试，避免漏判；稍有冗余（可能有少量误判为相交），但后续真实裁剪会过滤。
    """

    # 目标切片 AABB（加入微小容差，降低浮点误差导致的边界漏判）
    bx_min = (x_min - epsilon, y_min - epsilon, z_min - epsilon)
    bx_max = (x_max + epsilon, y_max + epsilon, z_max + epsilon)

    # 对象世界空间 AABB：使用 bound_box（局部坐标的 8 个角点）并转换到世界空间
    mw = obj.matrix_world
    # 延迟导入 mathutils，避免在非 Blender 环境下的静态检查报错
    mathutils = importlib.import_module('mathutils')
    Vector = mathutils.Vector
    world_corners = [mw @ Vector(c) for c in obj.bound_box]
    ox_min = (min(v.x for v in world_corners),
              min(v.y for v in world_corners),
              min(v.z for v in world_corners))
    ox_max = (max(v.x for v in world_corners),
              max(v.y for v in world_corners),
              max(v.z for v in world_corners))

    # AABB 相交判断：三个轴都有重叠则相交
    overlap = not (
        ox_max[0] < bx_min[0] or ox_min[0] > bx_max[0] or
        ox_max[1] < bx_min[1] or ox_min[1] > bx_max[1] or
        ox_max[2] < bx_min[2] or ox_min[2] > bx_max[2]
    )
    return overlap

def cut_model(steps, cube_size, output_folder,z_offset, building_models,threshold=200):
    """将模型切分成多个小块，并立即处理和保存"""
    # 获取整体模型的边界框
    overall_min_coords = None
    overall_max_coords = None
    for model in building_models:
        min_coords, max_coords = get_object_bounds(model)
        if overall_min_coords is None:
            overall_min_coords = min_coords
            overall_max_coords = max_coords
        else:
            overall_min_coords = [
                min(overall_min_coords[i], min_coords[i]) for i in range(3)
            ]
            overall_max_coords = [
                max(overall_max_coords[i], max_coords[i]) for i in range(3)
            ]

    size_x = overall_max_coords[0] - overall_min_coords[0]
    size_y = overall_max_coords[1] - overall_min_coords[1]
    step_x, step_y = steps
    
    grid_divisions = [
        math.ceil(size_x / step_x) if step_x > 0 else 1,
        math.ceil(size_y / step_y) if step_y > 0 else 1,
    ]
    
    processed_count = 0
    total_chunks = grid_divisions[0] * grid_divisions[1]

    indices = [i for i in range(len(building_models))]

    for i in range(0, grid_divisions[0]):
        for j in range(0, grid_divisions[1]):
                chunk_index = i * grid_divisions[1] + j
                print(f"处理小块 {i}_{j} ({chunk_index + 1}/{total_chunks})")
                
                # 计算切割长方体的位置
                cube_x = min_coords[0] + (i + 0.5) * step_x
                cube_y = min_coords[1] + (j + 0.5) * step_y
                cube_z = min_coords[2] + z_offset
                
                # 在创建切割立方体前检查该 grid 的 x,y 范围内是否存在 z > min_z + cube_size
                x_min_cell = cube_x - cube_size / 2
                x_max_cell = cube_x + cube_size / 2
                y_min_cell = cube_y - cube_size / 2
                y_max_cell = cube_y + cube_size / 2
                z_min_cell = cube_z - cube_size / 2
                z_max_cell = cube_z + cube_size / 2

                threshold_z = min_coords[2] + cube_size

                intersected_indices = []
                skip=False
                max_z_building=None
                for index in indices:
                    # if has_vertices_in_bbox(building_models[index], x_min_cell, x_max_cell, y_min_cell, y_max_cell, z_min_cell, z_max_cell):
                    #     intersected_indices.append(index)

                    max_z_building, count = max_z_in_xy_range(building_models[index], x_min_cell, x_max_cell, y_min_cell, y_max_cell)
                    if max_z_building is not None:
                        if max_z_building > threshold_z:
                            skip=True
                            break
                        else:
                            if count < threshold:
                                print(f"跳过建筑{index}，building范围内顶点数={count} < 阈值 {threshold}")
                            else:
                                intersected_indices.append(index)
                if skip:
                    print(f"跳过小块 {i}_{j}，building最大z={max_z_building:.4f} > 阈值 {threshold_z:.4f}")
                    processed_count+=1
                    continue

                if len(intersected_indices) > 0:
                    # 创建切割立方体
                    cube = create_cutting_cube(cube_size, (cube_x, cube_y, cube_z))
                    building_indices_combinations = get_combinations(intersected_indices,0,len(intersected_indices))
                    for k,building_indices in enumerate(building_indices_combinations):
                        os.makedirs(os.path.join(output_folder,f"{i}_{j}_{k}"), exist_ok=True)
                        ground_indices = set(intersected_indices) - set(building_indices) # 需要合并的 ground 模型index

                        print(building_indices)
                        print(ground_indices)

                        pieces = []
                        for building_index in building_indices:
                            pieces.append(copy_model(building_models[building_index]))

                        if len(pieces)>0:
                            # 合并pieces
                            merged_piece = merge_objects(pieces,f"bs_{i}_{j}_{k}")

                            b_piece = perform_boolean_intersection(merged_piece, cube,epsilon=1e-2)
                            # b_piece=retain_faces_in_cube(merged_piece,(cube_x, cube_y, cube_z),cube_size,'all')

                            export_object_as_obj(b_piece, os.path.join(output_folder,f"{i}_{j}_{k}"),use_coordinates=False)

                        data = {
                            "position":[cube_x,cube_z,-cube_y],
                            "size":cube_size,
                            "building_indices":list(building_indices), # 从0开始
                            "building_names":[building_models[index].name for index in building_indices],
                            "ground_indices":list(ground_indices), # 从0开始
                            "z_offset": z_offset,
                            "index":[i,j,k]
                        }

                        with open(os.path.join(output_folder,f"{i}_{j}_{k}","data.yaml"),"w") as f:
                            yaml.dump(data,f)
                    bpy.data.objects.remove(cube, do_unlink=True)
                print(f"处理小块 {i}_{j} ({chunk_index + 1}/{total_chunks}) 完成")
                processed_count+=1
    
    return processed_count

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
    parser.add_argument("-s", "--steps", type=str, default="100,100", help="在x,y轴上的步长, 格式为 'x,y' (默认: '100,100')")
    parser.add_argument("-c", "--cube_size", type=float, default=100.0, help="用于切割的立方体的边长 (默认: 100.0)")
    # parser.add_argument("-b", "--border_model_folder", type=str, default="D:/Projects/python/blender_python/border", help="边缘模型")
    parser.add_argument("-z", "--z_offset", type=float, default=0.0, help="z轴偏移量 (默认: 0.0)")
    parser.add_argument("-bu","--building_model_folder",type=str,default="D:/Projects/python/blender_python/building",help="建筑模型")
    parser.add_argument("-t", "--threshold", type=int, default=200, help="跳过顶点数少于该阈值的建筑 (默认: 200)")
    args = parser.parse_args()

    output_folder = args.output_folder
    cube_size = args.cube_size
    # border_model_folder = args.border_model_folder
    z_offset = args.z_offset
    building_model_folder = args.building_model_folder
    threshold = args.threshold

    try:
        steps = [float(x.strip()) for x in args.steps.split(',')]
        if len(steps) != 2:
            raise ValueError("需要提供x,y两个维度的步长")
    except ValueError as e:
        print(f"错误：无效的步长参数 '{args.steps}'. {e}")
        return
    
    # 清除场景
    clear_scene()
    
    building_models = import_obj_files(building_model_folder)
    building_models.sort(key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))

    cut_model(steps, cube_size, output_folder,z_offset,building_models,threshold)

    print("处理完成！")

if __name__ == "__main__":
    main()
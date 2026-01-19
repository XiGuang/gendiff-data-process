import bpy
import os
import bmesh
import mathutils
from mathutils.bvhtree import BVHTree

def clear_scene():
    """清除场景中的所有对象"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_obj_files(folder_path):
    """从指定文件夹（包括子文件夹）导入所有OBJ文件"""
    obj_files = []
    
    # 递归查找所有OBJ文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    
    print(f"找到 {len(obj_files)} 个OBJ文件")
    
    imported_objects = []
    for obj_file in obj_files:
        # 导入OBJ文件
        bpy.ops.wm.obj_import(filepath=obj_file, use_split_objects=False)
        # 获取刚导入的对象
        bpy.context.active_object.name = obj_file.split("\\")[-1].split("/")[-1].split(".")[0]
        imported_objects.extend(bpy.context.selected_objects)
    
    return imported_objects

def merge_objects(objects, name="模型a"):
    """合并多个对象为一个对象"""
    if not objects:
        return None
    
    # 选择所有对象
    bpy.ops.object.select_all(action='DESELECT')
    if len(objects) > 1:
        for obj in objects:
            obj.select_set(True)
        
        # 设置活动对象
        bpy.context.view_layer.objects.active = objects[0]
        
        # 合并对象
        bpy.ops.object.join()
        
        # 重命名合并后的对象
        merged_object = bpy.context.active_object
        merged_object.name = name
    else:
        merged_object = objects[0]
        merged_object.name = name
    
    return merged_object

def create_cutting_cube(size, location):
    """创建用于布尔运算的立方体"""
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    cube = bpy.context.active_object
    return cube

def perform_boolean_intersection(target_obj, cutter_obj, epsilon = 1e-4):
    cut_min, cut_max = get_object_bounds(cutter_obj)

    """执行布尔交集运算"""
    # 为目标对象添加布尔修改器
    bool_modifier = target_obj.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_modifier.operation = 'INTERSECT'
    bool_modifier.solver = 'EXACT'
    bool_modifier.double_threshold = 0.000001
    bool_modifier.object = cutter_obj
    
    # 应用修改器
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply(modifier="Boolean")

    # 切换到编辑模式
    bpy.ops.object.mode_set(mode='EDIT')

    # 创建 BMesh 以操作几何体
    bm = bmesh.from_edit_mesh(target_obj.data)

    # 删除所有没有面的边（Loose Edges）
    loose_edges = [e for e in bm.edges if not e.link_faces]
    bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')

    # 删除所有没有边的点（Loose Verts）
    loose_verts = [v for v in bm.verts if not v.link_edges]
    bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    ''' 删除可能留下的立方体面 '''
    # 获取 BMesh 对象
    # mesh = bmesh.from_edit_mesh(target_obj.data)

    # if len(mesh.faces) > 0:
    #     # 取消所有选中
    #     for f in mesh.faces:
    #         f.select = False

    #     # 获取所有顶点坐标（世界坐标）
    #     world_matrix = target_obj.matrix_world
    #     verts_world = [world_matrix @ v.co for v in mesh.verts]

    #     # 计算整体边界
    #     xs = [v.x for v in verts_world]
    #     ys = [v.y for v in verts_world]
    #     zs = [v.z for v in verts_world]
    #     xmin, xmax = min(xs), max(xs)
    #     ymin, ymax = min(ys), max(ys)
    #     zmin, zmax = min(zs), max(zs)

    #     # 选择靠近边界的面（立方体表面）
    #     for face in mesh.faces:
    #         verts = [world_matrix @ v.co for v in face.verts]
    #         if all(abs(v.x - cut_min[0]) < epsilon for v in verts) or \
    #         all(abs(v.x - cut_max[1]) < epsilon for v in verts) or \
    #         all(abs(v.y - cut_min[1]) < epsilon for v in verts) or \
    #         all(abs(v.y - cut_max[1]) < epsilon for v in verts) or \
    #         all(abs(v.z - cut_min[2]) < epsilon for v in verts) or \
    #         all(abs(v.z - cut_max[2]) < epsilon for v in verts) or \
    #         all(v.x>cut_max[0] or v.x<cut_min[0] or v.y>cut_max[1] or v.y<cut_min[1] or v.z>cut_max[2] or v.z<cut_min[2] for v in verts):
    #             face.select = True

    #     bmesh.ops.delete(mesh, geom=[f for f in mesh.faces if f.select], context='FACES')

    #     # 更新视图
    #     bmesh.update_edit_mesh(target_obj.data)

    bpy.ops.object.mode_set(mode='OBJECT')
    
    return target_obj

def get_object_bounds(obj):
    """获取对象在世界坐标系下的真实边界框"""
    local_corners = [mathutils.Vector(corner) for corner in obj.bound_box]
    world_corners = [obj.matrix_world @ corner for corner in local_corners]
    min_coords = [min(corner[i] for corner in world_corners) for i in range(3)]
    max_coords = [max(corner[i] for corner in world_corners) for i in range(3)]
    return min_coords, max_coords

def create_sealing_planes(size, location, name_prefix="cs"):
    """为对象创建封闭平面（四周和底部）"""
    cube=create_cutting_cube(size, location)
    cube.name=name_prefix
    # 进入编辑模式
    bpy.ops.object.mode_set(mode='EDIT')

    # 获取 BMesh 对象
    mesh = bmesh.from_edit_mesh(cube.data)

    # 找到顶面的面（z轴正向的那个）
    for face in mesh.faces:
        face.select = False  # 先清空选择
        # 判断法线是否几乎垂直向上（即顶面）
        if face.normal.z > 0.99:
            face.select = True
            break  # 如果你只想删掉一个顶面，找到一个就行

    # 删除选中的面
    bmesh.ops.delete(mesh, geom=[f for f in mesh.faces if f.select], context='FACES')

    # 更新编辑网格
    bmesh.update_edit_mesh(cube.data)

    # 回到对象模式（可选）
    bpy.ops.object.mode_set(mode='OBJECT')
    return cube

def export_object_as_obj(obj, output_folder, use_coordinates=True):
    """导出对象为OBJ文件"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 选择对象
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # 使用坐标命名时，确保对象的位置是最新的
    bpy.context.view_layer.update()

    if use_coordinates:
        # 使用坐标命名
        location = obj.location
        filename = f"{obj.name}_{location.x:.2f}_{location.y:.2f}_{location.z:.2f}.obj"
    else:
        filename = f"{(obj.name).split('.')[0]}.obj"
    
    filepath = os.path.join(output_folder, filename)
    
    # 导出OBJ
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        export_materials=False
    )
    
    print(f"已导出: {filepath}")

def build_bvh(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree

def is_intersection(obj1, obj2):
    tree1 = build_bvh(obj1)
    tree2 = build_bvh(obj2)

    return tree1.overlap(tree2)

from mathutils import Vector
def is_inside_cube(obj_mesh, obj_cube):
    # 获取 Cube 的世界坐标下的包围盒八个点
    cube_corners = [obj_cube.matrix_world @ Vector(corner) for corner in obj_cube.bound_box]
    min_corner = Vector((
        min(v.x for v in cube_corners),
        min(v.y for v in cube_corners),
        min(v.z for v in cube_corners)
    ))
    max_corner = Vector((
        max(v.x for v in cube_corners),
        max(v.y for v in cube_corners),
        max(v.z for v in cube_corners)
    ))

    # 检查 mesh 所有顶点是否都在这个包围盒内
    for v in obj_mesh.data.vertices:
        world_coord = obj_mesh.matrix_world @ v.co
        if not (min_corner.x <= world_coord.x <= max_corner.x and
                min_corner.y <= world_coord.y <= max_corner.y and
                min_corner.z <= world_coord.z <= max_corner.z):
            return False
    return True

def retain_faces_in_cube(obj, center, edge_length, contain_mode='any'):
    """仅保留给定对象中位于轴对齐立方体范围内的面。

    规则：若一个面的任意一个顶点落入该立方体（包含边界）则保留该面；否则删除。

    参数:
    - obj: bpy.types.Object，类型需为 'MESH'
    - center: 立方体中心点，tuple/list/Vector，世界坐标
    - edge_length: 立方体边长，float
    - contain_mode: 字符串，'any' 或 'all'。'any' 表示任一顶点在盒内即保留；'all' 表示所有顶点都在盒内才保留。默认 'any'。

    返回:
    - 处理后的 obj（同一对象）
    """
    if obj is None or obj.type != 'MESH':
        raise TypeError("obj 必须是一个网格对象 (type == 'MESH')")
    # 规范并校验模式
    contain_mode = (contain_mode or 'any').lower()
    if contain_mode not in ('any', 'all'):
        raise ValueError("contain_mode 必须是 'any' 或 'all'")

    # 计算世界空间下的AABB
    c = mathutils.Vector(center)
    half = float(edge_length) / 2.0
    min_corner = mathutils.Vector((c.x - half, c.y - half, c.z - half))
    max_corner = mathutils.Vector((c.x + half, c.y + half, c.z + half))

    # 切到对象模式，避免编辑模式下数据不同步
    try:
        if bpy.context.object is obj and obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        # 在无活动对象或无上下文时可忽略
        pass

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    world_matrix = obj.matrix_world

    def vert_in_box_world(w):
        return (
            min_corner.x <= w.x <= max_corner.x and
            min_corner.y <= w.y <= max_corner.y and
            min_corner.z <= w.z <= max_corner.z
        )

    def face_in_box(face):
        if contain_mode == 'any':
            for v in face.verts:
                w = world_matrix @ v.co
                if vert_in_box_world(w):
                    return True
            return False
        else:  # 'all'
            for v in face.verts:
                w = world_matrix @ v.co
                if not vert_in_box_world(w):
                    return False
            return True

    # 标记要删除的面（不满足任意顶点在盒内的面）
    faces_to_delete = [f for f in bm.faces if not face_in_box(f)]
    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')

    # 清理无面连接的边与无边连接的点
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    loose_edges = [e for e in bm.edges if not e.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context='EDGES')
    loose_verts = [v for v in bm.verts if not v.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')

    # 写回网格
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    return obj
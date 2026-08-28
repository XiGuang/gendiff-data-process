import trimesh
import os

def split_with_merge(input_file, output_folder, merge_threshold=1e-5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 加载网格
    # process=False 防止加载时自动发生一些不需要的预处理
    mesh = trimesh.load(input_file, force='mesh', process=False)
    print(f"原始顶点数: {len(mesh.vertices)}")

    # 2. 【关键步骤】合并顶点
    # 这会将距离小于 merge_threshold 的顶点合并为一个。
    # 这样，紧挨着的零件在拓扑上就变成连通的了。
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    print(f"合并后顶点数: {len(mesh.vertices)} (原本断开的部分已焊接)")

    # 3. 再进行连通性拆分
    # only_watertight=False 允许拆分出非闭合的曲面
    components = mesh.split(only_watertight=False)
    print(f"合并顶点后，拆分为 {len(components)} 个部分。")

    # 4. 保存
    for i, component in enumerate(components):
        # 过滤掉太小的碎片（可选），比如少于100个面的可能是噪点
        if len(component.faces) < 50: 
            continue
            
        output_filename = os.path.join(output_folder, f"part_{i:03d}.obj")
        component.export(output_filename)

if __name__ == "__main__":
    split_with_merge("/mnt/d/data/data/block/yingrenshi_building_proxy/0_2_3/bs_0_2_3.obj", "./output/merged_parts")
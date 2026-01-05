import trimesh
import numpy as np
import yaml
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

EPS = 1e-5


def extract_bottom_contour(mesh):
    v = mesh.vertices
    f = mesh.faces

    y_min = v[:, 1].min()

    bottom_faces = f[np.any(np.abs(v[f][:, :, 1] - y_min) < EPS, axis=1)]
    if len(bottom_faces) == 0:
        return None, y_min

    polys = []
    for tri in v[bottom_faces]:
        pts = [(tri[i][0], tri[i][2]) for i in range(3)]
        p = Polygon(pts)
        if p.is_valid and p.area > EPS:
            polys.append(p)

    if not polys:
        return None, y_min

    merged = unary_union(polys)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda p: p.area)

    contour_2d = list(merged.exterior.coords)
    contour_3d = [[float(x), float(y_min), float(z)] for x, z in contour_2d]
    return contour_3d, y_min

def split_with_merge(mesh, merge_threshold=1e-5):
    # 2. 【关键步骤】合并顶点
    # 这会将距离小于 merge_threshold 的顶点合并为一个。
    # 这样，紧挨着的零件在拓扑上就变成连通的了。
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    # print(f"合并后顶点数: {len(mesh.vertices)} (原本断开的部分已焊接)")

    # 3. 再进行连通性拆分
    # only_watertight=False 允许拆分出非闭合的曲面
    components = mesh.split(only_watertight=False)
    # print(f"合并顶点后，拆分为 {len(components)} 个部分。")
    return components

def process_obj(obj_path, yaml_path):
    mesh = trimesh.load(obj_path, force='mesh')

    submeshes = split_with_merge(mesh)

    output = []
    for i, sm in enumerate(submeshes):
        contour, y_min = extract_bottom_contour(sm)
        if contour is None:
            continue

        height = float(sm.vertices[:, 1].max() - y_min)

        output.append({
            "mesh_id": i,
            "height": height,
            "footprint": contour
        })

    with open(yaml_path, "w") as f:
        yaml.dump(output, f, sort_keys=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python obj_to_yaml.py input.obj output.yaml")
        exit(1)

    process_obj(sys.argv[1], sys.argv[2])

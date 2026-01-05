import numpy as np
import trimesh
from shapely.geometry import Polygon
import yaml


def extrude_polygon(contour, height):
    contour = np.asarray(contour, dtype=float)

    # 构造 XZ 平面 polygon
    poly_2d = Polygon([(p[0], p[2]) for p in contour])
    if not poly_2d.is_valid:
        raise ValueError("Invalid polygon")

    # trimesh 要求 CCW
    if not poly_2d.exterior.is_ccw:
        contour = contour[::-1]
        poly_2d = Polygon([(p[0], p[2]) for p in contour])

    y0 = contour[0, 1]

    # 使用 trimesh 官方挤出
    mesh = trimesh.creation.extrude_polygon(
        polygon=poly_2d,
        height=height
    )

    # extrude_polygon 默认 Z-up，这里转成 Y-up
    T = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, y0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    mesh.apply_transform(T)

    return mesh


def yaml_to_obj(yaml_path, obj_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    meshes = []
    for i, item in enumerate(data):
        contour = item["footprint"]
        height = item["height"]

        mesh = extrude_polygon(contour, height)
        mesh.metadata["name"] = f"mesh_{i}"
        meshes.append(mesh)

    combined = trimesh.util.concatenate(meshes)
    combined.export(obj_path)


if __name__ == "__main__":
    yaml_to_obj("output/test.yaml", "output/test.obj")

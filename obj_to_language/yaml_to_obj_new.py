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


def _normalize_entry(entry, index):
    if not isinstance(entry, dict):
        raise ValueError("Each YAML entry must be a mapping")

    contour = entry.get("footprint")
    if contour is None:
        contour = entry.get("bottom_contour")
    if contour is None:
        raise ValueError("Each entry must include footprint or bottom_contour")

    if not isinstance(contour, list):
        raise ValueError("footprint must be a list of [x, y, z] points")

    points = np.asarray(contour, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("footprint must be an Nx3 numeric array")

    unique_points = np.unique(np.round(points, decimals=8), axis=0)
    if len(unique_points) < 3:
        raise ValueError("footprint must contain at least three unique points")

    if "base_height" in entry:
        base_height = float(entry["base_height"])
        if not np.allclose(points[:, 1], base_height, atol=1e-5):
            raise ValueError("footprint Y values must match base_height")
    else:
        base_height = float(points[0, 1])

    height = float(entry["height"])
    if height <= 0:
        raise ValueError("height must be positive")

    name = str(entry.get("name", f"mesh_{index}"))
    return name, points, base_height, height


def mesh_from_yaml_entry(entry, index=0):
    name, contour, _base_height, height = _normalize_entry(entry, index)
    mesh = extrude_polygon(contour, height)
    mesh.metadata["name"] = name
    return mesh


def yaml_entries_to_mesh(entries):
    meshes = []
    for i, entry in enumerate(entries):
        part = mesh_from_yaml_entry(entry, i)
        meshes.append(part)
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def yaml_to_obj(yaml_path, obj_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("YAML root must be a list")
    combined = yaml_entries_to_mesh(data)
    if combined is None:
        raise ValueError("No valid mesh entries found")
    combined.export(obj_path)


if __name__ == "__main__":
    yaml_to_obj("output/test.yaml", "output/test.obj")

#!/usr/bin/env python3
"""Convert a YAML footprint/height description back into a simple OBJ prism."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with `pip install pyyaml`."
    ) from exc

Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]


def clean_polygon(points: Sequence[Point2D], tol: float = 1e-9) -> List[Point2D]:
    if not points:
        return []

    # Remove consecutive duplicates (including wrap-around)
    cleaned: List[Point2D] = []
    for pt in points:
        if cleaned and abs(pt[0] - cleaned[-1][0]) <= tol and abs(pt[1] - cleaned[-1][1]) <= tol:
            continue
        cleaned.append(pt)
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= tol and abs(cleaned[0][1] - cleaned[-1][1]) <= tol:
        cleaned.pop()

    if len(cleaned) <= 2:
        return cleaned

    def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    i = 0
    while len(cleaned) > 2 and i < len(cleaned):
        prev_pt = cleaned[i - 1]
        curr_pt = cleaned[i]
        next_pt = cleaned[(i + 1) % len(cleaned)]
        if abs(cross(prev_pt, curr_pt, next_pt)) <= tol:
            del cleaned[i]
            if len(cleaned) <= 2:
                break
            i = max(i - 1, 0)
            continue
        i += 1
    return cleaned


def load_yaml_meshes(yaml_path: Path) -> List[Tuple[str, List[Point3D], float]]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries: List[Any] = [data]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("YAML root must be a mapping or a list of mappings")

    meshes: List[Tuple[str, List[Point3D], float]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("Each mesh entry must be a mapping")
        name = str(entry.get("name", f"mesh_{idx}"))
        contour = entry.get("bottom_contour")
        height = entry.get("height")
        if contour is None or height is None:
            raise ValueError("Each mesh entry must have bottom_contour and height")
        if not isinstance(contour, list) or len(contour) < 3:
            raise ValueError("bottom_contour must be a list with at least three points")
        bottom: List[Point3D] = []
        for point in contour:
            if (
                not isinstance(point, list)
                or len(point) != 3
                or not all(isinstance(v, (int, float)) for v in point)
            ):
                raise ValueError("Each bottom_contour entry must be [x, y, z] numeric list")
            bottom.append((float(point[0]), float(point[1]), float(point[2])))
        if not isinstance(height, (int, float)):
            raise ValueError("height must be numeric")
        meshes.append((name, bottom, float(height)))

    return meshes


def build_vertices(bottom: Sequence[Point3D], height: float) -> List[Point3D]:
    vertices: List[Point3D] = []
    vertices.extend(bottom)
    for x, y, z in bottom:
        vertices.append((x, y + height, z))
    return vertices


def polygon_area(points: Sequence[Point2D]) -> float:
    area = 0.0
    for i, (x1, z1) in enumerate(points):
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return area * 0.5


def _cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(point: Point2D, tri: Sequence[Point2D], eps: float = 1e-12) -> bool:
    def sign(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, tri[0], tri[1])
    b2 = sign(point, tri[1], tri[2])
    b3 = sign(point, tri[2], tri[0])
    has_neg = (b1 < -eps) or (b2 < -eps) or (b3 < -eps)
    has_pos = (b1 > eps) or (b2 > eps) or (b3 > eps)
    return not (has_neg and has_pos)


def triangulate_polygon(points: Sequence[Point2D]) -> List[Tuple[int, int, int]]:
    points = clean_polygon(points)
    n = len(points)
    if n < 3:
        raise ValueError("Need at least three points to triangulate")
    area = polygon_area(points)
    if abs(area) < 1e-12:
        raise ValueError("Polygon area too small; check for degenerate contour")

    is_ccw = area > 0
    indices = list(range(n))
    if not is_ccw:
        indices.reverse()

    triangles: List[Tuple[int, int, int]] = []
    working = indices.copy()

    def is_convex(prev_idx: int, curr_idx: int, next_idx: int) -> bool:
        return _cross(points[prev_idx], points[curr_idx], points[next_idx]) > 0

    while len(working) > 3:
        ear_found = False
        for i in range(len(working)):
            prev_idx = working[i - 1]
            curr_idx = working[i]
            next_idx = working[(i + 1) % len(working)]
            if not is_convex(prev_idx, curr_idx, next_idx):
                continue
            tri_pts = (points[prev_idx], points[curr_idx], points[next_idx])
            has_point_inside = False
            for other_idx in working:
                if other_idx in (prev_idx, curr_idx, next_idx):
                    continue
                if _point_in_triangle(points[other_idx], tri_pts):
                    has_point_inside = True
                    break
            if has_point_inside:
                continue
            triangles.append((prev_idx, curr_idx, next_idx))
            del working[i]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Failed to triangulate polygon; ensure it is simple and non-self-intersecting")

    triangles.append((working[0], working[1], working[2]))

    if not is_ccw:
        triangles = [tuple(reversed(tri)) for tri in triangles]
    return triangles


def build_faces(bottom: Sequence[Point3D]) -> List[Tuple[int, ...]]:
    bottom_size = len(bottom)
    top_offset = bottom_size
    points_2d: List[Point2D] = [(x, z) for x, _, z in bottom]
    triangles = triangulate_polygon(points_2d)

    faces: List[Tuple[int, ...]] = []
    # Base triangles (reverse to make normals point downward)
    for a, b, c in triangles:
        faces.append((c + 1, b + 1, a + 1))
    # Top triangles (normals upward)
    for a, b, c in triangles:
        faces.append((a + 1 + top_offset, b + 1 + top_offset, c + 1 + top_offset))
    # Side quads along original order to preserve footprint
    for i in range(bottom_size):
        b0 = i + 1
        b1 = (i + 1) % bottom_size + 1
        t1 = b1 + top_offset
        t0 = b0 + top_offset
        faces.append((b0, b1, t1, t0))
    return faces


def write_obj(meshes: Sequence[Tuple[str, List[Point3D], List[Tuple[int, ...]]]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as obj_file:
        obj_file.write("# Generated by yaml_to_obj\n")
        obj_file.write(f"# Mesh count: {len(meshes)}\n")
        vertex_offset = 0
        for name, vertices, faces in meshes:
            obj_file.write(f"o {name}\n")
            for x, y, z in vertices:
                obj_file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for face in faces:
                adjusted = [idx + vertex_offset for idx in face]
                face_indices = " ".join(str(idx) for idx in adjusted)
                obj_file.write(f"f {face_indices}\n")
            vertex_offset += len(vertices)


def run(yaml_path: Path, output_path: Path) -> None:
    mesh_specs = load_yaml_meshes(yaml_path)
    mesh_payload: List[Tuple[str, List[Point3D], List[Tuple[int, ...]]]] = []
    for name, bottom, height in mesh_specs:
        vertices = build_vertices(bottom, height)
        faces = build_faces(bottom)
        mesh_payload.append((name, vertices, faces))
    write_obj(mesh_payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", type=Path, help="Path to the YAML file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination OBJ path (defaults to YAML name with .obj)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_path = args.yaml.expanduser().resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"YAML not found: {yaml_path}")
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else yaml_path.with_suffix(".obj")
    )
    run(yaml_path, output_path)


if __name__ == "__main__":
    main()

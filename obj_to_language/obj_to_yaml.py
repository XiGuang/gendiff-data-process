#!/usr/bin/env python3
"""Convert a Y-up OBJ mesh into a YAML summary with base contour and height."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with `pip install pyyaml`."
    ) from exc

try:
    import trimesh  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trimesh is required. Install it with `pip install trimesh`."
    ) from exc

Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]


def _ensure_trimesh(obj_path: Path) -> "trimesh.Trimesh":
    loaded = trimesh.load(str(obj_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):  # type: ignore[attr-defined]
        geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("Scene contains no mesh geometry")
        return trimesh.util.concatenate(geometries)
    raise TypeError("Unsupported OBJ structure; expected mesh or scene")


def split_connected_components(obj_path: Path, merge_threshold: float) -> List[Tuple[str, List[Point3D]]]:
    base_mesh = _ensure_trimesh(obj_path).copy()
    if merge_threshold > 0:
        base_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    components = base_mesh.split(only_watertight=False)
    if not components:
        components = [base_mesh]

    groups: List[Tuple[str, List[Point3D]]] = []
    for idx, component in enumerate(components):
        verts = component.vertices.tolist()
        if not verts:
            continue
        groups.append(
            (
                f"component_{idx}",
                [(float(x), float(y), float(z)) for x, y, z in verts],
            )
        )

    if not groups:
        raise ValueError("No vertices found after splitting; check the input mesh")

    return groups


def select_base_ring(vertices: Sequence[Point3D], epsilon: float) -> Tuple[List[Point3D], float, float]:
    ys = [v[1] for v in vertices]
    min_y = min(ys)
    max_y = max(ys)
    height = max_y - min_y
    tolerance = max(epsilon, 1e-6)
    base_vertices = [v for v in vertices if abs(v[1] - min_y) <= tolerance]
    if not base_vertices:
        raise ValueError("Failed to locate base vertices; try increasing epsilon")
    return base_vertices, min_y, height


def dedupe_points(points: Iterable[Point2D], tol: float) -> List[Point2D]:
    unique: List[Point2D] = []
    for x, z in points:
        is_duplicate = False
        for px, pz in unique:
            if abs(x - px) <= tol and abs(z - pz) <= tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append((x, z))
    return unique


def convex_hull(points: Sequence[Point2D]) -> List[Point2D]:
    if len(points) < 3:
        return list(points)

    def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    pts = dedupe_points(points, tol=1e-9)
    if len(pts) < 3:
        return pts
    pts = sorted(pts)

    lower: List[Point2D] = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)

    upper: List[Point2D] = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)

    hull = lower[:-1] + upper[:-1]
    return hull


def build_bottom_contour(base_vertices: Sequence[Point3D], base_y: float, planar_tol: float) -> List[List[float]]:
    projected = [(vx, vz) for vx, _, vz in base_vertices]
    pts2d = dedupe_points(projected, tol=planar_tol)
    if not pts2d:
        raise ValueError("No unique base points detected; increase planar tolerance")

    # ordered = convex_hull(pts2d)
    # if not ordered:
    #     ordered = pts2d

    contour = [[x, base_y, z] for x, z in pts2d]
    return contour


def save_yaml(data: Sequence[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as out_file:
        yaml.safe_dump(data, out_file, sort_keys=False)


def run(obj_path: Path, output_path: Path, epsilon: float, planar_tol: float, merge_threshold: float) -> None:
    mesh_vertex_groups = split_connected_components(obj_path, merge_threshold)
    payload: List[dict] = []
    for name, vertices in mesh_vertex_groups:
        base_vertices, base_y, height = select_base_ring(vertices, epsilon)
        contour = build_bottom_contour(base_vertices, base_y, planar_tol)
        payload.append(
            {
                "name": name,
                "bottom_contour": contour,
                "height": height,
            }
        )
    save_yaml(payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", type=Path, help="Path to the Y-up OBJ file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination YAML path (defaults to OBJ name with .yaml)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Tolerance when selecting the lowest ring of vertices",
    )
    parser.add_argument(
        "--planar-tol",
        type=float,
        default=1e-6,
        help="Tolerance when deduplicating planar points",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=1e-5,
        help="Vertex merge radius before detecting connected components (0 disables merging)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obj_path: Path = args.obj.expanduser().resolve()
    if not obj_path.is_file():
        raise SystemExit(f"OBJ not found: {obj_path}")
    output_path: Path = (
        args.output.expanduser().resolve()
        if args.output
        else obj_path.with_suffix(".yaml")
    )
    run(
        obj_path,
        output_path,
        epsilon=args.epsilon,
        planar_tol=args.planar_tol,
        merge_threshold=args.merge_threshold,
    )


if __name__ == "__main__":
    main()

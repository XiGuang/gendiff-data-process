from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from skimage import draw, measure


def cross2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


@dataclass(slots=True)
class ProxyConfig:
    weld_epsilon: float = 0.02
    degenerate_area_epsilon: float = 1e-6
    adjacency_angle_degrees: float = 8.0
    plane_distance_tolerance: float = 0.15
    patch_merge_angle_degrees: float = 5.0
    patch_merge_distance: float = 0.12
    horizontal_normal_cosine: float = 0.92
    ground_band_ratio: float = 0.08
    ground_band_min: float = 0.75
    ground_patch_min_area: float = 12.0
    ground_patch_area_ratio: float = 0.12
    min_component_area: float = 8.0
    min_component_faces: int = 32
    min_component_height: float = 2.5
    min_patch_faces: int = 8
    boundary_simplify_ratio: float = 0.015
    short_edge_ratio: float = 0.01
    regularization_angle_degrees: float = 12.0
    coordinate_snap_ratio: float = 0.015
    plane_cluster_angle_degrees: float = 7.0
    plane_cluster_distance: float = 0.2
    raster_pitch_ratio: float = 0.006
    raster_pitch_min: float = 0.12
    raster_padding_pixels: int = 4
    min_polygon_area: float = 5.0
    max_sample_points: int = 3000

    @classmethod
    def from_mapping(cls, mapping: dict[str, object] | None) -> "ProxyConfig":
        if not mapping:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        overrides = {key: value for key, value in mapping.items() if key in valid}
        return cls(**overrides)


@dataclass(slots=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray = field(repr=False)
    face_areas: np.ndarray = field(repr=False)
    face_centroids: np.ndarray = field(repr=False)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(slots=True)
class ProxyFace:
    face_id: str
    plane_normal: list[float]
    plane_offset: float
    vertices: list[list[float]]


@dataclass(slots=True)
class ProxyBuilding:
    component_id: str
    bbox: dict[str, list[float]]
    faces: list[ProxyFace]


@dataclass(slots=True)
class ProxyArtifact:
    tile_id: str
    source_obj: str
    up_axis: str
    buildings: list[ProxyBuilding]
    metrics: dict[str, float | int]
    mesh: Mesh = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "tile_id": self.tile_id,
            "source_obj": self.source_obj,
            "up_axis": self.up_axis,
            "buildings": [
                {
                    "component_id": building.component_id,
                    "bbox": building.bbox,
                    "faces": [asdict(face) for face in building.faces],
                }
                for building in self.buildings
            ],
            "metrics": self.metrics,
        }


def load_obj_mesh(path: str | Path, config: ProxyConfig | None = None) -> Mesh:
    cfg = config or ProxyConfig()
    source = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if line.startswith("f "):
                raw_indices: list[int] = []
                for token in line.split()[1:]:
                    index = int(token.split("/")[0])
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    raw_indices.append(index)
                for offset in range(1, len(raw_indices) - 1):
                    faces.append([raw_indices[0], raw_indices[offset], raw_indices[offset + 1]])

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    welded_vertices, welded_faces = weld_vertices(vertex_array, face_array, cfg.weld_epsilon)
    clean_vertices, clean_faces = remove_degenerate_faces(
        welded_vertices,
        welded_faces,
        cfg.degenerate_area_epsilon,
    )
    return make_mesh(clean_vertices, clean_faces)


def make_mesh(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(lengths[:, None], 1e-12)
    areas = lengths * 0.5
    centroids = triangles.mean(axis=1)
    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        face_normals=normals,
        face_areas=areas,
        face_centroids=centroids,
    )


def weld_vertices(vertices: np.ndarray, faces: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return vertices, faces
    quantized = np.round(vertices / epsilon).astype(np.int64)
    unique_keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    accum = np.zeros((len(unique_keys), 3), dtype=np.float64)
    counts = np.bincount(inverse)
    for axis in range(3):
        accum[:, axis] = np.bincount(inverse, weights=vertices[:, axis], minlength=len(unique_keys))
    welded_vertices = accum / np.maximum(counts[:, None], 1)
    welded_faces = inverse[faces]
    return welded_vertices, welded_faces


def remove_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    area_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices, faces
    unique_vertex_mask = np.array([len({int(a), int(b), int(c)}) == 3 for a, b, c in faces], dtype=bool)
    candidate_faces = faces[unique_vertex_mask]
    triangles = vertices[candidate_faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    keep = areas > area_epsilon
    clean_faces = candidate_faces[keep]
    used_vertices = np.unique(clean_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices))
    clean_vertices = vertices[used_vertices]
    return clean_vertices, remap[clean_faces]


def build_face_adjacency(faces: np.ndarray) -> list[list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(start), int(end))
            edge = edge if edge[0] < edge[1] else (edge[1], edge[0])
            edge_to_faces[edge].append(face_index)

    adjacency = [set() for _ in range(len(faces))]
    for shared_faces in edge_to_faces.values():
        if len(shared_faces) < 2:
            continue
        for index, face_a in enumerate(shared_faces[:-1]):
            for face_b in shared_faces[index + 1 :]:
                adjacency[face_a].add(face_b)
                adjacency[face_b].add(face_a)
    return [sorted(neighbors) for neighbors in adjacency]


def infer_up_axis(mesh: Mesh, adjacency: list[list[int]], config: ProxyConfig) -> int:
    bounds_min, bounds_max = mesh.bounds
    extents = bounds_max - bounds_min
    axis_scores: list[float] = []
    for axis in range(3):
        low_band = max(extents[axis] * config.ground_band_ratio, config.ground_band_min)
        low_limit = bounds_min[axis] + low_band
        near_horizontal = np.abs(mesh.face_normals[:, axis]) >= config.horizontal_normal_cosine
        low_triangles = near_horizontal & (mesh.face_centroids[:, axis] <= low_limit)
        cluster_area = largest_masked_component_area(low_triangles, adjacency, mesh.face_areas)
        total_horizontal_area = float(mesh.face_areas[near_horizontal].sum())
        score = cluster_area + 0.2 * total_horizontal_area - 0.05 * float(extents[axis])
        axis_scores.append(score)
    return int(np.argmax(axis_scores))


def largest_masked_component_area(mask: np.ndarray, adjacency: list[list[int]], face_areas: np.ndarray) -> float:
    visited = np.zeros(len(mask), dtype=bool)
    best = 0.0
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        total = 0.0
        while stack:
            current = stack.pop()
            total += float(face_areas[current])
            for neighbor in adjacency[current]:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        best = max(best, total)
    return best


def build_proxy_artifact(source_obj: str | Path, config: ProxyConfig | None = None) -> ProxyArtifact:
    cfg = config or ProxyConfig()
    source_path = Path(source_obj)
    tile_id = source_path.stem.removeprefix("bs_")
    mesh = load_obj_mesh(source_path, cfg)
    adjacency = build_face_adjacency(mesh.faces)
    up_axis = infer_up_axis(mesh, adjacency, cfg)
    components = extract_building_components(mesh, adjacency, up_axis, cfg)

    buildings: list[ProxyBuilding] = []
    proxy_face_total = 0
    for component_index, face_indices in enumerate(components):
        patch_groups = cluster_planar_patches(mesh, face_indices, adjacency, cfg)
        plane_clusters = cluster_patches_by_plane(mesh, patch_groups, cfg)
        proxy_faces: list[ProxyFace] = []
        face_counter = 0
        for cluster_faces in plane_clusters:
            polygons = extract_plane_cluster_polygons(mesh, cluster_faces, up_axis, cfg)
            for polygon in polygons:
                proxy_faces.append(
                    ProxyFace(
                        face_id=f"face_{component_index:03d}_{face_counter:04d}",
                        plane_normal=[round(float(value), 6) for value in polygon["plane_normal"]],
                        plane_offset=round(float(polygon["plane_offset"]), 6),
                        vertices=[[round(float(coord), 6) for coord in point] for point in polygon["vertices"]],
                    )
                )
                face_counter += 1
        if not proxy_faces:
            continue
        component_vertices = mesh.vertices[np.unique(mesh.faces[face_indices].reshape(-1))]
        buildings.append(
            ProxyBuilding(
                component_id=f"building_{component_index:03d}",
                bbox={
                    "min": [round(float(value), 6) for value in component_vertices.min(axis=0)],
                    "max": [round(float(value), 6) for value in component_vertices.max(axis=0)],
                },
                faces=proxy_faces,
            )
        )
        proxy_face_total += len(proxy_faces)

    metrics = {
        "input_triangles": int(mesh.face_count),
        "proxy_faces": int(proxy_face_total),
        "buildings_detected": int(len(buildings)),
    }
    return ProxyArtifact(
        tile_id=tile_id,
        source_obj=str(source_path),
        up_axis="xyz"[up_axis],
        buildings=buildings,
        metrics=metrics,
        mesh=mesh,
    )


def extract_building_components(
    mesh: Mesh,
    adjacency: list[list[int]],
    up_axis: int,
    config: ProxyConfig,
) -> list[np.ndarray]:
    components = face_components(np.arange(mesh.face_count, dtype=np.int64), adjacency)
    kept: list[np.ndarray] = []
    for component in components:
        if len(component) < config.min_component_faces:
            continue
        component_area = float(mesh.face_areas[component].sum())
        if component_area < config.min_component_area:
            continue
        trimmed = remove_ground_from_component(mesh, component, adjacency, up_axis, config)
        if trimmed.size == 0:
            continue
        for child in face_components(trimmed, adjacency):
            child_area = float(mesh.face_areas[child].sum())
            if len(child) < config.min_component_faces or child_area < config.min_component_area:
                continue
            child_vertices = mesh.vertices[np.unique(mesh.faces[child].reshape(-1))]
            height = float(child_vertices[:, up_axis].max() - child_vertices[:, up_axis].min())
            if height < config.min_component_height:
                continue
            kept.append(child)
    kept.sort(key=lambda component: float(mesh.face_areas[component].sum()), reverse=True)
    return kept


def remove_ground_from_component(
    mesh: Mesh,
    component: np.ndarray,
    adjacency: list[list[int]],
    up_axis: int,
    config: ProxyConfig,
) -> np.ndarray:
    component_vertices = mesh.vertices[np.unique(mesh.faces[component].reshape(-1))]
    min_up = float(component_vertices[:, up_axis].min())
    max_up = float(component_vertices[:, up_axis].max())
    if max_up - min_up < config.min_component_height:
        return np.empty(0, dtype=np.int64)

    band = max((max_up - min_up) * config.ground_band_ratio, config.ground_band_min)
    low_limit = min_up + band
    horizontal = np.abs(mesh.face_normals[component, up_axis]) >= config.horizontal_normal_cosine
    near_base = mesh.face_centroids[component, up_axis] <= low_limit
    candidate_mask = horizontal & near_base
    if not np.any(candidate_mask):
        return component

    local_candidates = component[candidate_mask]
    removal: set[int] = set()
    total_area = float(mesh.face_areas[component].sum())
    for patch in face_components(local_candidates, adjacency):
        patch_area = float(mesh.face_areas[patch].sum())
        if patch_area >= config.ground_patch_min_area or patch_area >= total_area * config.ground_patch_area_ratio:
            removal.update(int(face_index) for face_index in patch)
    if not removal:
        return component
    return np.asarray([face_index for face_index in component if int(face_index) not in removal], dtype=np.int64)


def face_components(face_indices: np.ndarray, adjacency: list[list[int]]) -> list[np.ndarray]:
    face_set = set(int(face_index) for face_index in face_indices.tolist())
    visited: set[int] = set()
    components: list[np.ndarray] = []
    for start in face_indices:
        start_index = int(start)
        if start_index in visited:
            continue
        stack = [start_index]
        visited.add(start_index)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in face_set and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def cluster_planar_patches(
    mesh: Mesh,
    face_indices: np.ndarray,
    adjacency: list[list[int]],
    config: ProxyConfig,
) -> list[np.ndarray]:
    face_set = set(int(face_index) for face_index in face_indices.tolist())
    remaining: set[int] = set(face_set)
    angle_cosine = math.cos(math.radians(config.adjacency_angle_degrees))
    patches: list[np.ndarray] = []

    while remaining:
        seed = max(remaining, key=lambda face_index: mesh.face_areas[face_index])
        remaining.remove(seed)
        seed_normal = mesh.face_normals[seed]
        seed_point = mesh.face_centroids[seed]
        queue = [seed]
        patch = [seed]
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor not in remaining:
                    continue
                alignment = abs(float(np.dot(mesh.face_normals[neighbor], seed_normal)))
                if alignment < angle_cosine:
                    continue
                distance = abs(float(np.dot(mesh.face_centroids[neighbor] - seed_point, seed_normal)))
                if distance > config.plane_distance_tolerance:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
                patch.append(neighbor)
        patches.append(np.asarray(patch, dtype=np.int64))

    merged = merge_coplanar_patches(mesh, patches, adjacency, config)
    return [patch for patch in merged if len(patch) >= config.min_patch_faces]


def merge_coplanar_patches(
    mesh: Mesh,
    patches: list[np.ndarray],
    adjacency: list[list[int]],
    config: ProxyConfig,
) -> list[np.ndarray]:
    if not patches:
        return []
    angle_cosine = math.cos(math.radians(config.patch_merge_angle_degrees))
    patch_planes = [fit_plane(mesh.vertices[np.unique(mesh.faces[patch].reshape(-1))], mesh.face_normals[patch]) for patch in patches]
    patch_lookup = np.full(mesh.face_count, -1, dtype=np.int64)
    for patch_index, patch in enumerate(patches):
        patch_lookup[patch] = patch_index

    parents = list(range(len(patches)))

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for patch_index, patch in enumerate(patches):
        normal_a, offset_a, _ = patch_planes[patch_index]
        for face_index in patch:
            for neighbor in adjacency[int(face_index)]:
                neighbor_patch = int(patch_lookup[neighbor])
                if neighbor_patch < 0 or neighbor_patch == patch_index:
                    continue
                normal_b, offset_b, _ = patch_planes[neighbor_patch]
                if abs(float(np.dot(normal_a, normal_b))) < angle_cosine:
                    continue
                if abs(offset_a - offset_b) > config.patch_merge_distance:
                    continue
                union(patch_index, neighbor_patch)

    grouped: dict[int, list[int]] = defaultdict(list)
    for patch_index, patch in enumerate(patches):
        grouped[find(patch_index)].extend(int(face_index) for face_index in patch)
    return [np.asarray(sorted(set(indices)), dtype=np.int64) for indices in grouped.values()]


def extract_patch_polygon(
    mesh: Mesh,
    patch_faces: np.ndarray,
    up_axis: int,
    config: ProxyConfig,
) -> dict[str, object] | None:
    polygons = extract_plane_cluster_polygons(mesh, patch_faces, up_axis, config)
    if not polygons:
        return None
    polygons.sort(key=lambda polygon: polygon_area_3d(np.asarray(polygon["vertices"][:-1])), reverse=True)
    return polygons[0]


def cluster_patches_by_plane(mesh: Mesh, patches: list[np.ndarray], config: ProxyConfig) -> list[np.ndarray]:
    if not patches:
        return []
    cosine = math.cos(math.radians(config.plane_cluster_angle_degrees))
    clusters: list[dict[str, object]] = []
    for patch in sorted(patches, key=len, reverse=True):
        normal, offset, _ = fit_plane(mesh.vertices[np.unique(mesh.faces[patch].reshape(-1))], mesh.face_normals[patch])
        normal, offset = canonical_plane(normal, offset)
        weight = float(mesh.face_areas[patch].sum())
        best_cluster: dict[str, object] | None = None
        best_alignment = -1.0
        for cluster in clusters:
            cluster_normal = cluster["normal"]
            cluster_offset = cluster["offset"]
            alignment = float(np.dot(normal, cluster_normal))
            if alignment < cosine:
                continue
            if abs(offset - cluster_offset) > config.plane_cluster_distance:
                continue
            if alignment > best_alignment:
                best_alignment = alignment
                best_cluster = cluster
        if best_cluster is None:
            clusters.append(
                {
                    "normal": normal.copy(),
                    "offset": float(offset),
                    "weight": weight,
                    "faces": [patch],
                }
            )
            continue
        total_weight = float(best_cluster["weight"]) + weight
        blended = (best_cluster["normal"] * float(best_cluster["weight"]) + normal * weight) / max(total_weight, 1e-12)
        blended = blended / max(float(np.linalg.norm(blended)), 1e-12)
        best_cluster["normal"] = blended
        best_cluster["offset"] = (float(best_cluster["offset"]) * float(best_cluster["weight"]) + offset * weight) / total_weight
        best_cluster["weight"] = total_weight
        best_cluster["faces"].append(patch)

    merged: list[np.ndarray] = []
    for cluster in clusters:
        merged_faces = np.concatenate(cluster["faces"]).astype(np.int64)
        merged.append(np.asarray(sorted(set(int(face_index) for face_index in merged_faces.tolist())), dtype=np.int64))
    return merged


def canonical_plane(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        return -normal, -offset
    return normal, offset


def extract_plane_cluster_polygons(
    mesh: Mesh,
    patch_faces: np.ndarray,
    up_axis: int,
    config: ProxyConfig,
) -> list[dict[str, object]]:
    plane_normal, plane_offset, plane_origin = fit_plane(
        mesh.vertices[np.unique(mesh.faces[patch_faces].reshape(-1))],
        mesh.face_normals[patch_faces],
    )
    plane_normal, plane_offset = canonical_plane(plane_normal, plane_offset)
    tangent_u, tangent_v = plane_basis(plane_normal, up_axis)
    projected_triangles = project_points(mesh.vertices[mesh.faces[patch_faces].reshape(-1)], plane_origin, tangent_u, tangent_v)
    projected_triangles = projected_triangles.reshape(-1, 3, 2)

    mask, min_xy, pitch = rasterize_projected_triangles(projected_triangles, config)
    if not np.any(mask):
        return []
    filled = binary_fill_holes(mask)
    contours = measure.find_contours(filled.astype(np.float32), 0.5)
    polygons: list[dict[str, object]] = []
    for contour in contours:
        contour_xy = np.column_stack(
            [
                contour[:, 1] * pitch + min_xy[0],
                contour[:, 0] * pitch + min_xy[1],
            ]
        )
        polygon = postprocess_contour(contour_xy, plane_origin, tangent_u, tangent_v, plane_normal, plane_offset, config)
        if polygon is not None:
            polygons.append(polygon)
    polygons.sort(key=lambda polygon: polygon_area_3d(np.asarray(polygon["vertices"][:-1])), reverse=True)
    return polygons


def rasterize_projected_triangles(
    projected_triangles: np.ndarray,
    config: ProxyConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    mins = projected_triangles.reshape(-1, 2).min(axis=0)
    maxs = projected_triangles.reshape(-1, 2).max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    pitch = max(float(extent.max()) * config.raster_pitch_ratio, config.raster_pitch_min)
    pad = config.raster_padding_pixels
    min_xy = mins - pitch * pad
    max_xy = maxs + pitch * pad
    width = int(math.ceil((max_xy[0] - min_xy[0]) / pitch)) + 3
    height = int(math.ceil((max_xy[1] - min_xy[1]) / pitch)) + 3
    mask = np.zeros((height, width), dtype=bool)
    for triangle in projected_triangles:
        cols = (triangle[:, 0] - min_xy[0]) / pitch
        rows = (triangle[:, 1] - min_xy[1]) / pitch
        rr, cc = draw.polygon(rows, cols, shape=mask.shape)
        mask[rr, cc] = True
    return mask, min_xy, pitch


def postprocess_contour(
    contour_xy: np.ndarray,
    plane_origin: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    config: ProxyConfig,
) -> dict[str, object] | None:
    cleaned_2d = dedupe_loop(contour_xy)
    cleaned_2d = ensure_ccw(cleaned_2d)
    if len(cleaned_2d) < 3:
        return None

    scale = polygon_scale(cleaned_2d)
    simplified = douglas_peucker_loop(cleaned_2d, max(scale * config.boundary_simplify_ratio, config.raster_pitch_min))
    simplified = collapse_short_edges(simplified, max(scale * config.short_edge_ratio, config.raster_pitch_min))
    regularized = regularize_polygon(simplified, config)
    regularized = collapse_short_edges(regularized, max(scale * config.short_edge_ratio, config.raster_pitch_min))
    regularized = remove_colinear_vertices(regularized, tolerance=1e-3)
    regularized = ensure_ccw(regularized)
    if len(regularized) < 3 or not is_simple_polygon(regularized):
        regularized = remove_colinear_vertices(ensure_ccw(simplified), tolerance=1e-3)
    if len(regularized) < 3 or not is_simple_polygon(regularized):
        regularized = remove_colinear_vertices(ensure_ccw(cleaned_2d), tolerance=1e-3)
    if len(regularized) < 3 or not is_simple_polygon(regularized):
        return None
    if abs(signed_area(regularized)) < config.min_polygon_area:
        return None

    vertices_3d = lift_polygon_to_plane(regularized, plane_origin, tangent_u, tangent_v, plane_normal)
    closed = np.vstack([vertices_3d, vertices_3d[0]])
    return {
        "plane_normal": plane_normal,
        "plane_offset": plane_offset,
        "vertices": closed,
    }


def polygon_area_3d(points_3d: np.ndarray) -> float:
    if len(points_3d) < 3:
        return 0.0
    origin = points_3d[0]
    area = 0.0
    for index in range(1, len(points_3d) - 1):
        area += 0.5 * float(np.linalg.norm(np.cross(points_3d[index] - origin, points_3d[index + 1] - origin)))
    return area


def extract_boundary_loops(faces: np.ndarray, patch_faces: np.ndarray) -> list[list[int]]:
    edge_counter: Counter[tuple[int, int]] = Counter()
    for face_index in patch_faces:
        face = faces[int(face_index)]
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(start), int(end))
            key = edge if edge[0] < edge[1] else (edge[1], edge[0])
            edge_counter[key] += 1

    graph: dict[int, list[int]] = defaultdict(list)
    for (start, end), count in edge_counter.items():
        if count != 1:
            continue
        graph[start].append(end)
        graph[end].append(start)

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for start, neighbors in graph.items():
        for next_vertex in neighbors:
            edge = edge_key(start, next_vertex)
            if edge in visited_edges:
                continue
            loop = [start]
            previous = None
            current = start
            candidate = next_vertex
            while True:
                visited_edges.add(edge_key(current, candidate))
                previous, current = current, candidate
                if current == start:
                    break
                loop.append(current)
                options = [neighbor for neighbor in graph[current] if neighbor != previous or len(graph[current]) == 1]
                next_candidates = [neighbor for neighbor in options if edge_key(current, neighbor) not in visited_edges]
                if not next_candidates:
                    break
                if len(next_candidates) == 1:
                    candidate = next_candidates[0]
                else:
                    candidate = min(next_candidates, key=lambda neighbor: edge_key(current, neighbor))
            if len(loop) >= 3 and loop[0] != loop[-1]:
                loops.append(loop)
    return loops


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def fit_plane(points: np.ndarray, reference_normals: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    origin = points.mean(axis=0)
    centered = points - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if reference_normals is not None and len(reference_normals):
        weighted = reference_normals.sum(axis=0)
        if np.dot(normal, weighted) < 0:
            normal = -normal
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    offset = float(np.dot(normal, origin))
    return normal, offset, origin


def plane_basis(normal: np.ndarray, up_axis: int) -> tuple[np.ndarray, np.ndarray]:
    up = np.zeros(3, dtype=np.float64)
    up[up_axis] = 1.0
    tangent = np.cross(normal, up)
    if np.linalg.norm(tangent) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, normal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = np.cross(normal, fallback)
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
    bitangent = np.cross(normal, tangent)
    bitangent = bitangent / max(float(np.linalg.norm(bitangent)), 1e-12)
    return tangent, bitangent


def project_points(points: np.ndarray, origin: np.ndarray, tangent_u: np.ndarray, tangent_v: np.ndarray) -> np.ndarray:
    centered = points - origin
    return np.column_stack([centered @ tangent_u, centered @ tangent_v])


def signed_area(points_2d: np.ndarray) -> float:
    if points_2d.ndim != 2 or points_2d.shape[0] < 3:
        return 0.0
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_scale(points_2d: np.ndarray) -> float:
    extent = points_2d.max(axis=0) - points_2d.min(axis=0)
    return float(max(extent.max(), 1.0))


def dedupe_loop(points_2d: np.ndarray) -> np.ndarray:
    if points_2d.ndim != 2 or len(points_2d) == 0:
        return np.empty((0, 2), dtype=np.float64)
    deduped: list[np.ndarray] = []
    for point in points_2d:
        if not deduped or np.linalg.norm(point - deduped[-1]) > 1e-8:
            deduped.append(point)
    if len(deduped) >= 2 and np.linalg.norm(deduped[0] - deduped[-1]) < 1e-8:
        deduped.pop()
    return np.asarray(deduped, dtype=np.float64)


def ensure_ccw(points_2d: np.ndarray) -> np.ndarray:
    if points_2d.ndim != 2 or len(points_2d) < 3:
        return np.asarray(points_2d, dtype=np.float64).reshape((-1, 2)) if np.asarray(points_2d).size else np.empty((0, 2), dtype=np.float64)
    return points_2d if signed_area(points_2d) >= 0 else points_2d[::-1]


def douglas_peucker_loop(points_2d: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points_2d) <= 3 or tolerance <= 0:
        return points_2d

    closed = np.vstack([points_2d, points_2d[0]])
    keep = douglas_peucker_indices(closed, tolerance)
    result = closed[sorted(set(index for index in keep if index < len(points_2d)))]
    return result if len(result) >= 3 else points_2d


def douglas_peucker_indices(points: np.ndarray, tolerance: float) -> set[int]:
    if len(points) <= 2:
        return {0, len(points) - 1}
    start = points[0]
    end = points[-1]
    line = end - start
    length = float(np.linalg.norm(line))
    if length < 1e-12:
        distances = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        distances = np.abs(
            np.array([cross2d(line, point - start) for point in points[1:-1]], dtype=np.float64) / length
        )
    if len(distances) == 0:
        return {0, len(points) - 1}
    split = int(np.argmax(distances)) + 1
    if distances[split - 1] <= tolerance:
        return {0, len(points) - 1}
    left = douglas_peucker_indices(points[: split + 1], tolerance)
    right = douglas_peucker_indices(points[split:], tolerance)
    return left | {index + split for index in right}


def collapse_short_edges(points_2d: np.ndarray, threshold: float) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2:
        return np.empty((0, 2), dtype=np.float64)
    if len(points_2d) <= 3:
        return points_2d
    changed = True
    current = points_2d.copy()
    while changed and len(current) > 3:
        changed = False
        updated: list[np.ndarray] = []
        total = len(current)
        for index in range(total):
            point = current[index]
            nxt = current[(index + 1) % total]
            if np.linalg.norm(nxt - point) < threshold and total - len(updated) > 3:
                changed = True
                continue
            updated.append(point)
        current = np.asarray(updated, dtype=np.float64).reshape((-1, 2)) if updated else np.empty((0, 2), dtype=np.float64)
    return current


def regularize_polygon(points_2d: np.ndarray, config: ProxyConfig) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2:
        return np.empty((0, 2), dtype=np.float64)
    if len(points_2d) <= 3:
        return points_2d
    directions = np.roll(points_2d, -1, axis=0) - points_2d
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 1e-8
    if not np.any(valid):
        return points_2d
    angles = np.mod(np.arctan2(directions[valid, 1], directions[valid, 0]), math.pi / 2.0)
    weighted_index = int(np.argmax(lengths[valid]))
    dominant_angle = float(angles[weighted_index])
    cos_a = math.cos(-dominant_angle)
    sin_a = math.sin(-dominant_angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (rotation @ points_2d.T).T
    snap_threshold = math.radians(config.regularization_angle_degrees)
    coord_snap = max(polygon_scale(points_2d) * config.coordinate_snap_ratio, 0.02)
    adjusted = rotated.copy()

    for index in range(len(adjusted)):
        current = adjusted[index]
        nxt = adjusted[(index + 1) % len(adjusted)]
        delta = nxt - current
        angle = abs(math.atan2(delta[1], delta[0]))
        angle = min(angle, abs((math.pi / 2.0) - angle))
        if angle <= snap_threshold:
            if abs(delta[0]) >= abs(delta[1]):
                adjusted[(index + 1) % len(adjusted), 1] = current[1]
            else:
                adjusted[(index + 1) % len(adjusted), 0] = current[0]

    for axis in range(2):
        values = adjusted[:, axis].copy()
        for index in range(len(values)):
            close = np.abs(values - values[index]) <= coord_snap
            if close.sum() > 1:
                values[close] = np.mean(values[close])
        adjusted[:, axis] = values

    inverse = rotation.T
    return (inverse @ adjusted.T).T


def remove_colinear_vertices(points_2d: np.ndarray, tolerance: float) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.ndim != 2:
        return np.empty((0, 2), dtype=np.float64)
    if len(points_2d) <= 3:
        return points_2d
    kept: list[np.ndarray] = []
    total = len(points_2d)
    for index in range(total):
        previous = points_2d[index - 1]
        current = points_2d[index]
        nxt = points_2d[(index + 1) % total]
        left = current - previous
        right = nxt - current
        cross = abs(cross2d(left, right))
        if cross <= tolerance and np.dot(left, right) >= 0:
            continue
        kept.append(current)
    if len(kept) < 3:
        return points_2d
    return np.asarray(kept, dtype=np.float64)


def is_simple_polygon(points_2d: np.ndarray) -> bool:
    total = len(points_2d)
    if total < 3:
        return False
    for index in range(total):
        a1 = points_2d[index]
        a2 = points_2d[(index + 1) % total]
        for other in range(index + 1, total):
            if abs(index - other) <= 1 or (index == 0 and other == total - 1):
                continue
            b1 = points_2d[other]
            b2 = points_2d[(other + 1) % total]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return abs(signed_area(points_2d)) > 1e-6


def segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return cross2d(q - p, r - p)

    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    return (o1 == 0.0 or o2 == 0.0 or o3 == 0.0 or o4 == 0.0 or (o1 > 0) != (o2 > 0)) and (
        (o3 > 0) != (o4 > 0)
    )


def lift_polygon_to_plane(
    points_2d: np.ndarray,
    origin: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    projected = origin + np.outer(points_2d[:, 0], tangent_u) + np.outer(points_2d[:, 1], tangent_v)
    plane_offset = float(np.dot(plane_normal, origin))
    distances = (projected @ plane_normal) - plane_offset
    return projected - np.outer(distances, plane_normal)


def triangulate_polygon(points_2d: np.ndarray) -> list[tuple[int, int, int]]:
    polygon = ensure_ccw(dedupe_loop(points_2d))
    if len(polygon) < 3:
        return []
    indices = list(range(len(polygon)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < len(polygon) * len(polygon):
        ear_found = False
        for offset, current in enumerate(indices):
            previous = indices[offset - 1]
            nxt = indices[(offset + 1) % len(indices)]
            triangle = polygon[[previous, current, nxt]]
            if np.cross(triangle[1] - triangle[0], triangle[2] - triangle[1]) <= 1e-10:
                continue
            contains_point = False
            for candidate in indices:
                if candidate in {previous, current, nxt}:
                    continue
                if point_in_triangle(polygon[candidate], triangle):
                    contains_point = True
                    break
            if contains_point:
                continue
            triangles.append((previous, current, nxt))
            del indices[offset]
            ear_found = True
            break
        if not ear_found:
            break
        guard += 1
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    if not triangles:
        triangles = [(0, index, index + 1) for index in range(1, len(polygon) - 1)]
    return triangles


def point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= 0 and v >= 0 and (u + v) <= 1


def write_proxy_outputs(
    artifact: ProxyArtifact,
    output_root: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    out_dir = Path(output_root) / artifact.tile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{artifact.tile_id}.proxy.json"
    obj_path = out_dir / f"{artifact.tile_id}.proxy.obj"
    metrics_path = out_dir / f"{artifact.tile_id}.metrics.json"
    if not overwrite:
        existing = [path for path in (json_path, obj_path, metrics_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    json_path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(artifact.metrics, indent=2), encoding="utf-8")
    obj_path.write_text(proxy_artifact_to_obj(artifact), encoding="utf-8")
    return json_path, obj_path, metrics_path


def proxy_artifact_to_obj(artifact: ProxyArtifact) -> str:
    lines = [f"# polygon proxy for {artifact.tile_id}"]
    vertex_offset = 1
    for building in artifact.buildings:
        lines.append(f"o {building.component_id}")
        for face in building.faces:
            points_3d = np.asarray(face.vertices[:-1], dtype=np.float64)
            for point in points_3d:
                lines.append(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
            face_indices = " ".join(str(vertex_offset + index) for index in range(len(points_3d)))
            lines.append(f"f {face_indices}")
            vertex_offset += len(points_3d)
    lines.append("")
    return "\n".join(lines)


def batch_convert(
    input_root: str | Path,
    output_root: str | Path,
    config: ProxyConfig | None = None,
    tiles: Iterable[str] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    cfg = config or ProxyConfig()
    input_path = Path(input_root)
    selected = set(tiles or [])
    tile_sources: list[Path] = []
    for tile_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
        if selected and tile_dir.name not in selected:
            continue
        source = tile_dir / f"bs_{tile_dir.name}.obj"
        if source.exists():
            tile_sources.append(source)

    jobs = [(str(source), str(output_root), cfg, overwrite) for source in tile_sources]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_batch_worker, jobs))
    return [_batch_worker(job) for job in jobs]


def _batch_worker(job: tuple[str, str, ProxyConfig, bool]) -> dict[str, object]:
    source_obj, output_root, config, overwrite = job
    artifact = build_proxy_artifact(source_obj, config)
    json_path, obj_path, metrics_path = write_proxy_outputs(artifact, output_root, overwrite=overwrite)
    result = artifact.metrics.copy()
    result.update(
        {
            "tile_id": artifact.tile_id,
            "json_path": str(json_path),
            "obj_path": str(obj_path),
            "metrics_path": str(metrics_path),
        }
    )
    return result


def load_config_file(path: str | Path | None) -> ProxyConfig:
    if path is None:
        return ProxyConfig()
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return ProxyConfig.from_mapping(json.loads(text))
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML config requested but PyYAML is not installed") from exc
    return ProxyConfig.from_mapping(yaml.safe_load(text))


def sample_mesh_points(mesh: Mesh, sample_count: int, seed: int = 0) -> np.ndarray:
    if mesh.face_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    weights = mesh.face_areas / np.maximum(mesh.face_areas.sum(), 1e-12)
    choices = rng.choice(mesh.face_count, size=min(sample_count, mesh.face_count), replace=True, p=weights)
    triangles = mesh.vertices[mesh.faces[choices]]
    r1 = np.sqrt(rng.random(len(choices)))
    r2 = rng.random(len(choices))
    points = (
        (1 - r1)[:, None] * triangles[:, 0]
        + (r1 * (1 - r2))[:, None] * triangles[:, 1]
        + (r1 * r2)[:, None] * triangles[:, 2]
    )
    return points


def approximation_error(source_mesh: Mesh, proxy_mesh: Mesh, sample_count: int) -> float:
    source_points = sample_mesh_points(source_mesh, sample_count, seed=0)
    proxy_points = sample_mesh_points(proxy_mesh, sample_count, seed=1)
    if len(source_points) == 0 or len(proxy_points) == 0:
        return math.inf
    tree = cKDTree(proxy_points)
    distances, _ = tree.query(source_points, workers=-1)
    return float(np.mean(distances))

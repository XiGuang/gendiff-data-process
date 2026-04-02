from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import trimesh
import yaml
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from obj_to_language.yaml_to_obj_new import yaml_entries_to_mesh

EPS = 1e-6


@dataclass(frozen=True)
class ProxyConfig:
    grid_pitch: float = 0.35
    height_bin: float = 0.5
    max_levels: int = 4
    min_stage_height: float = 1.5
    min_component_area: float = 4.0
    min_component_faces: int = 40
    min_fragment_projected_area: float = 1.0
    stage_support_ratio: float = 0.03
    cluster_gap: float = 1.0
    padding_cells: int = 1
    simplify_tolerance_factor: float = 0.5
    edge_smooth_radius_factor: float = 0.75
    keypoint_snap_tolerance_factor: float = 0.75
    curve_preserve_angle_deg: float = 18.0
    ray_batch_size: int = 8192


@dataclass
class HeightField:
    min_x: float
    min_z: float
    pitch: float
    base_y: float
    top_y: float
    occupancy: np.ndarray
    top_heights: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape


@dataclass
class ProxyArtifact:
    source_obj: Path
    entries: list[dict]
    mesh: trimesh.Trimesh | None
    levels: list[float]
    config: ProxyConfig


def load_obj_mesh(obj_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {obj_path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh payload for {obj_path}: {type(loaded)!r}")
    return loaded


def projected_area_xz(mesh: trimesh.Trimesh) -> float:
    triangles = mesh.vertices[mesh.faces][:, :, (0, 2)]
    cross = (
        triangles[:, 0, 0] * (triangles[:, 1, 1] - triangles[:, 2, 1])
        + triangles[:, 1, 0] * (triangles[:, 2, 1] - triangles[:, 0, 1])
        + triangles[:, 2, 0] * (triangles[:, 0, 1] - triangles[:, 1, 1])
    )
    return float(0.5 * np.abs(cross).sum())


def clean_mesh(mesh: trimesh.Trimesh, config: ProxyConfig) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    cleaned.merge_vertices(merge_tex=True, merge_norm=True)

    face_areas = cleaned.area_faces
    valid_faces = np.isfinite(face_areas) & (face_areas > 1e-10)
    cleaned.update_faces(valid_faces)
    cleaned.remove_unreferenced_vertices()

    parts = cleaned.split(only_watertight=False)
    kept: list[trimesh.Trimesh] = []
    for part in parts:
        if len(part.faces) == 0:
            continue
        area = projected_area_xz(part)
        if (
            len(part.faces) < config.min_component_faces
            and area < config.min_fragment_projected_area
        ):
            continue
        kept.append(part)

    if not kept:
        return cleaned
    if len(kept) == 1:
        return kept[0]
    return trimesh.util.concatenate(kept)


def _batched_ray_top_heights(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    directions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    top_hits = np.full(len(origins), -np.inf, dtype=float)
    for start in range(0, len(origins), batch_size):
        end = min(len(origins), start + batch_size)
        locations, ray_idx, _ = mesh.ray.intersects_location(
            origins[start:end],
            directions[start:end],
            multiple_hits=True,
        )
        if len(locations) == 0:
            continue
        maxima = np.full(end - start, -np.inf, dtype=float)
        np.maximum.at(maxima, ray_idx, locations[:, 1])
        valid = maxima > -np.inf
        top_hits[start:end][valid] = maxima[valid]
    top_hits[top_hits == -np.inf] = np.nan
    return top_hits


def build_heightfield(
    mesh: trimesh.Trimesh,
    config: ProxyConfig,
    *,
    min_x: float | None = None,
    max_x: float | None = None,
    min_z: float | None = None,
    max_z: float | None = None,
) -> HeightField:
    pitch = config.grid_pitch
    bounds = mesh.bounds
    pad = pitch * config.padding_cells
    min_x = float(bounds[0, 0] - pad if min_x is None else min_x)
    max_x = float(bounds[1, 0] + pad if max_x is None else max_x)
    min_z = float(bounds[0, 2] - pad if min_z is None else min_z)
    max_z = float(bounds[1, 2] + pad if max_z is None else max_z)
    top_y = float(bounds[1, 1])
    base_y = float(bounds[0, 1])

    width = max(1, int(math.ceil((max_x - min_x) / pitch)))
    height = max(1, int(math.ceil((max_z - min_z) / pitch)))
    x_centers = min_x + (np.arange(width, dtype=float) + 0.5) * pitch
    z_centers = min_z + (np.arange(height, dtype=float) + 0.5) * pitch
    grid_x, grid_z = np.meshgrid(x_centers, z_centers)

    origins = np.column_stack(
        [
            grid_x.ravel(),
            np.full(grid_x.size, top_y + max(1.0, pitch * 2.0), dtype=float),
            grid_z.ravel(),
        ]
    )
    directions = np.zeros_like(origins)
    directions[:, 1] = -1.0

    top_hits = _batched_ray_top_heights(mesh, origins, directions, config.ray_batch_size)
    occupancy = np.isfinite(top_hits).reshape(height, width)
    top_heights = top_hits.reshape(height, width)
    return HeightField(
        min_x=min_x,
        min_z=min_z,
        pitch=pitch,
        base_y=base_y,
        top_y=top_y,
        occupancy=occupancy,
        top_heights=top_heights,
    )


def _fill_heights_nearest(top_heights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(top_heights)
    if not valid.any():
        return top_heights.copy()
    nearest_indices = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )
    return top_heights[tuple(nearest_indices)]


def regularize_heightfield(heightfield: HeightField, config: ProxyConfig) -> HeightField:
    structure = np.ones((3, 3), dtype=bool)
    occupancy = ndimage.binary_closing(heightfield.occupancy, structure=structure)
    occupancy = ndimage.binary_opening(occupancy, structure=structure)
    occupancy = ndimage.binary_fill_holes(occupancy)

    min_cells = max(1, int(math.ceil(config.min_component_area / (config.grid_pitch ** 2))))
    labels, count = ndimage.label(occupancy)
    if count > 0:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_cells
        keep[0] = False
        occupancy = keep[labels]

    nearest = _fill_heights_nearest(heightfield.top_heights)
    median = ndimage.median_filter(nearest, size=3, mode="nearest")
    top_heights = np.where(occupancy, median, np.nan)

    return HeightField(
        min_x=heightfield.min_x,
        min_z=heightfield.min_z,
        pitch=heightfield.pitch,
        base_y=heightfield.base_y,
        top_y=heightfield.top_y,
        occupancy=occupancy,
        top_heights=top_heights,
    )


def _snap_height(value: float, height_bin: float) -> float:
    return round(value / height_bin) * height_bin


def _merge_cluster_pair(clusters: list[dict], idx: int, height_bin: float) -> list[dict]:
    left = clusters[idx]
    right = clusters[idx + 1]
    total = left["weight"] + right["weight"]
    center = ((left["center"] * left["weight"]) + (right["center"] * right["weight"])) / total
    merged = {
        "center": _snap_height(center, height_bin),
        "weight": total,
        "peak": max(left.get("peak", left["center"]), right.get("peak", right["center"])),
    }
    return clusters[:idx] + [merged] + clusters[idx + 2 :]


def select_stage_levels(heightfield: HeightField, config: ProxyConfig) -> list[float]:
    occupied = heightfield.top_heights[heightfield.occupancy]
    if occupied.size == 0:
        return []

    total_height = float(occupied.max() - heightfield.base_y)
    if total_height <= config.min_stage_height + EPS:
        return [float(occupied.max())]

    quantized = np.array(
        [_snap_height(float(v), config.height_bin) for v in occupied],
        dtype=float,
    )
    values, counts = np.unique(quantized, return_counts=True)
    support = counts / counts.sum()
    candidate_indices: list[int] = []
    for idx, ratio in enumerate(support.tolist()):
        if ratio < config.stage_support_ratio:
            continue
        left = counts[idx - 1] if idx > 0 else -1
        right = counts[idx + 1] if idx + 1 < len(counts) else -1
        if counts[idx] >= left and counts[idx] >= right:
            candidate_indices.append(idx)

    if not candidate_indices:
        candidate_indices = np.flatnonzero(support >= config.stage_support_ratio).tolist()
    if not candidate_indices:
        return [float(occupied.max())]

    clusters: list[dict] = []
    for idx in candidate_indices:
        level = float(values[idx])
        weight = int(counts[idx])
        if not clusters:
            clusters.append({"center": level, "weight": weight, "peak": level})
            continue
        if level - clusters[-1]["peak"] <= config.cluster_gap + EPS:
            total = clusters[-1]["weight"] + weight
            center = (
                (clusters[-1]["center"] * clusters[-1]["weight"]) + (level * weight)
            ) / total
            clusters[-1] = {
                "center": _snap_height(center, config.height_bin),
                "weight": total,
                "peak": max(clusters[-1]["peak"], level),
            }
            continue
        clusters.append({"center": level, "weight": weight, "peak": level})

    max_top = float(occupied.max())
    top_bin = _snap_height(max_top, config.height_bin)
    min_top_cells = max(
        1,
        int(math.ceil(0.75 * config.min_component_area / (config.grid_pitch ** 2))),
    )
    high_support_cells = int((occupied >= top_bin - config.height_bin).sum())
    reserve_top_stage = (
        high_support_cells >= min_top_cells
        and top_bin - max(cluster["center"] for cluster in clusters) > config.height_bin + EPS
    )

    keep_count = config.max_levels - int(reserve_top_stage)
    if len(clusters) > keep_count and keep_count > 0:
        lowest_cluster = min(clusters, key=lambda item: item["center"])
        preserved = [lowest_cluster]
        remaining = [item for item in clusters if item is not lowest_cluster]
        if keep_count >= 2 and remaining:
            highest_cluster = max(remaining, key=lambda item: item["center"])
            preserved.append(highest_cluster)
            remaining = [item for item in remaining if item is not highest_cluster]
        remaining.sort(key=lambda item: item["weight"], reverse=True)
        clusters = preserved + remaining[: max(0, keep_count - len(preserved))]
    else:
        clusters.sort(key=lambda item: item["weight"], reverse=True)
    clusters.sort(key=lambda item: item["center"])
    while len(clusters) > keep_count:
        gaps = [
            clusters[i + 1]["center"] - clusters[i]["center"]
            for i in range(len(clusters) - 1)
        ]
        merge_idx = int(np.argmin(gaps))
        clusters = _merge_cluster_pair(clusters, merge_idx, config.height_bin)

    while True:
        levels = [float(c["center"]) for c in clusters]
        if not levels:
            break
        first_gap = levels[0] - heightfield.base_y
        gaps = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
        too_small = first_gap < config.min_stage_height - EPS or any(
            gap < config.min_stage_height - EPS for gap in gaps
        )
        if not too_small or len(clusters) == 1:
            break

        candidates: list[tuple[float, int]] = [(first_gap, 0)]
        candidates.extend((gap, i) for i, gap in enumerate(gaps))
        gap, idx = min(candidates, key=lambda item: item[0])
        if idx == 0 and gap == first_gap and len(clusters) > 1:
            clusters = _merge_cluster_pair(clusters, 0, config.height_bin)
        elif idx < len(clusters) - 1:
            clusters = _merge_cluster_pair(clusters, idx, config.height_bin)
        else:
            break

    levels = sorted({_snap_height(float(c["center"]), config.height_bin) for c in clusters})
    levels = [min(level, max_top) for level in levels if level - heightfield.base_y > EPS]
    if not levels:
        return [max_top]
    if reserve_top_stage:
        levels.append(min(top_bin, max_top))
    deduped: list[float] = []
    for level in levels:
        if deduped and abs(level - deduped[-1]) < EPS:
            continue
        deduped.append(level)
    return deduped


def _turn_angle(
    prev_point: tuple[float, float],
    point: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    v1 = np.array([point[0] - prev_point[0], point[1] - prev_point[1]], dtype=float)
    v2 = np.array([next_point[0] - point[0], next_point[1] - point[1]], dtype=float)
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 <= EPS or norm2 <= EPS:
        return 0.0
    cos_angle = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
    return math.acos(cos_angle)


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    start_vec = np.array(start, dtype=float)
    end_vec = np.array(end, dtype=float)
    point_vec = np.array(point, dtype=float)
    chord = end_vec - start_vec
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= EPS:
        return float(np.linalg.norm(point_vec - start_vec))
    cross = abs(np.cross(chord, point_vec - start_vec))
    return float(cross / chord_norm)


def _simplify_ring(
    coords: Sequence[tuple[float, float]],
    config: ProxyConfig,
) -> list[tuple[float, float]]:
    if len(coords) <= 3:
        return list(coords)

    def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    cleaned: list[tuple[float, float]] = []
    for point in coords:
        if cleaned and math.isclose(point[0], cleaned[-1][0], abs_tol=EPS) and math.isclose(
            point[1], cleaned[-1][1], abs_tol=EPS
        ):
            continue
        cleaned.append(point)

    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        result: list[tuple[float, float]] = []
        count = len(cleaned)
        deviation_tol = config.grid_pitch * config.keypoint_snap_tolerance_factor
        angle_tol = math.radians(config.curve_preserve_angle_deg)
        for idx in range(count):
            prev_point = cleaned[idx - 1]
            point = cleaned[idx]
            next_point = cleaned[(idx + 1) % count]
            if abs(cross(prev_point, point, next_point)) <= 1e-8:
                changed = True
                continue
            turn_angle = _turn_angle(prev_point, point, next_point)
            deviation = _point_line_distance(point, prev_point, next_point)
            if turn_angle <= angle_tol and deviation <= deviation_tol:
                changed = True
                continue
            result.append(point)
        cleaned = result
    return cleaned


def _iter_polygons(geometry: Polygon | MultiPolygon) -> Iterable[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)


def _geometry_to_polygons(
    geometry: Polygon | MultiPolygon,
    config: ProxyConfig,
    min_area: float,
) -> list[Polygon]:
    simplify_tol = config.grid_pitch * config.simplify_tolerance_factor
    smooth_radius = config.grid_pitch * config.edge_smooth_radius_factor
    polygons: list[Polygon] = []
    for poly in _iter_polygons(geometry):
        if poly.area < min_area:
            continue
        smoothed = poly
        if smooth_radius > EPS:
            candidate = poly.buffer(smooth_radius, join_style=1).buffer(-smooth_radius, join_style=1)
            if not candidate.is_empty:
                smoothed = candidate
        simple = smoothed.simplify(simplify_tol, preserve_topology=True).buffer(0)
        for component in _iter_polygons(simple):
            outer = Polygon(component.exterior)
            if outer.area < min_area:
                continue
            ring = _simplify_ring(list(outer.exterior.coords)[:-1], config)
            if len(ring) < 3:
                continue
            candidate = Polygon(ring)
            if not candidate.is_valid:
                candidate = candidate.buffer(0)
            if candidate.is_empty or candidate.area < min_area:
                continue
            if not candidate.exterior.is_ccw:
                candidate = Polygon(list(candidate.exterior.coords)[::-1])
            polygons.append(candidate)
    return polygons


def mesh_projection_to_polygons(
    mesh: trimesh.Trimesh,
    config: ProxyConfig,
    min_area: float | None = None,
) -> list[Polygon]:
    min_area = config.min_component_area if min_area is None else min_area
    triangles = mesh.vertices[mesh.faces][:, :, (0, 2)]
    polys = []
    for tri in triangles:
        poly = Polygon(tri)
        if poly.is_valid and poly.area > 1e-8:
            polys.append(poly)
    if not polys:
        return []
    merged = unary_union(polys)
    return _geometry_to_polygons(merged, config, min_area)


def mask_to_polygons(
    mask: np.ndarray,
    heightfield: HeightField,
    config: ProxyConfig,
    min_area: float | None = None,
) -> list[Polygon]:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return []

    pitch = heightfield.pitch
    boxes = [
        box(
            heightfield.min_x + col * pitch,
            heightfield.min_z + row * pitch,
            heightfield.min_x + (col + 1) * pitch,
            heightfield.min_z + (row + 1) * pitch,
        )
        for row, col in zip(rows.tolist(), cols.tolist())
    ]
    merged = unary_union(boxes)
    min_area = config.min_component_area if min_area is None else min_area
    return _geometry_to_polygons(merged, config, min_area)


def generate_proxy_entries(mesh: trimesh.Trimesh, config: ProxyConfig) -> tuple[list[dict], list[float]]:
    cleaned = clean_mesh(mesh, config)
    heightfield = regularize_heightfield(build_heightfield(cleaned, config), config)
    levels = select_stage_levels(heightfield, config)
    if not levels:
        return [], []

    entries: list[dict] = []
    stage_base = heightfield.base_y
    level_index = 0
    proxy_id = 0
    for idx, level in enumerate(levels):
        if level - stage_base <= EPS:
            continue
        if idx == 0:
            polygons = mesh_projection_to_polygons(cleaned, config, min_area=config.min_component_area)
        else:
            mask = heightfield.occupancy & (heightfield.top_heights >= level - config.height_bin)
            stage_min_area = config.min_component_area
            if idx == len(levels) - 1:
                stage_min_area = max(0.75 * config.min_component_area, config.grid_pitch ** 2)
            polygons = mask_to_polygons(mask, heightfield, config, min_area=stage_min_area)
        if not polygons:
            continue
        height = float(level - stage_base)
        for poly in polygons:
            contour = [
                [float(x), float(stage_base), float(z)]
                for x, z in list(poly.exterior.coords)[:-1]
            ]
            entries.append(
                {
                    "proxy_id": proxy_id,
                    "level_index": level_index,
                    "base_height": float(stage_base),
                    "height": height,
                    "footprint": contour,
                }
            )
            proxy_id += 1
        stage_base = float(level)
        level_index += 1
    return entries, [float(level) for level in levels[:level_index]]


def build_proxy_artifact(obj_path: Path, config: ProxyConfig) -> ProxyArtifact:
    mesh = load_obj_mesh(obj_path)
    entries, levels = generate_proxy_entries(mesh, config)
    proxy_mesh = yaml_entries_to_mesh(entries) if entries else None
    return ProxyArtifact(
        source_obj=obj_path,
        entries=entries,
        mesh=proxy_mesh,
        levels=levels,
        config=config,
    )


def write_proxy_outputs(
    artifact: ProxyArtifact,
    output_dir: Path,
    source_tile_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{artifact.source_obj.stem}.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(artifact.entries, handle, sort_keys=False)

    if artifact.mesh is not None:
        artifact.mesh.export(output_dir / artifact.source_obj.name)

    metadata = {
        "source_obj": str(artifact.source_obj),
        "levels": artifact.levels,
        "config": asdict(artifact.config),
        "proxy_count": len(artifact.entries),
    }
    with (output_dir / "polygon_proxy_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    if source_tile_dir is not None:
        data_yaml = source_tile_dir / "data.yaml"
        if data_yaml.exists():
            shutil.copy2(data_yaml, output_dir / "data.yaml")


def process_obj_to_directory(obj_path: Path, output_dir: Path, config: ProxyConfig) -> ProxyArtifact:
    artifact = build_proxy_artifact(obj_path, config)
    write_proxy_outputs(artifact, output_dir, obj_path.parent)
    return artifact


def _shared_bounds(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, pitch: float) -> tuple[float, float, float, float]:
    min_x = float(min(mesh_a.bounds[0, 0], mesh_b.bounds[0, 0]) - pitch)
    max_x = float(max(mesh_a.bounds[1, 0], mesh_b.bounds[1, 0]) + pitch)
    min_z = float(min(mesh_a.bounds[0, 2], mesh_b.bounds[0, 2]) - pitch)
    max_z = float(max(mesh_a.bounds[1, 2], mesh_b.bounds[1, 2]) + pitch)
    return min_x, max_x, min_z, max_z


def compare_meshes_topdown(
    source_mesh: trimesh.Trimesh,
    proxy_mesh: trimesh.Trimesh,
    pitch: float,
) -> dict:
    min_x, max_x, min_z, max_z = _shared_bounds(source_mesh, proxy_mesh, pitch)
    source_config = ProxyConfig(grid_pitch=pitch)
    proxy_config = ProxyConfig(grid_pitch=pitch)

    source_hf = regularize_heightfield(
        build_heightfield(
            source_mesh,
            source_config,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
        ),
        source_config,
    )
    proxy_hf = regularize_heightfield(
        build_heightfield(
            proxy_mesh,
            proxy_config,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
        ),
        proxy_config,
    )

    source_occ = source_hf.occupancy
    proxy_occ = proxy_hf.occupancy
    union = source_occ | proxy_occ
    intersection = source_occ & proxy_occ
    iou = float(intersection.sum() / union.sum()) if union.any() else 1.0

    proxy_base = float(proxy_mesh.bounds[0, 1])
    proxy_top = np.where(proxy_occ, proxy_hf.top_heights, proxy_base)
    if source_occ.any():
        mae = float(np.mean(np.abs(source_hf.top_heights[source_occ] - proxy_top[source_occ])))
    else:
        mae = 0.0

    bbox_delta_xz = float(
        max(
            abs(source_mesh.bounds[0, 0] - proxy_mesh.bounds[0, 0]),
            abs(source_mesh.bounds[1, 0] - proxy_mesh.bounds[1, 0]),
            abs(source_mesh.bounds[0, 2] - proxy_mesh.bounds[0, 2]),
            abs(source_mesh.bounds[1, 2] - proxy_mesh.bounds[1, 2]),
        )
    )
    bbox_delta_y = float(
        max(
            abs(source_mesh.bounds[0, 1] - proxy_mesh.bounds[0, 1]),
            abs(source_mesh.bounds[1, 1] - proxy_mesh.bounds[1, 1]),
        )
    )

    return {
        "footprint_iou": iou,
        "top_height_mae": mae,
        "bbox_delta_xz": bbox_delta_xz,
        "bbox_delta_y": bbox_delta_y,
        "source_proxy_count": int(source_occ.sum()),
        "proxy_proxy_count": int(proxy_occ.sum()),
    }

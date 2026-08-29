from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isqrt

import fpsample
import numpy as np
from shapely import set_precision
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from .config import ConditionConfig
from .errors import CanonicalizationError
from .polygon import polygon_from_ring
from .quantize import quantize_scalar
from .serialize import canonical_hash
from .types import CanonicalStage

Point3Q = tuple[int, int, int]


@dataclass(frozen=True)
class CanonicalCondition:
    points_q: tuple[Point3Q, ...]
    condition_hash: str
    seed_hex: str
    primitive_count: int
    candidate_count: int


@dataclass(frozen=True)
class _HorizontalPrimitive:
    y_q: int
    polygon: Polygon
    descriptor: tuple
    weight2_q2: int


@dataclass(frozen=True)
class _VerticalPrimitive:
    lower_q: int
    upper_q: int
    start_q: tuple[int, int]
    end_q: tuple[int, int]
    descriptor: tuple
    weight2_q2: int


SurfacePrimitive = _HorizontalPrimitive | _VerticalPrimitive


def _polygons(geometry) -> tuple[Polygon, ...]:
    if geometry is None or geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        output: list[Polygon] = []
        for item in geometry.geoms:
            output.extend(_polygons(item))
        return tuple(output)
    return ()


def _union_stage_at_height(stage: CanonicalStage, lower_q: int, upper_q: int):
    polygons = [
        polygon_from_ring(layer.footprint_q)
        for layer in stage.layers
        if layer.min_height_q <= lower_q and layer.max_height_q >= upper_q
    ]
    if not polygons:
        return None
    return set_precision(unary_union(polygons), grid_size=1.0, mode="valid_output")


def _difference(left, right):
    if left is None or left.is_empty:
        return None
    if right is None or right.is_empty:
        return left
    return set_precision(left.difference(right), grid_size=1.0, mode="valid_output")


def _addition_slices(source: CanonicalStage, target: CanonicalStage):
    heights = sorted(
        {
            value
            for stage in (source, target)
            for layer in stage.layers
            for value in (layer.min_height_q, layer.max_height_q)
        }
    )
    slices = []
    for lower_q, upper_q in zip(heights, heights[1:]):
        target_geometry = _union_stage_at_height(target, lower_q, upper_q)
        source_geometry = _union_stage_at_height(source, lower_q, upper_q)
        addition = _difference(target_geometry, source_geometry)
        removal = _difference(source_geometry, target_geometry)
        if removal is not None and not removal.is_empty and removal.area > 0:
            raise CanonicalizationError(
                "E_CONSTRUCTION_REMOVAL",
                "condition 输入包含 removal solid",
                lower_q=lower_q,
                upper_q=upper_q,
            )
        if addition is not None and not addition.is_empty and addition.area > 0:
            slices.append((lower_q, upper_q, addition))
    return tuple(slices)


def _horizontal_primitives(y_q: int, geometry, face: str) -> list[_HorizontalPrimitive]:
    output = []
    for polygon_index, polygon in enumerate(_polygons(geometry)):
        weight2 = int(round(polygon.area * 2))
        if weight2 <= 0:
            continue
        bounds = tuple(int(round(value)) for value in polygon.bounds)
        descriptor = ("horizontal", face, y_q, bounds, polygon_index)
        output.append(_HorizontalPrimitive(y_q, polygon, descriptor, weight2))
    return output


def _surface_primitives(source: CanonicalStage, target: CanonicalStage) -> tuple[SurfacePrimitive, ...]:
    slices = _addition_slices(source, target)
    if not slices:
        raise CanonicalizationError("E_CONDITION_EMPTY", "source/target 没有 addition surface")
    primitives: list[SurfacePrimitive] = []
    for index, (lower_q, upper_q, geometry) in enumerate(slices):
        below = slices[index - 1][2] if index > 0 and slices[index - 1][1] == lower_q else None
        above = slices[index + 1][2] if index + 1 < len(slices) and slices[index + 1][0] == upper_q else None
        primitives.extend(_horizontal_primitives(lower_q, _difference(geometry, below), "bottom"))
        primitives.extend(_horizontal_primitives(upper_q, _difference(geometry, above), "top"))

        for polygon_index, polygon in enumerate(_polygons(geometry)):
            rings = (polygon.exterior, *polygon.interiors)
            for ring_index, ring in enumerate(rings):
                coordinates = tuple((int(round(x)), int(round(z))) for x, z in ring.coords)
                for segment_index, (start, end) in enumerate(zip(coordinates, coordinates[1:])):
                    if start == end:
                        continue
                    length_q = isqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    weight2 = 2 * length_q * (upper_q - lower_q)
                    if weight2 <= 0:
                        continue
                    descriptor = (
                        "vertical",
                        lower_q,
                        upper_q,
                        start,
                        end,
                        polygon_index,
                        ring_index,
                        segment_index,
                    )
                    primitives.append(
                        _VerticalPrimitive(lower_q, upper_q, start, end, descriptor, weight2)
                    )
    return tuple(sorted(primitives, key=lambda primitive: primitive.descriptor))


def _largest_remainder(weights: tuple[int, ...], total: int) -> tuple[int, ...]:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise CanonicalizationError("E_CONDITION_SAMPLING", "condition surface 总面积为零")
    counts = [(total * weight) // weight_sum for weight in weights]
    remainders = [(total * weight) % weight_sum for weight in weights]
    remaining = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: (-remainders[index], index))
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    value = index
    while value > 0:
        result += fraction * (value % base)
        value //= base
        fraction /= base
    return result


def _primitive_offset(seed_hex: str, descriptor: tuple) -> int:
    digest = hashlib.sha256(f"{seed_hex}\0{descriptor!r}".encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % 1_000_003


def _sample_horizontal(
    primitive: _HorizontalPrimitive,
    count: int,
    seed_hex: str,
) -> set[Point3Q]:
    if count <= 0:
        return set()
    min_x, min_z, max_x, max_z = primitive.polygon.bounds
    prepared = prep(primitive.polygon)
    output: set[Point3Q] = set()
    offset = _primitive_offset(seed_hex, primitive.descriptor)
    maximum_attempts = max(10_000, count * 500)
    for attempt in range(maximum_attempts):
        sample_index = offset + attempt
        x = min_x + _halton(sample_index, 2) * (max_x - min_x)
        z = min_z + _halton(sample_index, 3) * (max_z - min_z)
        x_q = quantize_scalar(x, "1")
        z_q = quantize_scalar(z, "1")
        if prepared.covers(Point(x_q, z_q)):
            output.add((x_q, primitive.y_q, z_q))
            if len(output) == count:
                break
    return output


def _sample_vertical(
    primitive: _VerticalPrimitive,
    count: int,
    seed_hex: str,
) -> set[Point3Q]:
    if count <= 0:
        return set()
    output: set[Point3Q] = set()
    offset = _primitive_offset(seed_hex, primitive.descriptor)
    maximum_attempts = max(10_000, count * 200)
    for attempt in range(maximum_attempts):
        sample_index = offset + attempt
        along = _halton(sample_index, 2)
        vertical = _halton(sample_index, 3)
        x_q = quantize_scalar(
            primitive.start_q[0] + along * (primitive.end_q[0] - primitive.start_q[0]),
            "1",
        )
        z_q = quantize_scalar(
            primitive.start_q[1] + along * (primitive.end_q[1] - primitive.start_q[1]),
            "1",
        )
        y_q = quantize_scalar(
            primitive.lower_q + vertical * (primitive.upper_q - primitive.lower_q),
            "1",
        )
        output.add((x_q, y_q, z_q))
        if len(output) == count:
            break
    return output


def build_canonical_condition(
    source: CanonicalStage,
    target: CanonicalStage,
    config: ConditionConfig,
) -> CanonicalCondition:
    seed_hex = hashlib.sha256(
        f"{source.stage_hash}\0{target.stage_hash}\0{config.config_hash}".encode("utf-8")
    ).hexdigest()
    primitives = _surface_primitives(source, target)
    requested_candidates = config.point_count * config.candidate_multiplier
    allocations = _largest_remainder(
        tuple(primitive.weight2_q2 for primitive in primitives),
        requested_candidates,
    )
    candidates: set[Point3Q] = set()
    for primitive, count in zip(primitives, allocations):
        if isinstance(primitive, _HorizontalPrimitive):
            candidates.update(_sample_horizontal(primitive, count, seed_hex))
        else:
            candidates.update(_sample_vertical(primitive, count, seed_hex))
    ordered_candidates = tuple(sorted(candidates))
    if len(ordered_candidates) < config.point_count:
        raise CanonicalizationError(
            "E_CONDITION_SAMPLING",
            "去重后的 condition candidate 不足",
            required=config.point_count,
            actual=len(ordered_candidates),
        )
    if len(ordered_candidates) == config.point_count:
        points = ordered_candidates
    else:
        candidate_array = np.asarray(ordered_candidates, dtype=np.float64)
        indices = fpsample.fps_sampling(candidate_array, config.point_count, start_idx=0)
        points = tuple(sorted(ordered_candidates[int(index)] for index in indices))
    condition_hash = canonical_hash(
        {
            "condition_config_hash": config.config_hash,
            "points_q": points,
        }
    )
    return CanonicalCondition(points, condition_hash, seed_hex, len(primitives), len(ordered_candidates))

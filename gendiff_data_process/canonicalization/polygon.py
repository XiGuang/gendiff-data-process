from __future__ import annotations

from collections.abc import Iterable

from shapely.geometry import LinearRing, Polygon
from shapely.validation import explain_validity

from .errors import CanonicalizationError
from .types import PointQ


def area2(ring: Iterable[PointQ]) -> int:
    points = tuple(ring)
    return sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def _between_collinear(previous: PointQ, current: PointQ, following: PointQ) -> bool:
    cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
        current[1] - previous[1]
    ) * (following[0] - current[0])
    if cross != 0:
        return False
    return (current[0] - previous[0]) * (current[0] - following[0]) + (
        current[1] - previous[1]
    ) * (current[1] - following[1]) <= 0


def cleanup_ring(points: Iterable[PointQ], *, remove_collinear: bool = True) -> tuple[PointQ, ...]:
    ring = list(points)
    while len(ring) > 1 and ring[-1] == ring[0]:
        ring.pop()

    deduplicated: list[PointQ] = []
    for point in ring:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    ring = deduplicated

    if remove_collinear:
        changed = True
        while changed and len(ring) >= 3:
            changed = False
            for index in range(len(ring)):
                if _between_collinear(ring[index - 1], ring[index], ring[(index + 1) % len(ring)]):
                    del ring[index]
                    changed = True
                    break

    if len(ring) < 3:
        raise CanonicalizationError("E_TOO_FEW_POINTS", "清理后 footprint 少于三个点", point_count=len(ring))
    linear_ring = LinearRing(ring)
    if not linear_ring.is_simple:
        raise CanonicalizationError("E_SELF_INTERSECTION", "量化后的 footprint 自交")
    signed_area = area2(ring)
    if signed_area == 0:
        raise CanonicalizationError("E_TOO_FEW_POINTS", "量化后 footprint 面积为零")

    polygon = Polygon(ring)
    if not polygon.is_valid:
        raise CanonicalizationError(
            "E_SELF_INTERSECTION",
            "量化后的 footprint 自交或无效",
            reason=explain_validity(polygon),
        )

    if signed_area < 0:
        ring.reverse()
    rotations = (tuple(ring[offset:] + ring[:offset]) for offset in range(len(ring)))
    return min(rotations)


def polygon_from_ring(ring: tuple[PointQ, ...]) -> Polygon:
    polygon = Polygon(ring)
    if polygon.interiors:
        raise CanonicalizationError("E_HOLE_UNSUPPORTED", "v1 不支持带洞 polygon")
    if not polygon.is_valid:
        raise CanonicalizationError("E_SELF_INTERSECTION", "canonical ring 无效", reason=explain_validity(polygon))
    return polygon


def rings_from_geometry(geometry) -> tuple[tuple[PointQ, ...], ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    elif geometry.geom_type == "GeometryCollection":
        polygons = [item for item in geometry.geoms if item.geom_type == "Polygon" and not item.is_empty]
        unexpected = [item.geom_type for item in geometry.geoms if item.geom_type not in {"Polygon", "LineString", "Point"}]
        if unexpected:
            raise CanonicalizationError("E_SOLID_MISMATCH", "union 返回不支持的几何", geometry_types=unexpected)
    else:
        raise CanonicalizationError("E_SOLID_MISMATCH", "union 返回非 polygon 几何", geometry_type=geometry.geom_type)

    rings: list[tuple[PointQ, ...]] = []
    for polygon in polygons:
        if polygon.interiors:
            raise CanonicalizationError("E_HOLE_UNSUPPORTED", "union 产生 hole")
        coordinates = tuple((int(round(x)), int(round(z))) for x, z in list(polygon.exterior.coords)[:-1])
        rings.append(cleanup_ring(coordinates))
    return tuple(sorted(rings))

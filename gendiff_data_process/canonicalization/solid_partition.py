from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from shapely import set_precision
from shapely.ops import unary_union

from .errors import CanonicalizationError
from .polygon import area2, polygon_from_ring, rings_from_geometry
from .types import CanonicalStage, PointQ, QuantizedLayer


@dataclass(frozen=True)
class SlabCell:
    min_height_q: int
    max_height_q: int
    footprint_q: tuple[PointQ, ...]


@dataclass(frozen=True)
class StageChange:
    change_kind: str
    added_volume2_q3: int
    removed_volume2_q3: int


def _union_polygons(polygons):
    if not polygons:
        return None
    return set_precision(unary_union(polygons), grid_size=1.0, mode="valid_output")


def _raw_warnings(layers: tuple[QuantizedLayer, ...]) -> set[str]:
    warnings: set[str] = set()
    signatures = Counter(
        (layer.min_height_q, layer.max_height_q, layer.footprint_q) for layer in layers
    )
    if any(count > 1 for count in signatures.values()):
        warnings.add("W_RAW_DUPLICATE_LAYER")
    for left, right in combinations(layers, 2):
        overlap_height = min(left.max_height_q, right.max_height_q) - max(
            left.min_height_q, right.min_height_q
        )
        if overlap_height <= 0:
            continue
        if (
            polygon_from_ring(left.footprint_q)
            .intersection(polygon_from_ring(right.footprint_q))
            .area
            > 0
        ):
            warnings.add("W_RAW_OVERLAP_CANONICALIZED")
            break
    return warnings


def build_slab_cells(
    layers: tuple[QuantizedLayer, ...],
) -> tuple[tuple[SlabCell, ...], tuple[str, ...]]:
    if not layers:
        return (), ()
    heights = sorted(
        {
            value
            for layer in layers
            for value in (layer.min_height_q, layer.max_height_q)
        }
    )
    warnings = _raw_warnings(layers)
    raw_cells: list[SlabCell] = []
    for lower, upper in zip(heights, heights[1:]):
        if lower == upper:
            continue
        active = [
            polygon_from_ring(layer.footprint_q)
            for layer in layers
            if layer.min_height_q <= lower and layer.max_height_q >= upper
        ]
        geometry = _union_polygons(active)
        if geometry is None or geometry.is_empty:
            continue
        for ring in rings_from_geometry(geometry):
            raw_cells.append(SlabCell(lower, upper, ring))

    merged: dict[tuple[PointQ, ...], list[SlabCell]] = {}
    for cell in sorted(
        raw_cells,
        key=lambda item: (item.footprint_q, item.min_height_q, item.max_height_q),
    ):
        chain = merged.setdefault(cell.footprint_q, [])
        if chain and chain[-1].max_height_q == cell.min_height_q:
            previous = chain[-1]
            chain[-1] = SlabCell(
                previous.min_height_q, cell.max_height_q, cell.footprint_q
            )
        else:
            chain.append(cell)

    cells = tuple(
        sorted(
            (cell for chain in merged.values() for cell in chain),
            key=lambda item: (
                item.min_height_q,
                item.max_height_q,
                abs(area2(item.footprint_q)),
                item.footprint_q,
            ),
        )
    )
    _validate_partition(layers, cells)
    return cells, tuple(sorted(warnings))


def _volume2_cells(cells: tuple[SlabCell, ...]) -> int:
    return sum(
        abs(area2(cell.footprint_q)) * (cell.max_height_q - cell.min_height_q)
        for cell in cells
    )


def _validate_partition(
    layers: tuple[QuantizedLayer, ...], cells: tuple[SlabCell, ...]
) -> None:
    heights = sorted(
        {
            value
            for layer in layers
            for value in (layer.min_height_q, layer.max_height_q)
        }
    )
    raw_volume2 = 0
    for lower, upper in zip(heights, heights[1:]):
        active = [
            polygon_from_ring(layer.footprint_q)
            for layer in layers
            if layer.min_height_q <= lower and layer.max_height_q >= upper
        ]
        geometry = _union_polygons(active)
        if geometry is not None:
            raw_volume2 += int(round(geometry.area * 2)) * (upper - lower)
    if raw_volume2 != _volume2_cells(cells):
        raise CanonicalizationError(
            "E_SOLID_MISMATCH",
            "canonical cells 与 raw solid 体积不一致",
            raw_volume2=raw_volume2,
            canonical_volume2=_volume2_cells(cells),
        )

    for left, right in combinations(cells, 2):
        overlap_height = min(left.max_height_q, right.max_height_q) - max(
            left.min_height_q, right.min_height_q
        )
        if overlap_height <= 0:
            continue
        if (
            polygon_from_ring(left.footprint_q)
            .intersection(polygon_from_ring(right.footprint_q))
            .area
            > 0
        ):
            raise CanonicalizationError(
                "E_CANONICAL_OVERLAP", "canonical cells 存在正体积交叠"
            )


def removed_volume2(
    source_cells: tuple[SlabCell, ...], target_cells: tuple[SlabCell, ...]
) -> int:
    heights = sorted(
        {
            value
            for cell in (*source_cells, *target_cells)
            for value in (cell.min_height_q, cell.max_height_q)
        }
    )
    removed = 0
    for lower, upper in zip(heights, heights[1:]):
        source = _union_polygons(
            [
                polygon_from_ring(cell.footprint_q)
                for cell in source_cells
                if cell.min_height_q <= lower and cell.max_height_q >= upper
            ]
        )
        if source is None or source.is_empty:
            continue
        target = _union_polygons(
            [
                polygon_from_ring(cell.footprint_q)
                for cell in target_cells
                if cell.min_height_q <= lower and cell.max_height_q >= upper
            ]
        )
        difference = (
            source
            if target is None
            else set_precision(
                source.difference(target), grid_size=1.0, mode="valid_output"
            )
        )
        removed += int(round(difference.area * 2)) * (upper - lower)
    return removed


def stage_cells(stage: CanonicalStage) -> tuple[SlabCell, ...]:
    return tuple(
        SlabCell(layer.min_height_q, layer.max_height_q, layer.footprint_q)
        for layer in stage.layers
    )


def classify_stage_change(
    source: CanonicalStage,
    target: CanonicalStage,
    *,
    volume_tolerance_q3: int = 0,
) -> StageChange:
    if volume_tolerance_q3 < 0:
        raise ValueError("volume tolerance 不能为负数")
    source_cells = stage_cells(source)
    target_cells = stage_cells(target)
    removed = removed_volume2(source_cells, target_cells)
    added = removed_volume2(target_cells, source_cells)
    tolerance2 = volume_tolerance_q3 * 2
    has_added = added > tolerance2
    has_removed = removed > tolerance2
    if has_added and has_removed:
        kind = "mixed"
    elif has_added:
        kind = "construction"
    elif has_removed:
        kind = "demolition"
    else:
        kind = "noop"
    return StageChange(kind, added, removed)

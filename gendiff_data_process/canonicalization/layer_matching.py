from __future__ import annotations

from dataclasses import dataclass

from .config import CanonicalizerConfig
from .polygon import area2, polygon_from_ring
from .types import CanonicalLayer, LayerMatch


@dataclass(frozen=True)
class _EdgeMetric:
    intersection_volume2: int
    iou_q: int
    symmetric_difference_volume2: int
    height_difference_q: int
    centroid_distance_q: int


def _rounded_ratio(numerator: int, denominator: int, scale: int) -> int:
    return (numerator * scale + denominator // 2) // denominator if denominator > 0 else 0


def _layer_volume2(layer: CanonicalLayer) -> int:
    return abs(area2(layer.footprint_q)) * (layer.max_height_q - layer.min_height_q)


def _edge_metric(source: CanonicalLayer, target: CanonicalLayer, cfg: CanonicalizerConfig) -> _EdgeMetric | None:
    overlap_height = min(source.max_height_q, target.max_height_q) - max(
        source.min_height_q, target.min_height_q
    )
    if overlap_height <= 0:
        return None
    intersection_area2 = int(
        round(polygon_from_ring(source.footprint_q).intersection(polygon_from_ring(target.footprint_q)).area * 2)
    )
    intersection = intersection_area2 * overlap_height
    if intersection <= 0 and cfg.layer_matching.require_positive_intersection:
        return None

    source_volume = _layer_volume2(source)
    target_volume = _layer_volume2(target)
    union_volume = source_volume + target_volume - intersection
    iou_q = _rounded_ratio(intersection, union_volume, cfg.layer_matching.metric_scale)
    coverage_q = _rounded_ratio(intersection, min(source_volume, target_volume), cfg.layer_matching.metric_scale)
    if iou_q < cfg.layer_matching.min_iou_q and coverage_q < cfg.layer_matching.min_smaller_coverage_q:
        return None

    source_centroid = polygon_from_ring(source.footprint_q).centroid
    target_centroid = polygon_from_ring(target.footprint_q).centroid
    centroid_distance_q = int(round(source_centroid.distance(target_centroid) * cfg.layer_matching.metric_scale))
    return _EdgeMetric(
        intersection_volume2=intersection,
        iou_q=iou_q,
        symmetric_difference_volume2=source_volume + target_volume - 2 * intersection,
        height_difference_q=abs(source.min_height_q - target.min_height_q)
        + abs(source.max_height_q - target.max_height_q),
        centroid_distance_q=centroid_distance_q,
    )


def _hungarian_min(cost: list[list[int]]) -> list[int]:
    """返回每一行的列；使用 Python 整数并固定同成本列顺序。"""

    row_count = len(cost)
    if row_count == 0:
        return []
    column_count = len(cost[0])
    if row_count > column_count:
        raise ValueError("assignment 要求列数不少于行数")

    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)
    infinity = max(max(row) for row in cost) * (row_count + column_count + 2) + 1

    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        column_zero = 0
        minimum = [infinity] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column_zero] = True
            current_row = matched_row[column_zero]
            delta = infinity
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = cost[current_row - 1][column - 1] - u[current_row] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    previous_column[column] = column_zero
                if minimum[column] < delta or (minimum[column] == delta and column < next_column):
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column_zero = next_column
            if matched_row[column_zero] == 0:
                break
        while True:
            previous = previous_column[column_zero]
            matched_row[column_zero] = matched_row[previous]
            column_zero = previous
            if column_zero == 0:
                break

    result = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            result[matched_row[column] - 1] = column - 1
    return result


def match_layers(
    source_layers: tuple[CanonicalLayer, ...],
    target_layers: tuple[CanonicalLayer, ...],
    cfg: CanonicalizerConfig,
) -> tuple[tuple[LayerMatch, ...], tuple[str, ...]]:
    source_count = len(source_layers)
    target_count = len(target_layers)
    if source_count == 0 or target_count == 0:
        return (), ()

    metrics: dict[tuple[int, int], _EdgeMetric] = {}
    for source_index, source in enumerate(source_layers):
        for target_index, target in enumerate(target_layers):
            metric = _edge_metric(source, target, cfg)
            if metric is not None:
                metrics[(source_index, target_index)] = metric

    warnings: set[str] = set()
    for source_index in range(source_count):
        if sum(1 for key in metrics if key[0] == source_index) > 1:
            warnings.add("W_LAYER_SPLIT")
    for target_index in range(target_count):
        if sum(1 for key in metrics if key[1] == target_index) > 1:
            warnings.add("W_LAYER_MERGE")

    max_symmetric = max((item.symmetric_difference_volume2 for item in metrics.values()), default=0)
    max_height = max((item.height_difference_q for item in metrics.values()), default=0)
    max_centroid = max((item.centroid_distance_q for item in metrics.values()), default=0)
    metric_scale = cfg.layer_matching.metric_scale

    weight_centroid = 1
    weight_height = source_count * max_centroid + 1
    weight_symmetric = weight_height * (source_count * max_height + 1)
    weight_iou = weight_symmetric * (source_count * max_symmetric + 1)
    weight_intersection = weight_iou * (source_count * metric_scale + 1)

    tie_base = target_count + 1
    tie_factor = tie_base**source_count
    scores: list[list[int | None]] = []
    for source_index in range(source_count):
        row: list[int | None] = []
        tie_position_weight = tie_base ** (source_count - source_index - 1)
        for target_index in range(target_count):
            metric = metrics.get((source_index, target_index))
            if metric is None:
                row.append(None)
                continue
            primary = (
                metric.intersection_volume2 * weight_intersection
                + metric.iou_q * weight_iou
                + (max_symmetric - metric.symmetric_difference_volume2) * weight_symmetric
                + (max_height - metric.height_difference_q) * weight_height
                + (max_centroid - metric.centroid_distance_q) * weight_centroid
            )
            tie_score = (target_count - target_index) * tie_position_weight
            row.append(primary * tie_factor + tie_score)
        for dummy_index in range(source_count):
            row.append(0 if dummy_index == source_index else None)
        scores.append(row)

    maximum_score = max((score for row in scores for score in row if score is not None), default=0)
    prohibited_cost = (maximum_score + 1) * (source_count + target_count + 2)
    costs = [
        [prohibited_cost if score is None else maximum_score - score for score in row]
        for row in scores
    ]
    assigned_columns = _hungarian_min(costs)
    matches = tuple(
        LayerMatch(source_index, column)
        for source_index, column in enumerate(assigned_columns)
        if 0 <= column < target_count and (source_index, column) in metrics
    )
    return matches, tuple(sorted(warnings))

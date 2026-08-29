from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isqrt

from .config import CanonicalizerConfig
from .types import CanonicalLayer, PointAlignment, PointEdit, PointQ


_ACTION_PRIORITY = {"KEEP_POINT": 0, "MOVE_POINT": 1, "DELETE_POINT": 2, "INSERT_POINT": 3}
_MISSING_INDEX = 1 << 30


@dataclass(frozen=True)
class _Candidate:
    delete_insert_count: int
    move_count: int
    distance_squared_q: int
    edits: tuple[PointEdit, ...]

    @property
    def key(self) -> tuple:
        serialized = tuple(
            (
                _ACTION_PRIORITY[edit.action],
                edit.source_index if edit.source_index is not None else _MISSING_INDEX,
                edit.target_index if edit.target_index is not None else _MISSING_INDEX,
            )
            for edit in self.edits
        )
        return self.delete_insert_count, self.move_count, self.distance_squared_q, serialized


def _prepend(candidate: _Candidate, edit: PointEdit, *, distance_squared: int = 0) -> _Candidate:
    return _Candidate(
        candidate.delete_insert_count + int(edit.action in {"DELETE_POINT", "INSERT_POINT"}),
        candidate.move_count + int(edit.action == "MOVE_POINT"),
        candidate.distance_squared_q + distance_squared,
        (edit,) + candidate.edits,
    )


def _move_limit(source: tuple[PointQ, ...], target: tuple[PointQ, ...], cfg: CanonicalizerConfig) -> int:
    all_points = source + target
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_z = min(point[1] for point in all_points)
    max_z = max(point[1] for point in all_points)
    diagonal = isqrt((max_x - min_x) ** 2 + (max_z - min_z) ** 2)
    ratio_limit = int(cfg.point_matching.move_ratio * diagonal)
    return max(cfg.point_matching.min_move_distance_q, ratio_limit)


def align_points(source_layer: CanonicalLayer, target_ring: tuple[PointQ, ...], cfg: CanonicalizerConfig) -> PointAlignment:
    source = source_layer.footprint_q
    if not source and not target_ring:
        return PointAlignment(())
    move_limit_squared = _move_limit(source, target_ring, cfg) ** 2 if source and target_ring else 0
    candidates: list[_Candidate] = []

    for offset in range(max(1, len(target_ring))):
        target_order = tuple((offset + index) % len(target_ring) for index in range(len(target_ring)))

        @lru_cache(maxsize=None)
        def solve(source_index: int, target_offset_index: int) -> _Candidate:
            if source_index == len(source) and target_offset_index == len(target_order):
                return _Candidate(0, 0, 0, ())
            options: list[_Candidate] = []
            if source_index < len(source):
                delete_edit = PointEdit(
                    "DELETE_POINT",
                    source_layer.point_lineage_ids[source_index] if source_layer.point_lineage_ids else None,
                    source_index,
                    None,
                )
                options.append(_prepend(solve(source_index + 1, target_offset_index), delete_edit))
            if target_offset_index < len(target_order):
                target_index = target_order[target_offset_index]
                insert_edit = PointEdit("INSERT_POINT", None, None, target_index, target_ring[target_index])
                options.append(_prepend(solve(source_index, target_offset_index + 1), insert_edit))
            if source_index < len(source) and target_offset_index < len(target_order):
                target_index = target_order[target_offset_index]
                source_point = source[source_index]
                target_point = target_ring[target_index]
                distance_squared = (source_point[0] - target_point[0]) ** 2 + (source_point[1] - target_point[1]) ** 2
                if distance_squared == 0 or distance_squared <= move_limit_squared:
                    action = "KEEP_POINT" if distance_squared == 0 else "MOVE_POINT"
                    match_edit = PointEdit(
                        action,
                        source_layer.point_lineage_ids[source_index] if source_layer.point_lineage_ids else None,
                        source_index,
                        target_index,
                        target_point if action == "MOVE_POINT" else None,
                    )
                    options.append(
                        _prepend(
                            solve(source_index + 1, target_offset_index + 1),
                            match_edit,
                            distance_squared=distance_squared,
                        )
                    )
            return min(options, key=lambda candidate: candidate.key)

        candidates.append(solve(0, 0))

    best = min(candidates, key=lambda candidate: candidate.key)
    source_backed = sorted(
        (edit for edit in best.edits if edit.source_index is not None),
        key=lambda edit: (edit.source_index, _ACTION_PRIORITY[edit.action]),
    )
    inserts = sorted(
        (edit for edit in best.edits if edit.source_index is None),
        key=lambda edit: edit.target_index if edit.target_index is not None else _MISSING_INDEX,
    )
    edits = tuple(source_backed + inserts)
    matched_count = sum(edit.action in {"KEEP_POINT", "MOVE_POINT"} for edit in edits)
    warning = "W_POINT_ALIGNMENT_FALLBACK" if source and target_ring and matched_count == 0 else None
    return PointAlignment(edits, warning)

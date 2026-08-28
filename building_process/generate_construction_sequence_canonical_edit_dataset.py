from __future__ import annotations

import argparse
import math
import random
import sys
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_process import build_layer_edit_dataset_from_sequence as dataset_stage
from building_process import generate_construction_proxy as proxy_stage
from building_process import generate_construction_sequence_edit_dataset as sequence_stage


POLICIES = ("vertical", "footprint", "hybrid")
POINT_ID_STRIDE = 100000
GENERATED_POINT_ID_OFFSET = 900000000000
POINT_KEEP_EPS = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate construction-stage sequences with stable layer/point IDs and "
            "canonical layer-edit supervision."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=None,
        help="Optional dataset output root. Defaults to <sequence_dir>/layer_edit_dataset.",
    )
    parser.add_argument("--stage-count", type=int, default=5)
    parser.add_argument("--policy", choices=("auto",) + POLICIES, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-progress", type=float, default=0.15)
    parser.add_argument("--axis", choices=("random", "x", "z"), default="random")
    parser.add_argument("--keep-side", choices=("random", "min", "max"), default="random")
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--export-obj", action="store_true")
    parser.add_argument(
        "--pair-mode",
        choices=("consecutive", "all_forward"),
        default="consecutive",
    )
    parser.add_argument("--condition-point-count", type=int, default=8192)
    parser.add_argument("--change-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--point-keep-eps", type=float, default=POINT_KEEP_EPS)
    parser.add_argument("--copy-objs", action="store_true")
    parser.add_argument("--save-condition-ply", action="store_true")
    parser.add_argument(
        "--no-v2-preview-obj",
        action="store_true",
        help="Disable OBJ previews reconstructed from edit_sequences_v2 YAML files.",
    )
    return parser.parse_args()


def _round_coord(value: float, digits: int = 6) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == -0.0 else rounded


def _stable_crc_id(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return GENERATED_POINT_ID_OFFSET + int(zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF)


def _source_point_id(source_proxy_id: int, point_index: int) -> int:
    return int(source_proxy_id) * POINT_ID_STRIDE + int(point_index)


def _assign_source_point_ids(entries: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for entry in entries:
        source_proxy_id = int(entry["proxy_id"])
        point_count = len(entry.get("footprint") or [])
        point_ids = [_source_point_id(source_proxy_id, index) for index in range(point_count)]
        copied = dict(entry)
        copied["point_ids"] = list(point_ids)
        copied["source_point_ids"] = list(point_ids)
        copied["point_roles"] = ["original"] * point_count
        annotated.append(copied)
    return annotated


def _point_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return float(math.hypot(float(lhs[0]) - float(rhs[0]), float(lhs[1]) - float(rhs[1])))


def _nearest_source_point(
    point: Sequence[float],
    source_entry: dict,
    eps: float,
) -> tuple[int, int] | None:
    best: tuple[float, int, int] | None = None
    for index, source_point in enumerate(source_entry.get("footprint") or []):
        distance = _point_distance(point, source_point)
        if distance <= eps and (best is None or distance < best[0]):
            source_point_id = int(source_entry["point_ids"][index])
            best = (distance, index, source_point_id)
    if best is None:
        return None
    return best[1], best[2]


def _nearest_source_edge(
    point: Sequence[float],
    source_entry: dict,
) -> tuple[int, int, float, float]:
    points = source_entry.get("footprint") or []
    if len(points) < 2:
        return 0, 0, 0.0, float("inf")

    target = np.asarray(point, dtype=float).reshape(2)
    best_edge = 0
    best_next = 0
    best_u = 0.0
    best_distance = float("inf")
    for edge_index, start_raw in enumerate(points):
        start = np.asarray(start_raw, dtype=float).reshape(2)
        end = np.asarray(points[(edge_index + 1) % len(points)], dtype=float).reshape(2)
        edge = end - start
        length_sq = float(np.dot(edge, edge))
        if length_sq <= 1.0e-12:
            continue
        u = float(np.dot(target - start, edge) / length_sq)
        u_clamped = min(1.0, max(0.0, u))
        projected = start + u_clamped * edge
        distance = float(np.linalg.norm(target - projected))
        if distance < best_distance:
            best_edge = edge_index
            best_next = (edge_index + 1) % len(points)
            best_u = u_clamped
            best_distance = distance
    return best_edge, best_next, best_u, best_distance


def _clip_boundary_id(point: Sequence[float], metadata: dict, eps: float) -> str:
    bounds = metadata.get("clip_bounds")
    if not isinstance(bounds, dict):
        return "none"
    x = float(point[0])
    z = float(point[1])
    checks = (
        ("min_x", x),
        ("max_x", x),
        ("min_z", z),
        ("max_z", z),
    )
    for key, value in checks:
        bound = bounds.get(key)
        if bound is not None and abs(float(bound) - value) <= eps:
            return key
    return "none"


def _generated_point_id(
    source_proxy_id: int,
    point: Sequence[float],
    metadata: dict,
    role: str,
    edge_start: int | None = None,
    edge_end: int | None = None,
    boundary_id: str | None = None,
) -> int:
    return _stable_crc_id(
        "point",
        role,
        int(source_proxy_id),
        edge_start if edge_start is not None else "na",
        edge_end if edge_end is not None else "na",
        boundary_id or "none",
        metadata.get("policy", metadata.get("mode", "unknown")),
        metadata.get("axis", "none"),
        metadata.get("keep_side", "none"),
        _round_coord(float(point[0])),
        _round_coord(float(point[1])),
    )


def _deduplicate_point_ids(entry: dict) -> dict:
    seen: set[int] = set()
    point_ids = list(entry["point_ids"])
    source_point_ids = list(entry["source_point_ids"])
    roles = list(entry["point_roles"])
    for index, point_id in enumerate(point_ids):
        if point_id not in seen:
            seen.add(point_id)
            continue
        point_ids[index] = _stable_crc_id(
            "dedupe",
            entry.get("source_proxy_id", entry.get("proxy_id", 0)),
            point_id,
            index,
            _round_coord(entry["footprint"][index][0]),
            _round_coord(entry["footprint"][index][1]),
        )
        source_point_ids[index] = None
        roles[index] = "generated"
        seen.add(point_ids[index])
    entry["point_ids"] = point_ids
    entry["source_point_ids"] = source_point_ids
    entry["point_roles"] = roles
    return entry


def _annotate_stage_point_lineage(
    stage_entries: list[dict],
    source_entries: list[dict],
    metadata: dict,
    point_keep_eps: float,
) -> list[dict]:
    source_by_proxy = {int(entry["proxy_id"]): entry for entry in source_entries}
    annotated: list[dict] = []
    boundary_eps = max(float(point_keep_eps), proxy_stage.EPS * 10.0)

    for entry in stage_entries:
        source_proxy_id = int(entry.get("source_proxy_id", entry.get("proxy_id", 0)))
        source_entry = source_by_proxy.get(source_proxy_id)
        if source_entry is None:
            annotated.append(dict(entry))
            continue

        point_ids: list[int] = []
        source_point_ids: list[int | None] = []
        point_roles: list[str] = []
        for point in entry.get("footprint") or []:
            source_match = _nearest_source_point(point, source_entry, float(point_keep_eps))
            if source_match is not None:
                _, source_pid = source_match
                point_ids.append(source_pid)
                source_point_ids.append(source_pid)
                point_roles.append("original")
                continue

            edge_start, edge_end, _, edge_distance = _nearest_source_edge(point, source_entry)
            boundary_id = _clip_boundary_id(point, metadata, boundary_eps)
            if boundary_id != "none" and edge_distance <= boundary_eps:
                point_id = _generated_point_id(
                    source_proxy_id,
                    point,
                    metadata,
                    "intersection",
                    edge_start=edge_start,
                    edge_end=edge_end,
                    boundary_id=boundary_id,
                )
                point_ids.append(point_id)
                source_point_ids.append(None)
                point_roles.append("intersection")
                continue

            point_ids.append(_generated_point_id(source_proxy_id, point, metadata, "generated"))
            source_point_ids.append(None)
            point_roles.append("generated")

        copied = dict(entry)
        copied["point_ids"] = point_ids
        copied["source_point_ids"] = source_point_ids
        copied["point_roles"] = point_roles
        annotated.append(_deduplicate_point_ids(copied))
    return annotated


def _layer_lineage_entries(entries: Sequence[dict]) -> list[dict]:
    lineage: list[dict] = []
    for layer in entries:
        source_proxy_id = int(layer.get("source_proxy_id", layer.get("proxy_id", 0)))
        for point_index, point_id in enumerate(layer.get("point_ids") or []):
            source_point_ids = layer.get("source_point_ids") or []
            roles = layer.get("point_roles") or []
            lineage.append(
                {
                    "layer_proxy_id": int(layer.get("proxy_id", -1)),
                    "point_index": int(point_index),
                    "point_id": int(point_id),
                    "source_proxy_id": source_proxy_id,
                    "source_point_id": source_point_ids[point_index]
                    if point_index < len(source_point_ids)
                    else None,
                    "role": roles[point_index] if point_index < len(roles) else "unknown",
                }
            )
    return lineage


def _write_stage_outputs(
    entries: list[dict],
    source_yaml: Path,
    output_dir: Path,
    metadata: dict,
    export_obj: bool,
) -> None:
    enriched_meta = {
        **metadata,
        "point_lineage_enabled": True,
        "point_id_policy": "deterministic_source_or_intersection",
        "point_keep_eps": float(metadata.get("point_keep_eps", POINT_KEEP_EPS)),
    }
    sequence_stage._write_stage_outputs(entries, source_yaml, output_dir, enriched_meta, export_obj)
    dataset_stage.save_yaml(output_dir / "point_lineage.yaml", _layer_lineage_entries(entries))


def _write_sequence_meta(
    sequence_dir: Path,
    *,
    source_yaml: Path,
    sequence_seed: int,
    policy: str,
    axis: str | None,
    keep_side: str | None,
    stage_records: list[dict],
    point_keep_eps: float,
) -> None:
    payload = {
        "source_yaml": str(source_yaml),
        "sequence_seed": int(sequence_seed),
        "policy": policy,
        "axis": axis,
        "keep_side": keep_side,
        "stage_count": len(stage_records),
        "stable_proxy_ids": True,
        "stable_point_ids": True,
        "point_lineage_enabled": True,
        "point_id_policy": "deterministic_source_or_intersection",
        "point_id_stride": POINT_ID_STRIDE,
        "point_keep_eps": float(point_keep_eps),
        "stages": stage_records,
    }
    dataset_stage.save_yaml(sequence_dir / "sequence_meta.yaml", payload)


def _normalize_layers_with_lineage(path: Path) -> list[dict]:
    data = dataset_stage.load_yaml(path) or []
    if not isinstance(data, list):
        raise ValueError(f"Malformed layer yaml: {path}")

    layers: list[dict] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            continue
        footprint: list[list[float]] = []
        for point in raw.get("footprint") or []:
            coords = np.asarray(point, dtype=float).reshape(-1).tolist()
            if len(coords) >= 2:
                footprint.append([float(coords[0]), float(coords[1])])

        layer = {
            "proxy_id": int(raw.get("proxy_id", index)),
            "source_proxy_id": int(raw.get("source_proxy_id", raw.get("proxy_id", index))),
            "level_index": int(raw.get("level_index", index)),
            "min_height": float(raw.get("min_height", 0.0)),
            "max_height": float(raw.get("max_height", 0.0)),
            "footprint": footprint,
        }
        for key in ("point_ids", "source_point_ids", "point_roles"):
            values = raw.get(key)
            if isinstance(values, list) and len(values) == len(footprint):
                layer[key] = list(values)
        layers.append(layer)
    return layers


def _height_changed(source_layer: dict, target_layer: dict, eps: float) -> bool:
    return (
        abs(float(source_layer["min_height"]) - float(target_layer["min_height"])) > eps
        or abs(float(source_layer["max_height"]) - float(target_layer["max_height"])) > eps
    )


def _point_ids(layer: dict) -> list[int] | None:
    values = layer.get("point_ids")
    footprint = layer.get("footprint") or []
    if not isinstance(values, list) or len(values) != len(footprint):
        return None
    ids: list[int] = []
    for value in values:
        if value is None:
            return None
        ids.append(int(value))
    if len(set(ids)) != len(ids):
        return None
    return ids


def _same_point_geometry(source_layer: dict, target_layer: dict, eps: float) -> bool:
    source_points = source_layer.get("footprint") or []
    target_points = target_layer.get("footprint") or []
    if len(source_points) != len(target_points):
        return False
    return all(_point_distance(src, dst) <= eps for src, dst in zip(source_points, target_points))


def _match_layers(history_layers: Sequence[dict], target_layers: Sequence[dict]) -> dict[int, int]:
    matches: dict[int, int] = {}
    used_targets: set[int] = set()

    target_by_proxy: dict[int, int] = {}
    for target_index, target_layer in enumerate(target_layers):
        target_by_proxy.setdefault(int(target_layer.get("proxy_id", target_index)), target_index)

    for history_index, history_layer in enumerate(history_layers):
        target_index = target_by_proxy.get(int(history_layer.get("proxy_id", history_index)))
        if target_index is not None and target_index not in used_targets:
            matches[history_index] = target_index
            used_targets.add(target_index)

    candidates: list[tuple[int, float, int, int]] = []
    for history_index, history_layer in enumerate(history_layers):
        if history_index in matches:
            continue
        for target_index, target_layer in enumerate(target_layers):
            if target_index in used_targets:
                continue
            same_source = int(history_layer.get("source_proxy_id", history_layer.get("proxy_id", history_index))) == int(
                target_layer.get("source_proxy_id", target_layer.get("proxy_id", target_index))
            )
            if not same_source:
                continue
            score = float(sequence_stage._lineage_match_score(history_layer, target_layer))
            candidates.append((0, score, history_index, target_index))

    for history_index, history_layer in enumerate(history_layers):
        if history_index in matches:
            continue
        for target_index, target_layer in enumerate(target_layers):
            if target_index in used_targets:
                continue
            score = float(sequence_stage._lineage_match_score(history_layer, target_layer))
            if score > 0.0:
                candidates.append((1, score, history_index, target_index))

    candidates.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
    used_history = set(matches)
    for _, score, history_index, target_index in candidates:
        if score <= 0.0 or history_index in used_history or target_index in used_targets:
            continue
        matches[history_index] = target_index
        used_history.add(history_index)
        used_targets.add(target_index)
    return matches


def _edge_anchor(source_points: Sequence[Sequence[float]], target_point: Sequence[float]) -> dict:
    if len(source_points) < 2:
        return {
            "type": "POINT" if len(source_points) == 1 else "LAYER_FRAME",
            "index": 0,
            "value": [float(target_point[0]), float(target_point[1])],
        }

    target = np.asarray(target_point, dtype=float).reshape(2)
    best: tuple[float, int, np.ndarray] | None = None
    for edge_index, start_raw in enumerate(source_points):
        start = np.asarray(start_raw, dtype=float).reshape(2)
        end = np.asarray(source_points[(edge_index + 1) % len(source_points)], dtype=float).reshape(2)
        edge = end - start
        length_sq = float(np.dot(edge, edge))
        if length_sq <= 1.0e-12:
            continue
        u = min(1.0, max(0.0, float(np.dot(target - start, edge) / length_sq)))
        projected = start + u * edge
        length = math.sqrt(length_sq)
        tangent = edge / length
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        v = float(np.dot(target - projected, normal))
        distance_sq = float(np.sum((target - projected) ** 2))
        value = np.asarray([u, v], dtype=float)
        if best is None or distance_sq < best[0] - 1.0e-12:
            best = (distance_sq, edge_index, value)
    if best is None:
        return {"type": "LAYER_FRAME", "index": 0, "value": [float(target[0]), float(target[1])]}
    return {"type": "EDGE", "index": int(best[1]), "value": [float(best[2][0]), float(best[2][1])]}


def _layer_frame_anchor(target_layer: dict, target_point: Sequence[float]) -> dict:
    points = np.asarray(target_layer.get("footprint") or [[0.0, 0.0]], dtype=float).reshape(-1, 2)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    scale = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1.0e-6)
    value = (np.asarray(target_point, dtype=float).reshape(2) - center) / scale
    return {
        "type": "LAYER_FRAME",
        "index": 0,
        "value": [float(value[0]), float(value[1])],
        "frame": {"center": [float(center[0]), float(center[1])], "scale": float(scale)},
    }


def _point_edit(
    action: str,
    *,
    source_layer: dict | None,
    target_layer: dict | None,
    source_point_index: int | None,
    target_point_index: int | None,
    source_point_id: int | None,
    target_point_id: int | None,
    keep_eps: float,
) -> dict:
    source_coord = (
        [float(v) for v in source_layer["footprint"][source_point_index]]
        if source_layer is not None and source_point_index is not None
        else None
    )
    target_coord = (
        [float(v) for v in target_layer["footprint"][target_point_index]]
        if target_layer is not None and target_point_index is not None
        else None
    )

    if action == "MOVE" and source_coord is not None and target_coord is not None:
        value = [target_coord[0] - source_coord[0], target_coord[1] - source_coord[1]]
    else:
        value = [0.0, 0.0]

    anchor = None
    if action == "INSERT" and target_coord is not None:
        if source_layer is not None and source_layer.get("footprint"):
            anchor = _edge_anchor(source_layer["footprint"], target_coord)
        elif target_layer is not None:
            anchor = _layer_frame_anchor(target_layer, target_coord)
        value = list(anchor["value"]) if anchor is not None else list(target_coord)

    if action == "KEEP":
        value = [0.0, 0.0]
    elif action == "MOVE" and source_coord is not None and target_coord is not None:
        if _point_distance(source_coord, target_coord) <= keep_eps:
            action = "KEEP"
            value = [0.0, 0.0]

    return {
        "action": action,
        "source_point_id": source_point_id,
        "target_point_id": target_point_id,
        "source_point_index": source_point_index,
        "target_point_index": target_point_index,
        "source_coord": source_coord,
        "target_coord": target_coord,
        "value": [float(value[0]), float(value[1])],
        "anchor": anchor,
    }


def _point_edits_for_matched_layer(
    source_layer: dict,
    target_layer: dict,
    keep_eps: float,
) -> tuple[list[dict], bool]:
    source_ids = _point_ids(source_layer)
    target_ids = _point_ids(target_layer)
    fallback_used = source_ids is None or target_ids is None
    edits: list[dict] = []

    if not fallback_used:
        assert source_ids is not None and target_ids is not None
        target_by_id = {point_id: index for index, point_id in enumerate(target_ids)}
        source_id_set = set(source_ids)

        for source_index, source_id in enumerate(source_ids):
            target_index = target_by_id.get(source_id)
            if target_index is None:
                continue
            action = (
                "KEEP"
                if _point_distance(source_layer["footprint"][source_index], target_layer["footprint"][target_index]) <= keep_eps
                else "MOVE"
            )
            edits.append(
                _point_edit(
                    action,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_point_index=source_index,
                    target_point_index=target_index,
                    source_point_id=source_id,
                    target_point_id=source_id,
                    keep_eps=keep_eps,
                )
            )

        for source_index, source_id in enumerate(source_ids):
            if source_id not in target_by_id:
                edits.append(
                    _point_edit(
                        "DELETE",
                        source_layer=source_layer,
                        target_layer=None,
                        source_point_index=source_index,
                        target_point_index=None,
                        source_point_id=source_id,
                        target_point_id=None,
                        keep_eps=keep_eps,
                    )
                )

        for target_index, target_id in enumerate(target_ids):
            if target_id not in source_id_set:
                edits.append(
                    _point_edit(
                        "INSERT",
                        source_layer=source_layer,
                        target_layer=target_layer,
                        source_point_index=None,
                        target_point_index=target_index,
                        source_point_id=None,
                        target_point_id=target_id,
                        keep_eps=keep_eps,
                    )
                )
        return edits, False

    source_count = len(source_layer.get("footprint") or [])
    target_count = len(target_layer.get("footprint") or [])
    paired = min(source_count, target_count)
    for index in range(paired):
        action = (
            "KEEP"
            if _point_distance(source_layer["footprint"][index], target_layer["footprint"][index]) <= keep_eps
            else "MOVE"
        )
        edits.append(
            _point_edit(
                action,
                source_layer=source_layer,
                target_layer=target_layer,
                source_point_index=index,
                target_point_index=index,
                source_point_id=None,
                target_point_id=None,
                keep_eps=keep_eps,
            )
        )
    for index in range(paired, source_count):
        edits.append(
            _point_edit(
                "DELETE",
                source_layer=source_layer,
                target_layer=None,
                source_point_index=index,
                target_point_index=None,
                source_point_id=None,
                target_point_id=None,
                keep_eps=keep_eps,
            )
        )
    for index in range(paired, target_count):
        edits.append(
            _point_edit(
                "INSERT",
                source_layer=source_layer,
                target_layer=target_layer,
                source_point_index=None,
                target_point_index=index,
                source_point_id=None,
                target_point_id=None,
                keep_eps=keep_eps,
            )
        )
    return edits, True


def _point_edits_for_inserted_layer(target_layer: dict, keep_eps: float) -> list[dict]:
    target_ids = _point_ids(target_layer) or [None] * len(target_layer.get("footprint") or [])
    return [
        _point_edit(
            "INSERT",
            source_layer=None,
            target_layer=target_layer,
            source_point_index=None,
            target_point_index=index,
            source_point_id=None,
            target_point_id=target_ids[index],
            keep_eps=keep_eps,
        )
        for index in range(len(target_layer.get("footprint") or []))
    ]


def _point_edits_for_deleted_layer(source_layer: dict, keep_eps: float) -> list[dict]:
    source_ids = _point_ids(source_layer) or [None] * len(source_layer.get("footprint") or [])
    return [
        _point_edit(
            "DELETE",
            source_layer=source_layer,
            target_layer=None,
            source_point_index=index,
            target_point_index=None,
            source_point_id=source_ids[index],
            target_point_id=None,
            keep_eps=keep_eps,
        )
        for index in range(len(source_layer.get("footprint") or []))
    ]


def canonicalize_edit_object(
    history_layers: Sequence[dict],
    target_layers: Sequence[dict],
    keep_eps: float,
) -> tuple[list[dict], bool]:
    matches = _match_layers(history_layers, target_layers)
    matched_targets = set(matches.values())
    edit_objects: list[dict] = []
    fallback_alignment_used = False

    for source_index, source_layer in enumerate(history_layers):
        target_index = matches.get(source_index)
        if target_index is None:
            edit_objects.append(
                {
                    "action": "DELETE",
                    "source_layer_index": source_index,
                    "target_layer_index": None,
                    "source_proxy_id": int(source_layer.get("proxy_id", source_index)),
                    "target_proxy_id": None,
                    "source_point_count": len(source_layer.get("footprint") or []),
                    "target_point_count": 0,
                    "height_edit": {
                        "source_min_height": float(source_layer["min_height"]),
                        "source_max_height": float(source_layer["max_height"]),
                        "target_min_height": None,
                        "target_max_height": None,
                    },
                    "point_edits": _point_edits_for_deleted_layer(source_layer, keep_eps),
                }
            )
            continue

        target_layer = target_layers[target_index]
        point_edits, point_fallback = _point_edits_for_matched_layer(source_layer, target_layer, keep_eps)
        fallback_alignment_used = fallback_alignment_used or point_fallback
        same_heights = not _height_changed(source_layer, target_layer, keep_eps)
        same_points = _same_point_geometry(source_layer, target_layer, keep_eps)
        action = "KEEP" if same_heights and same_points else "MODIFY"
        edit_objects.append(
            {
                "action": action,
                "source_layer_index": source_index,
                "target_layer_index": target_index,
                "source_proxy_id": int(source_layer.get("proxy_id", source_index)),
                "target_proxy_id": int(target_layer.get("proxy_id", target_index)),
                "source_point_count": len(source_layer.get("footprint") or []),
                "target_point_count": len(target_layer.get("footprint") or []),
                "height_edit": {
                    "source_min_height": float(source_layer["min_height"]),
                    "source_max_height": float(source_layer["max_height"]),
                    "target_min_height": float(target_layer["min_height"]),
                    "target_max_height": float(target_layer["max_height"]),
                },
                "point_edits": point_edits,
            }
        )

    for target_index, target_layer in enumerate(target_layers):
        if target_index in matched_targets:
            continue
        edit_objects.append(
            {
                "action": "INSERT",
                "source_layer_index": None,
                "target_layer_index": target_index,
                "source_proxy_id": None,
                "target_proxy_id": int(target_layer.get("proxy_id", target_index)),
                "source_point_count": 0,
                "target_point_count": len(target_layer.get("footprint") or []),
                "height_edit": {
                    "source_min_height": None,
                    "source_max_height": None,
                    "target_min_height": float(target_layer["min_height"]),
                    "target_max_height": float(target_layer["max_height"]),
                },
                "point_edits": _point_edits_for_inserted_layer(target_layer, keep_eps),
            }
        )
    return edit_objects, fallback_alignment_used


def compile_v2_sequence_from_edit_object(edit_objects: Sequence[dict]) -> list[dict]:
    sequence: list[dict] = []
    for layer_edit in edit_objects:
        action = str(layer_edit["action"])
        sequence.append(
            {
                "type": f"{action}_LAYER",
                "value": {
                    "source_layer_index": layer_edit.get("source_layer_index"),
                    "target_layer_index": layer_edit.get("target_layer_index"),
                    "source_proxy_id": layer_edit.get("source_proxy_id"),
                    "target_proxy_id": layer_edit.get("target_proxy_id"),
                },
            }
        )
        if action == "INSERT":
            height_type = "ADD_HEIGHT"
        elif action == "KEEP":
            height_type = "KEEP_HEIGHT"
        else:
            height_type = "MODIFY_HEIGHT"
        sequence.append({"type": height_type, "value": layer_edit["height_edit"]})
        for point_edit in layer_edit.get("point_edits") or []:
            sequence.append({"type": f"{point_edit['action']}_POINT", "value": point_edit})
    return sequence


def apply_v2_sequence_to_layers(source_layers: Sequence[dict], v2_sequence: Sequence[dict]) -> list[dict]:
    layers: list[dict] = []
    current_layer: dict | None = None
    current_action: str | None = None
    current_source_layer_index: int | None = None
    current_points: dict[int, list[float]] = {}
    current_append_index = 0

    def source_layer_at(index: int | None) -> dict | None:
        if index is None or index < 0 or index >= len(source_layers):
            return None
        return source_layers[index]

    def source_point_at(layer: dict | None, index: int | None) -> list[float] | None:
        if layer is None or index is None:
            return None
        footprint = layer.get("footprint") or []
        if index < 0 or index >= len(footprint):
            return None
        point = footprint[index]
        return [float(point[0]), float(point[1])]

    def finish_current_layer() -> None:
        nonlocal current_layer, current_action, current_source_layer_index, current_points, current_append_index
        if current_layer is None or current_action == "DELETE":
            current_layer = None
            current_action = None
            current_source_layer_index = None
            current_points = {}
            current_append_index = 0
            return
        footprint = [point for _, point in sorted(current_points.items(), key=lambda item: item[0])]
        current_layer["footprint"] = footprint
        if len(footprint) >= 3 and float(current_layer["max_height"]) > float(current_layer["min_height"]):
            layers.append(current_layer)
        current_layer = None
        current_action = None
        current_source_layer_index = None
        current_points = {}
        current_append_index = 0

    for entry in v2_sequence:
        if not isinstance(entry, dict):
            continue
        token_type = str(entry.get("type", ""))
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}

        if token_type.endswith("_LAYER"):
            finish_current_layer()
            current_action = token_type[: -len("_LAYER")]
            if current_action == "DELETE":
                current_layer = None
                current_source_layer_index = None
                continue
            target_layer_index = value.get("target_layer_index")
            source_layer_index = value.get("source_layer_index")
            target_proxy_id = value.get("target_proxy_id")
            source_proxy_id = value.get("source_proxy_id")
            current_source_layer_index = int(source_layer_index) if source_layer_index is not None else None
            source_layer = source_layer_at(current_source_layer_index)
            layer_index = target_layer_index if target_layer_index is not None else source_layer_index
            if layer_index is None:
                layer_index = len(layers)
            current_layer = {
                "proxy_id": int(
                    target_proxy_id
                    if target_proxy_id is not None
                    else source_layer.get("proxy_id", layer_index)
                    if source_layer is not None
                    else layer_index
                ),
                "source_proxy_id": int(
                    source_proxy_id
                    if source_proxy_id is not None
                    else source_layer.get("source_proxy_id", source_layer.get("proxy_id", layer_index))
                    if source_layer is not None
                    else target_proxy_id
                    if target_proxy_id is not None
                    else layer_index
                ),
                "level_index": int(
                    source_layer.get("level_index", layer_index)
                    if source_layer is not None and target_layer_index is None
                    else layer_index
                ),
                "min_height": float(source_layer.get("min_height", 0.0)) if source_layer is not None else 0.0,
                "max_height": float(source_layer.get("max_height", 0.0)) if source_layer is not None else 0.0,
                "footprint": [],
            }
            continue

        if current_layer is None:
            continue

        if token_type in {"KEEP_HEIGHT", "MODIFY_HEIGHT", "ADD_HEIGHT"}:
            target_min = value.get("target_min_height")
            target_max = value.get("target_max_height")
            if target_min is None:
                target_min = value.get("source_min_height", current_layer["min_height"])
            if target_max is None:
                target_max = value.get("source_max_height", current_layer["max_height"])
            current_layer["min_height"] = float(target_min)
            current_layer["max_height"] = float(target_max)
            continue

        if token_type in {"KEEP_POINT", "MOVE_POINT", "INSERT_POINT"}:
            target_index = value.get("target_point_index")
            point_index = int(target_index) if target_index is not None else current_append_index
            current_append_index = max(current_append_index, point_index + 1)

            source_index = value.get("source_point_index")
            source_index_int = int(source_index) if source_index is not None else None
            source_layer = source_layer_at(current_source_layer_index)

            if token_type == "KEEP_POINT":
                point_value = source_point_at(source_layer, source_index_int) or value.get("source_coord") or value.get("target_coord")
            elif token_type == "MOVE_POINT":
                base_point = source_point_at(source_layer, source_index_int) or value.get("source_coord")
                delta = value.get("value") or [0.0, 0.0]
                if base_point is not None:
                    point_value = [float(base_point[0]) + float(delta[0]), float(base_point[1]) + float(delta[1])]
                else:
                    point_value = value.get("target_coord")
            else:
                point_value = value.get("target_coord")

            if point_value is None:
                continue
            current_points[point_index] = [float(point_value[0]), float(point_value[1])]

    finish_current_layer()
    layers.sort(
        key=lambda layer: (
            int(layer.get("level_index", 0)),
            float(layer.get("min_height", 0.0)),
            float(layer.get("max_height", 0.0)),
            int(layer.get("proxy_id", 0)),
        )
    )
    return layers


def export_v2_sequence_preview_obj(source_layers: Sequence[dict], v2_sequence_path: Path, obj_path: Path) -> bool:
    v2_sequence = dataset_stage.load_yaml(v2_sequence_path) or []
    if not isinstance(v2_sequence, list):
        raise ValueError(f"Malformed v2 edit sequence: {v2_sequence_path}")
    layers = apply_v2_sequence_to_layers(source_layers, v2_sequence)
    if not layers:
        return False
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_stage._export_stage_obj(layers, obj_path)
    return obj_path.exists()


def _append_old_layer_sequence(sequence: list[dict], layer_edit: dict) -> None:
    action = str(layer_edit["action"])
    height = layer_edit["height_edit"]
    sequence.append({"type": "LAYER_START"})

    if action == "INSERT":
        sequence.append({"type": "ADD_MIN_HEIGHT", "value": float(height["target_min_height"] or 0.0)})
        sequence.append({"type": "ADD_MAX_HEIGHT", "value": float(height["target_max_height"] or 0.0)})
        point_edits = sorted(
            (edit for edit in layer_edit.get("point_edits") or [] if edit.get("target_point_index") is not None),
            key=lambda edit: int(edit["target_point_index"]),
        )
        for point_edit in point_edits:
            target_coord = point_edit.get("target_coord") or [0.0, 0.0]
            sequence.append({"type": "ADD_POINT", "value": [float(target_coord[0]), float(target_coord[1])]})
        sequence.append({"type": "LAYER_END"})
        return

    if action == "DELETE":
        sequence.append({"type": "MIN_HEIGHT", "value": 0.0})
        sequence.append({"type": "MAX_HEIGHT", "value": 0.0})
        for _ in range(int(layer_edit.get("source_point_count", 0))):
            sequence.append({"type": "DELETE_POINT"})
        sequence.append({"type": "LAYER_END"})
        return

    min_delta = float(height["target_min_height"]) - float(height["source_min_height"])
    max_delta = float(height["target_max_height"]) - float(height["source_max_height"])
    sequence.append({"type": "MIN_HEIGHT", "value": min_delta})
    sequence.append({"type": "MAX_HEIGHT", "value": max_delta})

    target_order = sorted(
        (
            edit
            for edit in layer_edit.get("point_edits") or []
            if edit.get("target_point_index") is not None and edit.get("action") != "DELETE"
        ),
        key=lambda edit: int(edit["target_point_index"]),
    )
    source_indices_in_target_order = [
        int(edit["source_point_index"])
        for edit in target_order
        if edit.get("source_point_index") is not None
    ]
    is_monotonic_source_order = all(
        prev < curr
        for prev, curr in zip(source_indices_in_target_order, source_indices_in_target_order[1:])
    )
    if not is_monotonic_source_order:
        # Old edit streams consume source points once from left to right. Clipped polygons can
        # rotate their start vertex, so fall back to a deterministic delete-then-add stream.
        for _ in range(int(layer_edit.get("source_point_count", 0))):
            sequence.append({"type": "DELETE_POINT"})
        for point_edit in target_order:
            target_coord = point_edit.get("target_coord") or [0.0, 0.0]
            sequence.append({"type": "ADD_POINT", "value": [float(target_coord[0]), float(target_coord[1])]})
        sequence.append({"type": "LAYER_END"})
        return

    source_cursor = 0
    source_count = int(layer_edit.get("source_point_count", 0))
    for point_edit in target_order:
        source_index = point_edit.get("source_point_index")
        if source_index is None:
            target_coord = point_edit.get("target_coord") or [0.0, 0.0]
            sequence.append({"type": "ADD_POINT", "value": [float(target_coord[0]), float(target_coord[1])]})
            continue
        source_index = int(source_index)
        while source_cursor < source_index:
            sequence.append({"type": "DELETE_POINT"})
            source_cursor += 1
        value = point_edit.get("value") or [0.0, 0.0]
        sequence.append({"type": "MOVE_POINT", "value": [float(value[0]), float(value[1])]})
        source_cursor = max(source_cursor, source_index + 1)
    while source_cursor < source_count:
        sequence.append({"type": "DELETE_POINT"})
        source_cursor += 1
    sequence.append({"type": "LAYER_END"})


def compile_old_sequence_from_edit_object(edit_objects: Sequence[dict]) -> list[dict]:
    sequence: list[dict] = [{"type": "BOS"}]
    target_order_edits = sorted(
        (edit for edit in edit_objects if edit.get("target_layer_index") is not None),
        key=lambda edit: int(edit["target_layer_index"]),
    )
    source_only_edits = {
        int(edit["source_layer_index"]): edit
        for edit in edit_objects
        if edit.get("source_layer_index") is not None and edit.get("target_layer_index") is None
    }
    source_indices = [
        int(edit["source_layer_index"])
        for edit in edit_objects
        if edit.get("source_layer_index") is not None
    ]
    source_count = max(source_indices) + 1 if source_indices else 0
    history_cursor = 0

    for layer_edit in target_order_edits:
        source_index = layer_edit.get("source_layer_index")
        if source_index is None:
            _append_old_layer_sequence(sequence, layer_edit)
            continue
        source_index = int(source_index)
        while history_cursor < source_index:
            delete_edit = source_only_edits.get(history_cursor)
            if delete_edit is not None:
                _append_old_layer_sequence(sequence, delete_edit)
            history_cursor += 1
        _append_old_layer_sequence(sequence, layer_edit)
        history_cursor = source_index + 1

    while history_cursor < source_count:
        delete_edit = source_only_edits.get(history_cursor)
        if delete_edit is not None:
            _append_old_layer_sequence(sequence, delete_edit)
        history_cursor += 1

    sequence.append({"type": "EOS"})
    return sequence


def _validate_point_lineage(layers: Sequence[dict], path: Path) -> None:
    for layer_index, layer in enumerate(layers):
        footprint = layer.get("footprint") or []
        point_ids = layer.get("point_ids") or []
        if len(point_ids) != len(footprint):
            raise ValueError(f"{path}: layer {layer_index} has mismatched footprint/point_ids length.")
        if len(set(int(point_id) for point_id in point_ids)) != len(point_ids):
            raise ValueError(f"{path}: layer {layer_index} has duplicate point_ids.")


def build_pair_dataset_canonical(
    sequence_dir: Path,
    output_dir: Path,
    pair_mode: str,
    condition_point_count: int,
    change_tolerance: float,
    point_keep_eps: float,
    copy_objs: bool,
    save_condition_ply_flag: bool,
    export_v2_preview_obj_flag: bool,
) -> None:
    sequence_meta = dataset_stage.load_yaml(sequence_dir / "sequence_meta.yaml") or {}
    stage_names = [
        dataset_stage.normalize_stage_name(stage["stage_name"])
        for stage in sequence_meta.get("stages", [])
        if isinstance(stage, dict) and "stage_name" in stage
    ]
    if not stage_names:
        stage_names = sorted(p.name for p in sequence_dir.iterdir() if p.is_dir() and p.name.startswith("stage_"))

    normalized_stage_dirs: dict[str, Path] = {}
    for stage_name in stage_names:
        normalized_stage_dirs[stage_name] = dataset_stage.ensure_stage_copy(
            sequence_dir,
            output_dir,
            stage_name,
            copy_objs=copy_objs,
        )

    pairs = dataset_stage.enumerate_pairs(stage_names, pair_mode)
    pair_records: list[dict] = []
    fallback_pair_count = 0
    for src_stage, dst_stage in pairs:
        history_yaml = sequence_dir / src_stage / "building1.yaml"
        target_yaml = sequence_dir / dst_stage / "building1.yaml"
        history_layers = _normalize_layers_with_lineage(history_yaml)
        target_layers = _normalize_layers_with_lineage(target_yaml)
        _validate_point_lineage(history_layers, history_yaml)
        _validate_point_lineage(target_layers, target_yaml)

        edit_objects, fallback_alignment_used = canonicalize_edit_object(
            history_layers,
            target_layers,
            point_keep_eps,
        )
        fallback_pair_count += int(fallback_alignment_used)
        pair_name = f"{src_stage}_to_{dst_stage}"

        edit_object_path = output_dir / "edit_objects" / f"{pair_name}.yaml"
        v2_path = output_dir / "edit_sequences_v2" / f"{pair_name}.yaml"
        old_path = output_dir / "edit_sequences" / f"{pair_name}.yaml"
        dataset_stage.save_yaml(edit_object_path, edit_objects)
        dataset_stage.save_yaml(v2_path, compile_v2_sequence_from_edit_object(edit_objects))
        dataset_stage.save_yaml(old_path, compile_old_sequence_from_edit_object(edit_objects))
        preview_obj_path = output_dir / "preview_objs" / f"{pair_name}_from_v2.obj"
        v2_preview_obj_written = False
        if export_v2_preview_obj_flag:
            v2_preview_obj_written = export_v2_sequence_preview_obj(history_layers, v2_path, preview_obj_path)

        changed_layers = dataset_stage.select_changed_target_layers(
            history_layers,
            target_layers,
            change_tolerance,
        )
        condition_points = dataset_stage.sample_condition_points(changed_layers, condition_point_count)
        condition_path = output_dir / "conditions" / f"{pair_name}_r0.pt"
        condition_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_stage.torch.save(dataset_stage.torch.from_numpy(condition_points), condition_path)
        condition_ply_path = condition_path.with_suffix(".ply")
        if save_condition_ply_flag:
            dataset_stage.save_condition_ply(condition_ply_path, condition_points)

        pair_meta = {
            "pair_name": pair_name,
            "sequence_dir": str(sequence_dir.resolve()),
            "history_stage": src_stage,
            "target_stage": dst_stage,
            "history_yaml": str(history_yaml.resolve()),
            "target_yaml": str(target_yaml.resolve()),
            "canonical_supervision": True,
            "point_id_supervision": not fallback_alignment_used,
            "fallback_alignment_used": bool(fallback_alignment_used),
            "edit_object": str(edit_object_path.resolve()),
            "v2_edit_sequence": str(v2_path.resolve()),
            "old_edit_sequence": str(old_path.resolve()),
            "edit_sequence": str(old_path.resolve()),
            "v2_preview_obj": str(preview_obj_path.resolve()) if v2_preview_obj_written else "",
            "condition": str(condition_path.resolve()),
            "condition_ply": str(condition_ply_path.resolve()) if save_condition_ply_flag else "",
            "changed_layer_count": len(changed_layers),
            "condition_point_count": int(condition_points.shape[0]),
        }
        dataset_stage.save_yaml(output_dir / "pair_meta" / f"{pair_name}.yaml", pair_meta)

        pair_records.append(
            {
                "t1": str(normalized_stage_dirs[src_stage]),
                "t2": str(normalized_stage_dirs[dst_stage]),
                "condition": str(condition_path.resolve()),
                "edit_sequence": str(old_path.resolve()),
                "edit_object": str(edit_object_path.resolve()),
                "v2_edit_sequence": str(v2_path.resolve()),
                "v2_preview_obj": str(preview_obj_path.resolve()) if v2_preview_obj_written else "",
            }
        )

    for split in ("train", "val", "test"):
        dataset_stage.save_yaml(output_dir / f"{split}.yaml", pair_records)
    dataset_stage.save_yaml(
        output_dir / "dataset_meta.yaml",
        {
            "sequence_dir": str(sequence_dir.resolve()),
            "pair_mode": pair_mode,
            "pair_count": len(pair_records),
            "condition_point_count": int(condition_point_count),
            "change_tolerance": float(change_tolerance),
            "point_keep_eps": float(point_keep_eps),
            "stage_names": stage_names,
            "canonical_supervision": True,
            "stable_point_ids": True,
            "v2_preview_obj": bool(export_v2_preview_obj_flag),
            "fallback_pair_count": int(fallback_pair_count),
        },
    )


def _dataset_dir_for_sequence(dataset_output_root: Path | None, sequence_dir: Path, yaml_path: Path) -> Path:
    if dataset_output_root is None:
        return sequence_dir / "layer_edit_dataset"
    return dataset_output_root / yaml_path.stem / sequence_dir.name


def _generate_sequence_for_yaml(
    yaml_path: Path,
    sequence_output_root: Path,
    dataset_output_root: Path | None,
    args: argparse.Namespace,
) -> int:
    sequence_seed = sequence_stage._sequence_seed(args.seed, yaml_path)
    rng = random.Random(sequence_seed)
    policy = sequence_stage._choose_policy(args.policy, rng)
    axis = sequence_stage._choose_axis(args.axis, rng) if policy in {"footprint", "hybrid"} else None
    keep_side = (
        sequence_stage._choose_keep_side(args.keep_side, rng)
        if policy in {"footprint", "hybrid"}
        else None
    )

    entries = _assign_source_point_ids(proxy_stage._load_yaml_entries(yaml_path))
    if not entries:
        print(f"[SKIP] {yaml_path}: no entries")
        return 0

    vertical_progress = (
        sequence_stage._build_progression(args.stage_count, args.min_progress, rng)
        if policy in {"vertical", "hybrid"}
        else None
    )
    footprint_progress = (
        sequence_stage._build_progression(args.stage_count, args.min_progress, rng)
        if policy in {"footprint", "hybrid"}
        else None
    )

    source_root = sequence_output_root / yaml_path.stem
    sequence_dir = source_root / sequence_stage._sequence_dir_name(policy, sequence_seed)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    next_proxy_id = max(int(entry["proxy_id"]) for entry in entries) + 1
    used_proxy_ids: set[int] = set()
    previous_entries: list[dict] = []
    stage_records: list[dict] = []

    for stage_index in range(args.stage_count):
        raw_stage_entries, metadata, stage_record = sequence_stage._generate_stage_entries(
            policy=policy,
            stage_index=stage_index,
            entries=entries,
            vertical_progress=vertical_progress,
            footprint_progress=footprint_progress,
            axis=axis,
            keep_side=keep_side,
            min_area=args.min_area,
            stage_count=args.stage_count,
        )
        stage_entries, next_proxy_id = sequence_stage._stabilize_stage_entries(
            raw_stage_entries,
            previous_entries,
            used_proxy_ids,
            next_proxy_id,
        )
        metadata = {
            **metadata,
            "policy": policy,
            "sequence_seed": int(sequence_seed),
            "stage_index": stage_index + 1,
            "stage_count": args.stage_count,
            "source_entry_count": len(entries),
            "stable_proxy_ids": True,
            "stable_point_ids": True,
            "point_keep_eps": float(args.point_keep_eps),
        }
        stage_entries = _annotate_stage_point_lineage(
            stage_entries,
            entries,
            metadata,
            float(args.point_keep_eps),
        )

        stage_dir = sequence_dir / stage_record["stage_name"]
        _write_stage_outputs(stage_entries, yaml_path, stage_dir, metadata, args.export_obj)

        previous_entries = [dict(entry) for entry in stage_entries]
        stage_record["entry_count"] = len(stage_entries)
        stage_record["point_lineage_enabled"] = True
        stage_records.append(stage_record)
        print(
            f"[OK] {yaml_path.name} -> {stage_dir.relative_to(sequence_output_root)}: "
            f"{len(stage_entries)} entries"
        )

    _write_sequence_meta(
        sequence_dir,
        source_yaml=yaml_path,
        sequence_seed=sequence_seed,
        policy=policy,
        axis=axis,
        keep_side=keep_side,
        stage_records=stage_records,
        point_keep_eps=float(args.point_keep_eps),
    )

    dataset_dir = _dataset_dir_for_sequence(dataset_output_root, sequence_dir, yaml_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build_pair_dataset_canonical(
        sequence_dir=sequence_dir,
        output_dir=dataset_dir,
        pair_mode=args.pair_mode,
        condition_point_count=args.condition_point_count,
        change_tolerance=args.change_tolerance,
        point_keep_eps=float(args.point_keep_eps),
        copy_objs=args.copy_objs,
        save_condition_ply_flag=args.save_condition_ply,
        export_v2_preview_obj_flag=not args.no_v2_preview_obj,
    )
    print(f"[DATASET] {yaml_path.name} -> {dataset_dir}")
    return len(stage_records)


def main() -> int:
    args = parse_args()
    yaml_files = proxy_stage._discover_yaml_files(args.input.resolve())
    if not yaml_files:
        print("No input YAML files found.")
        return 1

    sequence_output_root = args.output.resolve()
    dataset_output_root = args.dataset_output.resolve() if args.dataset_output is not None else None
    generated = 0
    for yaml_path in yaml_files:
        generated += _generate_sequence_for_yaml(yaml_path, sequence_output_root, dataset_output_root, args)

    print(f"Generated {generated} stage(s) under {sequence_output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

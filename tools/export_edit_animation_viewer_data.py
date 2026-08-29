import argparse
import json
import pathlib
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from gendiff_data_process.viewer_packed import (
    SPLITS,
    PackedSample,
    PackedViewerError,
    find_packed_sample,
    is_packed_dataset,
    iter_packed_samples,
    load_packed_metadata,
    load_packed_states_payload,
    packed_sample_count,
)

POINT_OP_TYPES = {"KEEP_POINT", "MOVE_POINT", "INSERT_POINT", "DELETE_POINT"}
LAYER_OP_TYPES = {
    "KEEP_LAYER",
    "MODIFY_LAYER",
    "INSERT_LAYER",
    "DELETE_LAYER",
    "REMOVE_LAYER",
}
HEIGHT_OP_TYPES = {"KEEP_HEIGHT", "MODIFY_HEIGHT", "ADD_HEIGHT"}


class ExportError(RuntimeError):
    pass


def read_yaml(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def natural_key(path: pathlib.Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def as_vec2_list(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, list) or len(value) < 1:
        return None
    points: List[List[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            return None
    return points


def recursive_items(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from recursive_items(value)
    elif isinstance(node, list):
        for item in node:
            yield from recursive_items(item)


def find_stage_file(dataset_dir: pathlib.Path, stage_id: str) -> pathlib.Path:
    stage_dir = dataset_dir / "stages" / stage_id
    if not stage_dir.exists():
        raise ExportError(f"stage directory not found: {stage_dir}")
    preferred = stage_dir / f"bs_{stage_id}_r0.yaml"
    if preferred.exists():
        return preferred
    candidates = [p for p in stage_dir.glob("*.yaml") if "meta" not in p.name.lower()]
    if not candidates:
        raise ExportError(f"no stage geometry YAML found under {stage_dir}")
    return sorted(candidates, key=natural_key)[0]


def find_state_file(dataset_dir: pathlib.Path, state_id: str) -> pathlib.Path:
    state_dir = dataset_dir / "states" / state_id
    if not state_dir.exists():
        raise ExportError(f"state directory not found: {state_dir}")
    preferred = state_dir / f"bs_{state_id}_r0.yaml"
    if preferred.exists():
        return preferred
    candidates = [p for p in state_dir.glob("*.yaml") if "meta" not in p.name.lower()]
    if not candidates:
        raise ExportError(f"no state geometry YAML found under {state_dir}")
    return sorted(candidates, key=natural_key)[0]


def load_state_meta(dataset_dir: pathlib.Path, state_id: str) -> Dict[str, Any]:
    meta_path = dataset_dir / "states" / state_id / "area_state_meta.yaml"
    if not meta_path.exists():
        return {}
    meta = read_yaml(meta_path)
    return meta if isinstance(meta, dict) else {}


def infer_layer_id(item: Dict[str, Any], index: int) -> str:
    for key in ("layer_id", "proxy_id", "target_proxy_id", "source_proxy_id", "id"):
        if key in item and item[key] is not None:
            return str(item[key])
    return str(index)


def to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_stage_layers_data(
    data: Any, source_name: str, allow_empty: bool = False
) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for item in recursive_items(data):
        if id(item) in seen:
            continue
        seen.add(id(item))
        footprint = as_vec2_list(item.get("footprint"))
        if footprint is None:
            continue
        raw_index = len(layers)
        min_height = to_float(
            item.get("min_height", item.get("source_min_height", 0.0)), 0.0
        )
        max_height = to_float(
            item.get("max_height", item.get("source_max_height", min_height + 1.0)),
            min_height + 1.0,
        )
        layers.append(
            {
                "layer_id": infer_layer_id(item, raw_index),
                "proxy_id": item.get(
                    "proxy_id", item.get("source_proxy_id", item.get("target_proxy_id"))
                ),
                "source_proxy_id": item.get("source_proxy_id"),
                "target_proxy_id": item.get("target_proxy_id"),
                "local_proxy_id": item.get("local_proxy_id"),
                "building_id": item.get("building_id"),
                "building_name": item.get("building_name"),
                "building_stage_name": item.get("building_stage_name"),
                "building_layer_index": item.get("building_layer_index"),
                "level_index": to_optional_int(item.get("level_index")),
                "points": footprint,
                "height": [min_height, max_height],
                "point_ids": item.get("point_ids"),
                "source_point_ids": item.get("source_point_ids"),
                "point_roles": item.get("point_roles"),
                "raw_order": raw_index,
            }
        )
    if not layers and not allow_empty:
        raise ExportError(f"could not parse any footprint geometry from {source_name}")
    return sorted(
        layers,
        key=lambda layer: (
            (
                layer["building_id"]
                if to_optional_int(layer.get("building_id")) is not None
                else -1
            ),
            layer["level_index"] if layer["level_index"] is not None else 10**9,
            str(layer["proxy_id"]) if layer["proxy_id"] is not None else "",
            layer["raw_order"],
        ),
    )


def extract_stage_layers(
    stage_file: pathlib.Path, allow_empty: bool = False
) -> List[Dict[str, Any]]:
    return extract_stage_layers_data(
        read_yaml(stage_file), str(stage_file), allow_empty
    )


def split_pair_id(pair_id: str) -> Tuple[str, str]:
    if "_to_" not in pair_id:
        raise ExportError(
            f"cannot parse source/target stage ids from pair file name: {pair_id}"
        )
    source_stage_id, target_stage_id = pair_id.split("_to_", 1)
    return source_stage_id, target_stage_id


def resolve_pair_states(pair_id: str, pair_meta: Any) -> Tuple[str, str]:
    if isinstance(pair_meta, dict):
        source = pair_meta.get("source_state")
        target = pair_meta.get("target_state")
        if isinstance(source, str) and isinstance(target, str):
            return source, target
    match = re.search(r"(area_state_\d+)_to_(area_state_\d+)", pair_id)
    if not match:
        raise ExportError(
            f"cannot parse source/target area state ids from pair file name: {pair_id}"
        )
    return match.group(1), match.group(2)


def load_optional(
    dataset_dir: pathlib.Path, folder: str, pair_id: str
) -> Tuple[Optional[pathlib.Path], Any]:
    path = dataset_dir / folder / f"{pair_id}.yaml"
    if not path.exists():
        return None, None
    return path, read_yaml(path)


def same_ref(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    return str(a) == str(b)


def layer_uid(layer: Dict[str, Any]) -> str:
    return str(layer.get("raw_order", layer.get("layer_id")))


def has_area_layer_identity(layers: Sequence[Dict[str, Any]]) -> bool:
    return any(
        layer.get("building_id") is not None
        or layer.get("building_layer_index") is not None
        for layer in layers
    )


def layer_has_proxy(layer: Dict[str, Any], proxy: Any) -> bool:
    return any(
        same_ref(layer.get(key), proxy)
        for key in ("proxy_id", "source_proxy_id", "target_proxy_id", "layer_id")
    )


def layer_identity_values(layer: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    if layer.get("building_id") is not None and layer.get("proxy_id") is not None:
        values.append(
            f"building:{layer.get('building_id')}:proxy:{layer.get('proxy_id')}"
        )
    if (
        layer.get("building_id") is not None
        and layer.get("building_layer_index") is not None
    ):
        values.append(
            f"building:{layer.get('building_id')}:layer:{layer.get('building_layer_index')}"
        )
    if layer.get("building_id") is not None and layer.get("local_proxy_id") is not None:
        values.append(
            f"building:{layer.get('building_id')}:local:{layer.get('local_proxy_id')}"
        )
    values.extend(
        [
            str(layer.get("proxy_id")),
            str(layer.get("source_proxy_id")),
            str(layer.get("target_proxy_id")),
            str(layer.get("layer_id")),
        ]
    )
    deduped: List[str] = []
    for value in values:
        if value not in {"None", ""} and value not in deduped:
            deduped.append(value)
    return deduped


def layer_lookup(layers: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for layer in layers:
        for value in layer_identity_values(layer):
            lookup.setdefault(value, layer)
    return lookup


def layer_matches_token_ref(
    value: Dict[str, Any],
    layer: Dict[str, Any],
    side: str,
    allow_global_index: bool = True,
) -> bool:
    building_id = value.get(f"{side}_building_id")
    building_layer_index = value.get(f"{side}_building_layer_index")
    proxy_id = value.get(f"{side}_proxy_id")
    layer_id = value.get(f"{side}_layer_id")
    layer_index = value.get(f"{side}_layer_index")

    if (
        building_id is not None
        and building_layer_index is not None
        and proxy_id is not None
    ):
        if (
            same_ref(layer.get("building_id"), building_id)
            and same_ref(layer.get("building_layer_index"), building_layer_index)
            and layer_has_proxy(layer, proxy_id)
        ):
            return True
    if building_id is not None and proxy_id is not None:
        if same_ref(layer.get("building_id"), building_id) and layer_has_proxy(
            layer, proxy_id
        ):
            return True
    if building_id is not None and building_layer_index is not None:
        if same_ref(layer.get("building_id"), building_id) and same_ref(
            layer.get("building_layer_index"), building_layer_index
        ):
            return True
    if proxy_id is not None and layer_has_proxy(layer, proxy_id):
        return True
    if layer_id is not None and layer_has_proxy(layer, layer_id):
        return True
    if (
        allow_global_index
        and layer_index is not None
        and same_ref(layer.get("raw_order"), layer_index)
    ):
        return True
    return False


def find_layer_from_token_ref(
    value: Dict[str, Any], layers: Sequence[Dict[str, Any]], side: str
) -> Optional[Dict[str, Any]]:
    for allow_global_index in (False, True):
        for layer in layers:
            if layer_matches_token_ref(
                value, layer, side, allow_global_index=allow_global_index
            ):
                return layer
    return None


def token_has_side_ref(value: Dict[str, Any], side: str) -> bool:
    return any(
        value.get(f"{side}_{key}") is not None
        for key in (
            "building_id",
            "building_layer_index",
            "proxy_id",
            "layer_id",
            "layer_index",
        )
    )


def token_matches_layer(
    value: Dict[str, Any],
    source: Optional[Dict[str, Any]],
    target: Optional[Dict[str, Any]],
) -> bool:
    has_source_ref = token_has_side_ref(value, "source")
    has_target_ref = token_has_side_ref(value, "target")
    source_ok = source is not None and layer_matches_token_ref(value, source, "source")
    target_ok = target is not None and layer_matches_token_ref(value, target, "target")
    if has_source_ref and has_target_ref:
        return source_ok and target_ok
    if has_source_ref:
        return source_ok
    if has_target_ref:
        return target_ok
    return False


def matched_layer_id(
    source: Optional[Dict[str, Any]], target: Optional[Dict[str, Any]], fallback: int
) -> str:
    ref = target or source
    if ref is None:
        return str(fallback)
    proxy = ref.get(
        "proxy_id",
        ref.get("target_proxy_id", ref.get("source_proxy_id", ref.get("layer_id"))),
    )
    building = ref.get("building_id")
    if building is not None and proxy is not None:
        return f"building:{building}:proxy:{proxy}"
    if proxy is not None:
        return str(proxy)
    if building is not None and ref.get("building_layer_index") is not None:
        return f"building:{building}:layer:{ref.get('building_layer_index')}"
    return str(ref.get("layer_id", fallback))


def match_layers(
    source_layers: Sequence[Dict[str, Any]],
    target_layers: Sequence[Dict[str, Any]],
    ops: Sequence[Dict[str, Any]] = (),
) -> List[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    source_by_id = layer_lookup(source_layers)
    matched: List[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
    used_source: set[str] = set()
    used_target: set[str] = set()
    area_mode = has_area_layer_identity(source_layers) or has_area_layer_identity(
        target_layers
    )

    for op in ops:
        op_type = op.get("type")
        if op_type not in LAYER_OP_TYPES:
            continue
        raw_value = op.get("value")
        value: Dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
        action = str(op_type).replace("_LAYER", "")
        source = (
            None
            if action == "INSERT"
            else find_layer_from_token_ref(value, source_layers, "source")
        )
        target = (
            None
            if action in {"DELETE", "REMOVE"}
            else find_layer_from_token_ref(value, target_layers, "target")
        )
        if source is None and target is None:
            continue
        source_uid = layer_uid(source) if source else None
        target_uid = layer_uid(target) if target else None
        if source_uid is not None and source_uid in used_source:
            continue
        if target_uid is not None and target_uid in used_target:
            continue
        matched.append((matched_layer_id(source, target, len(matched)), source, target))
        if source_uid is not None:
            used_source.add(source_uid)
        if target_uid is not None:
            used_target.add(target_uid)

    for target in target_layers:
        if layer_uid(target) in used_target:
            continue
        matched_source = None
        for identity in layer_identity_values(target):
            if identity in source_by_id:
                candidate = source_by_id[identity]
                if layer_uid(candidate) not in used_source:
                    matched_source = candidate
                    break
        if matched_source:
            matched.append(
                (
                    matched_layer_id(matched_source, target, len(matched)),
                    matched_source,
                    target,
                )
            )
            used_source.add(layer_uid(matched_source))
            used_target.add(layer_uid(target))

    remaining_sources = [
        layer for layer in source_layers if layer_uid(layer) not in used_source
    ]
    remaining_targets = [
        layer for layer in target_layers if layer_uid(layer) not in used_target
    ]
    if not area_mode:
        for index, target in enumerate(remaining_targets):
            source = (
                remaining_sources[index] if index < len(remaining_sources) else None
            )
            matched.append(
                (matched_layer_id(source, target, len(matched)), source, target)
            )
            if source:
                used_source.add(layer_uid(source))
            used_target.add(layer_uid(target))

    for target in target_layers:
        if layer_uid(target) not in used_target:
            matched.append((matched_layer_id(None, target, len(matched)), None, target))
            used_target.add(layer_uid(target))

    for source in source_layers:
        if layer_uid(source) not in used_source:
            matched.append((matched_layer_id(source, None, len(matched)), source, None))
            used_source.add(layer_uid(source))

    if not matched and (source_layers or target_layers):
        raise ExportError("no source/target layers could be matched")
    return matched


def normalize_layer_ref(
    value: Any,
    matched: Sequence[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    for lid, source, target in matched:
        ids = {lid}
        for layer in (source, target):
            if layer:
                ids.update(layer_identity_values(layer))
        if text in ids:
            return lid
    return None


def layer_ref_from_token(
    value: Dict[str, Any],
    matched: Sequence[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
) -> Optional[str]:
    source_matched = token_has_side_ref(value, "source")
    target_matched = token_has_side_ref(value, "target")
    for lid, source, target in matched:
        if token_matches_layer(value, source, target):
            return lid
    candidates = (
        value.get("target_proxy_id"),
        value.get("source_proxy_id"),
        value.get("target_layer_id"),
        value.get("source_layer_id"),
    )
    for candidate in candidates:
        normalized_lid = normalize_layer_ref(candidate, matched)
        if normalized_lid is not None:
            return normalized_lid
    if not source_matched and not target_matched:
        for candidate in (
            value.get("target_layer_index"),
            value.get("source_layer_index"),
        ):
            normalized_lid = normalize_layer_ref(candidate, matched)
            if normalized_lid is not None:
                return normalized_lid
    return None


def split_ops_by_layer(
    ops: List[Dict[str, Any]],
    matched: Sequence[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    by_layer: Dict[str, List[Dict[str, Any]]] = {lid: [] for lid, _, _ in matched}
    actions: Dict[str, str] = {}
    current_lid: Optional[str] = matched[0][0] if matched else None
    saw_layer_markers = False

    for op in ops:
        op_type = op.get("type")
        raw_value = op.get("value")
        value: Dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
        if op_type in LAYER_OP_TYPES:
            saw_layer_markers = True
            next_lid = layer_ref_from_token(value, matched)
            current_lid = next_lid or current_lid
            if current_lid is not None:
                actions[current_lid] = (
                    str(op_type).replace("_LAYER", "").replace("REMOVE", "DELETE")
                )
            continue
        if op_type in HEIGHT_OP_TYPES:
            continue
        if op_type not in POINT_OP_TYPES:
            continue
        if current_lid is not None:
            by_layer.setdefault(current_lid, []).append(op)

    if not saw_layer_markers and ops:
        point_ops = [op for op in ops if op.get("type") in POINT_OP_TYPES]
        # Some generated datasets do not carry reliable per-layer op markers.
        # In that case every matched layer receives the pair ops, and the viewer
        # will validate indices per layer when tracks are built.
        return {lid: list(point_ops) for lid, _, _ in matched}, actions
    return by_layer, actions


def debug_objects_for_layer(
    edit_objects: Any,
    lid: str,
    order: int,
    source: Optional[Dict[str, Any]],
    target: Optional[Dict[str, Any]],
) -> List[Any]:
    if isinstance(edit_objects, list):
        matches = []
        for item in edit_objects:
            if not isinstance(item, dict):
                continue
            if token_matches_layer(item, source, target):
                matches.append(item)
        return matches
    if isinstance(edit_objects, dict):
        value = edit_objects.get(lid, edit_objects.get(str(order), []))
        return value if isinstance(value, list) else [value]
    return []


def action_from_debug_objects(debug_objects: List[Any]) -> Optional[str]:
    for item in debug_objects:
        if isinstance(item, dict) and isinstance(item.get("action"), str):
            return item["action"]
    return None


def building_summaries(
    source_meta: Dict[str, Any], target_meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for side, meta in (("source", source_meta), ("target", target_meta)):
        selections = meta.get("selections") if isinstance(meta, dict) else None
        if not isinstance(selections, list):
            continue
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            building_id = selection.get("building_id")
            key = str(building_id)
            item = by_id.setdefault(
                key,
                {
                    "building_id": building_id,
                    "building_name": selection.get("building_name"),
                },
            )
            item[f"{side}_stage_name"] = selection.get("stage_name")
            item[f"{side}_stage_index"] = selection.get("stage_index")
            item[f"{side}_stage_position"] = selection.get("stage_position")
            if item.get("building_name") is None:
                item["building_name"] = selection.get("building_name")
    return sorted(
        by_id.values(),
        key=lambda item: (
            to_optional_int(item.get("building_id")) or 0,
            str(item.get("building_name") or ""),
        ),
    )


def packed_edit_objects_to_ops(edit_objects: Any) -> List[Dict[str, Any]]:
    if not isinstance(edit_objects, list):
        raise ExportError("packed sample edit_object 必须是 list")
    ops: List[Dict[str, Any]] = []
    for edit_index, edit_object in enumerate(edit_objects):
        if not isinstance(edit_object, dict):
            raise ExportError(f"packed edit_object[{edit_index}] 必须是 mapping")
        action = str(edit_object.get("action", "")).upper()
        if action not in {"KEEP", "MODIFY", "INSERT", "DELETE", "REMOVE"}:
            raise ExportError(f"packed layer action 不受支持: {action}")
        layer_value = {
            key: edit_object.get(key)
            for key in (
                "source_layer_index",
                "target_layer_index",
                "source_proxy_id",
                "target_proxy_id",
                "source_building_id",
                "target_building_id",
                "source_building_layer_index",
                "target_building_layer_index",
            )
            if edit_object.get(key) is not None
        }
        layer_type = "DELETE_LAYER" if action == "REMOVE" else f"{action}_LAYER"
        ops.append({"type": layer_type, "value": layer_value})
        point_edits = edit_object.get("point_edits", [])
        if not isinstance(point_edits, list):
            raise ExportError(
                f"packed edit_object[{edit_index}].point_edits 必须是 list"
            )
        for point_index, point_edit in enumerate(point_edits):
            if not isinstance(point_edit, dict):
                raise ExportError(
                    f"packed edit_object[{edit_index}].point_edits[{point_index}] 必须是 mapping"
                )
            point_action = str(point_edit.get("action", "")).upper()
            if point_action not in {"KEEP", "MOVE", "INSERT", "DELETE"}:
                raise ExportError(f"packed point action 不受支持: {point_action}")
            value = dict(point_edit)
            lineage = point_edit.get("point_lineage_id")
            if lineage is not None:
                if point_action != "INSERT":
                    value.setdefault("source_point_id", lineage)
                if point_action != "DELETE":
                    value.setdefault("target_point_id", lineage)
            if point_action in {"MOVE", "INSERT"} and value.get("target_coord") is None:
                value["target_coord"] = value.get("value")
            ops.append({"type": f"{point_action}_POINT", "value": value})
    return ops


def build_viewer_layers(
    source_layers: Sequence[Dict[str, Any]],
    target_layers: Sequence[Dict[str, Any]],
    ops: List[Dict[str, Any]],
    edit_objects: Any,
) -> List[Dict[str, Any]]:
    matched = match_layers(source_layers, target_layers, ops)
    ops_by_layer, layer_actions = split_ops_by_layer(ops, matched)
    layers: List[Dict[str, Any]] = []
    for order, (lid, source, target) in enumerate(
        sorted(
            matched,
            key=lambda item: (
                (
                    (item[1] or item[2] or {}).get("level_index")
                    if (item[1] or item[2] or {}).get("level_index") is not None
                    else 10**9
                ),
                str((item[1] or item[2] or {}).get("proxy_id", item[0])),
                (item[1] or item[2] or {}).get("raw_order", 0),
            ),
        )
    ):
        ref = source or target
        debug_objects = debug_objects_for_layer(
            edit_objects, lid, order, source, target
        )
        layer_action = layer_actions.get(lid) or action_from_debug_objects(
            debug_objects
        )
        if layer_action is None:
            layer_action = (
                "INSERT" if source is None else "DELETE" if target is None else "MODIFY"
            )
        source_height = (
            source["height"]
            if source
            else ([target["height"][0], target["height"][0]] if target else [0.0, 0.0])
        )
        target_height = (
            target["height"]
            if target
            else ([source["height"][0], source["height"][0]] if source else [0.0, 0.0])
        )
        layers.append(
            {
                "layer_id": lid,
                "layer_order": order,
                "level_index": ref.get("level_index") if ref else None,
                "building_id": ref.get("building_id") if ref else None,
                "building_name": ref.get("building_name") if ref else None,
                "building_stage_name": ref.get("building_stage_name") if ref else None,
                "building_layer_index": (
                    ref.get("building_layer_index") if ref else None
                ),
                "local_proxy_id": ref.get("local_proxy_id") if ref else None,
                "proxy_id": ref.get("proxy_id") if ref else None,
                "source_building_id": source.get("building_id") if source else None,
                "target_building_id": target.get("building_id") if target else None,
                "source_building_layer_index": (
                    source.get("building_layer_index") if source else None
                ),
                "target_building_layer_index": (
                    target.get("building_layer_index") if target else None
                ),
                "source_proxy_id": source.get("proxy_id") if source else None,
                "target_proxy_id": target.get("proxy_id") if target else None,
                "source_layer_index": source.get("raw_order") if source else None,
                "target_layer_index": target.get("raw_order") if target else None,
                "layer_action": layer_action,
                "source_points": source["points"] if source else [],
                "target_points": target["points"] if target else [],
                "source_height": source_height,
                "target_height": target_height,
                "ops": ops_by_layer.get(lid, []),
                "debug_edit_objects": debug_objects,
            }
        )
    return layers


def packed_building_summaries(
    source_layers: Sequence[Dict[str, Any]],
    target_layers: Sequence[Dict[str, Any]],
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for layer in [*source_layers, *target_layers]:
        building_id = layer.get("building_id")
        key = str(building_id)
        by_id.setdefault(
            key,
            {
                "building_id": building_id,
                "building_name": layer.get("building_name"),
            },
        )
    source_meta = source_state.get("meta")
    target_meta = target_state.get("meta")
    source_building = (
        source_meta.get("building_key") if isinstance(source_meta, Mapping) else None
    )
    target_building = (
        target_meta.get("building_key") if isinstance(target_meta, Mapping) else None
    )
    for item in by_id.values():
        item["source_stage_name"] = source_state.get("state_name")
        item["target_stage_name"] = target_state.get("state_name")
        if item.get("building_name") is None:
            item["building_name"] = target_building or source_building
    return sorted(
        by_id.values(),
        key=lambda item: (
            to_optional_int(item.get("building_id")) or 0,
            str(item.get("building_name") or ""),
        ),
    )


def packed_normalization(payload: Mapping[str, Any]) -> Dict[str, Any]:
    values = payload.get("normalization_stats_tensor")
    if values is not None and hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        return {}
    return {
        "center_x": float(values[0]),
        "center_z": float(values[1]),
        "scale_xz": float(values[2]),
        "center_y": float(values[3]),
        "scale_y": float(values[4]),
    }


def _state_name(state: Mapping[str, Any], index: int) -> str:
    value = state.get("state_name")
    return value if isinstance(value, str) and value else f"state_{index:06d}"


def _packed_pair(
    record: PackedSample,
    states: Sequence[Mapping[str, Any]],
    dataset_meta: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> Dict[str, Any]:
    sample = record.sample
    try:
        source_index = int(sample["source_state_index"])
        target_index = int(sample["target_state_index"])
        source_state = states[source_index]
        target_state = states[target_index]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ExportError(
            f"packed pair 的 source/target state index 无效: {sample.get('pair_name')}"
        ) from exc
    source_layers = extract_stage_layers_data(
        source_state.get("layers"), f"states[{source_index}]", allow_empty=True
    )
    target_layers = extract_stage_layers_data(
        target_state.get("layers"), f"states[{target_index}]", allow_empty=True
    )
    edit_objects = sample.get("edit_object")
    ops = packed_edit_objects_to_ops(edit_objects)
    metadata = sample.get("canonical_metadata")
    if not isinstance(metadata, Mapping):
        raise ExportError("packed pair 缺少 canonical_metadata")
    viewer_metadata = dict(metadata)
    viewer_metadata["edit_schema_version"] = dataset_meta.get("edit_schema_version")
    condition_value = sample.get("condition")
    condition_shape = getattr(condition_value, "shape", None)
    viewer_metadata["condition_point_count"] = (
        int(condition_shape[0])
        if condition_shape is not None and len(condition_shape) >= 1
        else (
            len(condition_value) if isinstance(condition_value, (list, tuple)) else None
        )
    )
    viewer_metadata["normalization_stats"] = dict(normalization)
    canonical_contract = dataset_meta.get("canonical_contract")
    if isinstance(canonical_contract, Mapping):
        viewer_metadata["task_contract_id"] = canonical_contract.get("task_contract_id")
        viewer_metadata["condition_surface_mode"] = canonical_contract.get(
            "condition_surface_mode"
        )
    pair_id = sample.get("pair_name")
    if not isinstance(pair_id, str) or not pair_id:
        raise ExportError("packed pair 缺少 pair_name")
    change_kind = metadata.get("change_kind")
    validation = sample.get("validation")
    return {
        "pair_id": pair_id,
        "dataset_locator": record.locator,
        "source_stage_id": _state_name(source_state, source_index),
        "target_stage_id": _state_name(target_state, target_index),
        "source_state_id": _state_name(source_state, source_index),
        "target_state_id": _state_name(target_state, target_index),
        "edit_sequence_path": f"{record.shard_path}#samples[{record.sample_index}]",
        "base_edit_sequence_path": None,
        "override_edit_sequence_path": None,
        "edit_object_path": f"{record.shard_path}#samples[{record.sample_index}].edit_object",
        "metadata": viewer_metadata,
        "validation": dict(validation) if isinstance(validation, Mapping) else {},
        "source_state_meta": (
            dict(source_state.get("meta", {}))
            if isinstance(source_state.get("meta"), Mapping)
            else {}
        ),
        "target_state_meta": (
            dict(target_state.get("meta", {}))
            if isinstance(target_state.get("meta"), Mapping)
            else {}
        ),
        "buildings": packed_building_summaries(
            source_layers, target_layers, source_state, target_state
        ),
        "include_demolition": (
            canonical_contract.get("task_contract_id") == "bidirectional_monotonic_v1"
            if isinstance(canonical_contract, Mapping)
            else None
        ),
        "is_demolition_pair": change_kind == "demolition",
        "change_kind": change_kind,
        "pair_hash": metadata.get("pair_hash"),
        "layers": build_viewer_layers(source_layers, target_layers, ops, edit_objects),
    }


def export_packed_viewer_data(
    dataset_dir: pathlib.Path,
    pair_id_filter: Optional[str] = None,
    pair_locator: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        meta = load_packed_metadata(dataset_dir)
        states_payload = load_packed_states_payload(dataset_dir)
        states_value = states_payload["states"]
        if not all(isinstance(state, Mapping) for state in states_value):
            raise PackedViewerError("states 中存在非 mapping 项")
        states = states_value
        if pair_locator or pair_id_filter:
            records = [
                find_packed_sample(
                    dataset_dir, pair_name=pair_id_filter, locator=pair_locator
                )
            ]
        else:
            total = sum(packed_sample_count(dataset_dir, split) for split in SPLITS)
            if total > 1000:
                raise ExportError(f"packed 数据包含 {total} 个 pair；请指定 --pair-id")
            records = [
                record
                for split in SPLITS
                for record in iter_packed_samples(dataset_dir, split)
            ]
    except PackedViewerError as exc:
        raise ExportError(str(exc)) from exc
    output: Dict[str, Any] = {
        "schema": "edit_sequence_multiview_animation_v1",
        "sequence_id": dataset_dir.name,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_kind": "area",
        "dataset_format": "packed",
        "normalization": packed_normalization(states_payload),
        "dataset_meta": dict(meta),
        "pairs": [],
    }
    output["pairs"] = [
        _packed_pair(record, states, meta, output["normalization"])
        for record in records
    ]
    if not output["pairs"]:
        raise ExportError("packed 数据中没有可导出的 pair")
    if sum(len(pair["layers"]) for pair in output["pairs"]) == 0:
        raise ExportError("packed pair 没有可显示的 layer")
    return output


def is_area_dataset(dataset_dir: pathlib.Path, dataset_meta: Any) -> bool:
    if (dataset_dir / "states").exists():
        return True
    if isinstance(dataset_meta, dict):
        return str(dataset_meta.get("edit_schema_version", "")).startswith("area_v2")
    return False


def export_viewer_data(
    dataset_dir: pathlib.Path,
    pair_id_filter: Optional[str] = None,
    override_pair_id: Optional[str] = None,
    override_edit_sequence_path: Optional[pathlib.Path] = None,
    pair_locator: Optional[str] = None,
) -> Dict[str, Any]:
    if not dataset_dir.exists():
        raise ExportError(f"dataset-dir does not exist: {dataset_dir}")
    try:
        packed = is_packed_dataset(dataset_dir)
    except PackedViewerError as exc:
        raise ExportError(str(exc)) from exc
    if packed:
        if override_pair_id or override_edit_sequence_path:
            raise ExportError("packed 数据不支持替换外部 edit sequence YAML")
        return export_packed_viewer_data(dataset_dir, pair_id_filter, pair_locator)
    if pair_locator:
        raise ExportError("raw 数据不接受 packed pair locator")
    edit_seq_dir = dataset_dir / "edit_sequences_v2"
    if not edit_seq_dir.exists():
        raise ExportError(
            f"edit_sequences_v2 not found under dataset-dir: {edit_seq_dir}"
        )
    if pair_id_filter:
        pair_path = edit_seq_dir / f"{pair_id_filter}.yaml"
        if not pair_path.exists():
            raise ExportError(
                f"pair YAML not found for --pair-id {pair_id_filter} in {edit_seq_dir}"
            )
        seq_files = [pair_path]
    else:
        seq_files = sorted(edit_seq_dir.glob("*.yaml"), key=natural_key)
    if not seq_files:
        raise ExportError(f"no pair YAML files found in {edit_seq_dir}")

    dataset_meta_path = dataset_dir / "dataset_meta.yaml"
    dataset_meta = read_yaml(dataset_meta_path) if dataset_meta_path.exists() else {}
    normalization: Dict[str, Any] = {}
    if isinstance(dataset_meta, dict):
        normalization = (
            dataset_meta.get("normalization", {})
            or dataset_meta.get("normalization_stats", {})
            or {}
        )

    output: Dict[str, Any] = {
        "schema": "edit_sequence_multiview_animation_v1",
        "sequence_id": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "dataset_kind": (
            "area" if is_area_dataset(dataset_dir, dataset_meta) else "building"
        ),
        "dataset_format": "raw",
        "normalization": normalization,
        "dataset_meta": dataset_meta or {},
        "pairs": [],
    }
    stage_cache: Dict[str, List[Dict[str, Any]]] = {}
    state_meta_cache: Dict[str, Dict[str, Any]] = {}

    def cached_stage_layers(stage_id: str) -> List[Dict[str, Any]]:
        if stage_id not in stage_cache:
            stage_cache[stage_id] = extract_stage_layers(
                find_stage_file(dataset_dir, stage_id)
            )
        return stage_cache[stage_id]

    def cached_state_layers(state_id: str) -> List[Dict[str, Any]]:
        if state_id not in stage_cache:
            stage_cache[state_id] = extract_stage_layers(
                find_state_file(dataset_dir, state_id), allow_empty=True
            )
        return stage_cache[state_id]

    def cached_state_meta(state_id: str) -> Dict[str, Any]:
        if state_id not in state_meta_cache:
            state_meta_cache[state_id] = load_state_meta(dataset_dir, state_id)
        return state_meta_cache[state_id]

    for seq_file in seq_files:
        pair_id = seq_file.stem
        edit_object_path, edit_objects = load_optional(
            dataset_dir, "edit_objects", pair_id
        )
        _, pair_meta = load_optional(dataset_dir, "pair_meta", pair_id)
        _, validation = load_optional(dataset_dir, "validation_reports", pair_id)

        if output["dataset_kind"] == "area":
            source_stage_id, target_stage_id = resolve_pair_states(pair_id, pair_meta)
            source_state_meta = cached_state_meta(source_stage_id)
            target_state_meta = cached_state_meta(target_stage_id)
            source_layers = cached_state_layers(source_stage_id)
            target_layers = cached_state_layers(target_stage_id)
        else:
            source_stage_id, target_stage_id = split_pair_id(pair_id)
            source_state_meta = {}
            target_state_meta = {}
            source_layers = cached_stage_layers(source_stage_id)
            target_layers = cached_stage_layers(target_stage_id)
        effective_seq_file = (
            override_edit_sequence_path
            if override_pair_id == pair_id and override_edit_sequence_path
            else seq_file
        )
        if effective_seq_file and not effective_seq_file.exists():
            raise ExportError(
                f"override edit sequence YAML does not exist: {effective_seq_file}"
            )
        ops_raw = read_yaml(effective_seq_file)
        if not isinstance(ops_raw, list):
            raise ExportError(
                f"edit sequence must be a YAML list: {effective_seq_file}"
            )
        ops = [op for op in ops_raw if isinstance(op, dict)]

        pair = {
            "pair_id": pair_id,
            "source_stage_id": source_stage_id,
            "target_stage_id": target_stage_id,
            "source_state_id": (
                source_stage_id if output["dataset_kind"] == "area" else None
            ),
            "target_state_id": (
                target_stage_id if output["dataset_kind"] == "area" else None
            ),
            "edit_sequence_path": str(effective_seq_file),
            "base_edit_sequence_path": str(seq_file),
            "override_edit_sequence_path": (
                str(override_edit_sequence_path)
                if override_pair_id == pair_id and override_edit_sequence_path
                else None
            ),
            "edit_object_path": str(edit_object_path) if edit_object_path else None,
            "metadata": pair_meta or {},
            "validation": validation or {},
            "source_state_meta": source_state_meta,
            "target_state_meta": target_state_meta,
            "buildings": building_summaries(source_state_meta, target_state_meta),
            "include_demolition": (
                pair_meta.get("include_demolition")
                if isinstance(pair_meta, dict)
                else None
            ),
            "is_demolition_pair": (
                pair_meta.get("is_demolition_pair")
                if isinstance(pair_meta, dict)
                else None
            ),
            "change_kind": (
                "demolition"
                if isinstance(pair_meta, dict) and pair_meta.get("is_demolition_pair")
                else None
            ),
            "pair_hash": (
                pair_meta.get("pair_hash") if isinstance(pair_meta, dict) else None
            ),
            "layers": build_viewer_layers(
                source_layers, target_layers, ops, edit_objects
            ),
        }
        output["pairs"].append(pair)

    if not output["pairs"]:
        raise ExportError("no pairs were exported")
    total_layers = sum(len(pair["layers"]) for pair in output["pairs"])
    if total_layers == 0:
        raise ExportError("stage geometry parsed but no layers were exported")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export construction edit animation viewer JSON."
    )
    parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--pair-id",
        type=str,
        default=None,
        help="Export only one pair. Useful for large area-level datasets.",
    )
    parser.add_argument(
        "--pair-locator",
        type=str,
        default=None,
        help="Packed shard locator returned by the dataset browser API.",
    )
    parser.add_argument(
        "--override-pair-id",
        type=str,
        default=None,
        help="Pair id whose edit sequence should be replaced by --override-edit-sequence-yaml.",
    )
    parser.add_argument(
        "--override-edit-sequence-yaml",
        type=pathlib.Path,
        default=None,
        help="External edit sequence YAML to use for --override-pair-id.",
    )
    args = parser.parse_args()

    try:
        if bool(args.override_pair_id) != bool(args.override_edit_sequence_yaml):
            raise ExportError(
                "--override-pair-id and --override-edit-sequence-yaml must be provided together"
            )
        data = export_viewer_data(
            args.dataset_dir,
            args.pair_id,
            args.override_pair_id,
            args.override_edit_sequence_yaml,
            args.pair_locator,
        )
    except ExportError as exc:
        raise SystemExit(f"error: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    pair_count = len(data["pairs"])
    layer_count = sum(len(pair["layers"]) for pair in data["pairs"])
    print(f"Exported pairs: {pair_count}")
    print(f"Exported layers: {layer_count}")
    print(f"Output path: {args.output.resolve()}")


if __name__ == "__main__":
    main()

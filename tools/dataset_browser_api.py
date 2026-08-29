import argparse
import json
import pathlib
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from gendiff_data_process.viewer_packed import (
    SPLITS,
    PackedSample,
    PackedViewerError,
    find_packed_sample,
    is_packed_dataset,
    iter_packed_samples,
    load_packed_metadata,
    load_packed_states,
    load_packed_states_payload,
    packed_condition_points,
    packed_sample_count,
)

PACKED_QUERY_SCAN_LIMIT = 5000


def read_yaml(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def natural_key(text: str) -> List[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def pair_id_from_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return pathlib.Path(value).stem


def pair_id_from_combo(combo: Dict[str, Any]) -> Optional[str]:
    for key in ("pair_name", "pair_id"):
        value = combo.get(key)
        if isinstance(value, str) and value:
            return value
    for key in (
        "v2_edit_sequence",
        "edit_sequence",
        "edit_object",
        "condition",
        "v2_preview_obj",
    ):
        pair_id = pair_id_from_path(combo.get(key))
        if pair_id:
            if pair_id.endswith("_r0"):
                pair_id = pair_id[:-3]
            if pair_id.endswith("_from_v2"):
                pair_id = pair_id[:-8]
            return pair_id
    return None


def state_id_from_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    return pathlib.Path(value).name


def compact_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    keep = (
        "pair_name",
        "source_state",
        "target_state",
        "source_state_index",
        "target_state_index",
        "source_state_tuple",
        "target_state_tuple",
        "include_demolition",
        "is_demolition_pair",
        "edit_schema_version",
        "condition",
        "condition_ply",
        "condition_point_count",
        "reconstructed_layer_count_match",
        "reconstructed_point_count_match",
        "max_coord_error",
        "max_height_error",
        "max_ar_tokens_required",
    )
    return {key: meta.get(key) for key in keep if key in meta}


def load_pair_meta(dataset_dir: pathlib.Path, pair_id: str) -> Dict[str, Any]:
    path = dataset_dir / "pair_meta" / f"{pair_id}.yaml"
    if not path.exists():
        return {}
    return compact_meta(read_yaml(path))


def list_edit_sequence_pairs(dataset_dir: pathlib.Path) -> List[str]:
    edit_seq_dir = dataset_dir / "edit_sequences_v2"
    if not edit_seq_dir.exists():
        return []
    return sorted((path.stem for path in edit_seq_dir.glob("*.y*ml")), key=natural_key)


def load_split_pairs(dataset_dir: pathlib.Path, split: str) -> List[Dict[str, Any]]:
    split_path = dataset_dir / f"{split}.yaml"
    if not split_path.exists():
        return []
    data = read_yaml(split_path)
    if not isinstance(data, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pair_id = pair_id_from_combo(item)
        if not pair_id:
            continue
        rows.append(
            {
                "pair_id": pair_id,
                "source_state": state_id_from_path(item.get("t1")),
                "target_state": state_id_from_path(item.get("t2")),
                "condition_path": (
                    item.get("condition")
                    if isinstance(item.get("condition"), str)
                    else None
                ),
                "edit_sequence_path": (
                    item.get("v2_edit_sequence")
                    if isinstance(item.get("v2_edit_sequence"), str)
                    else None
                ),
            }
        )
    return rows


def split_counts(dataset_dir: pathlib.Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.yaml"
        if not split_path.exists():
            continue
        count = 0
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("- "):
                    count += 1
        counts[split] = count
    return counts


def compact_dataset_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    keep = (
        "schema_version",
        "area_dir",
        "building_count",
        "state_count",
        "stage_count",
        "legal_pair_count",
        "sample_count",
        "pair_count",
        "include_demolition",
        "condition_point_count",
        "canonical_supervision",
        "edit_schema_version",
        "coordinate_normalized",
        "normalization_scope",
        "point_value_semantics",
        "anchor_supervision",
        "fallback_pair_count",
        "validation_failure_count",
        "split_sample_counts",
    )
    compact = {key: meta.get(key) for key in keep if key in meta}
    if isinstance(meta.get("normalization_stats"), dict):
        compact["normalization_stats"] = meta["normalization_stats"]
    if isinstance(meta.get("canonical_contract"), Mapping):
        compact["canonical_contract"] = dict(meta["canonical_contract"])
    buildings = meta.get("buildings")
    if isinstance(buildings, list):
        compact["buildings"] = [
            {
                "building_id": item.get("building_id"),
                "building_name": item.get("building_name"),
                "stage_count": item.get("stage_count"),
            }
            for item in buildings
            if isinstance(item, dict)
        ]
    return compact


def packed_normalization(payload: Mapping[str, Any]) -> Dict[str, float]:
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


def packed_dataset_summary(dataset_dir: pathlib.Path) -> Dict[str, Any]:
    meta = load_packed_metadata(dataset_dir)
    states_payload = load_packed_states_payload(dataset_dir)
    states = states_payload["states"]
    splits = {split: packed_sample_count(dataset_dir, split) for split in SPLITS}
    return {
        "datasetDir": str(dataset_dir),
        "datasetName": dataset_dir.name,
        "datasetKind": "area",
        "datasetFormat": "packed",
        "splits": splits,
        "pairTotal": sum(splits.values()),
        "stateTotal": len(states),
        "stageTotal": 0,
        "hasConditions": sum(splits.values()) > 0,
        "datasetMeta": compact_dataset_meta(meta),
        "normalization": packed_normalization(states_payload),
    }


def dataset_summary(dataset_dir: pathlib.Path) -> Dict[str, Any]:
    if is_packed_dataset(dataset_dir):
        return packed_dataset_summary(dataset_dir)
    meta_path = dataset_dir / "dataset_meta.yaml"
    dataset_meta = read_yaml(meta_path) if meta_path.exists() else {}
    splits = split_counts(dataset_dir)
    pair_total = len(list_edit_sequence_pairs(dataset_dir))
    state_total = (
        len([path for path in (dataset_dir / "states").glob("*") if path.is_dir()])
        if (dataset_dir / "states").exists()
        else 0
    )
    stage_total = (
        len([path for path in (dataset_dir / "stages").glob("*") if path.is_dir()])
        if (dataset_dir / "stages").exists()
        else 0
    )
    return {
        "datasetDir": str(dataset_dir),
        "datasetName": dataset_dir.name,
        "datasetKind": "area" if (dataset_dir / "states").exists() else "building",
        "datasetFormat": "raw",
        "splits": splits,
        "pairTotal": pair_total,
        "stateTotal": state_total,
        "stageTotal": stage_total,
        "hasConditions": (dataset_dir / "conditions").exists(),
        "datasetMeta": compact_dataset_meta(dataset_meta),
        "normalization": (
            (
                dataset_meta.get("normalization", {})
                or dataset_meta.get("normalization_stats", {})
            )
            if isinstance(dataset_meta, dict)
            else {}
        ),
    }


def base_pair_rows(dataset_dir: pathlib.Path, split: str) -> List[Dict[str, Any]]:
    if split != "all":
        rows = load_split_pairs(dataset_dir, split)
        if rows:
            return rows
    by_id: Dict[str, Dict[str, Any]] = {}
    split_names: Iterable[str] = SPLITS if split == "all" else (split,)
    for split_name in split_names:
        for row in load_split_pairs(dataset_dir, split_name):
            by_id.setdefault(row["pair_id"], row)
    if by_id:
        return sorted(by_id.values(), key=lambda row: natural_key(row["pair_id"]))
    return [{"pair_id": pair_id} for pair_id in list_edit_sequence_pairs(dataset_dir)]


def _packed_state_name(states: Sequence[Mapping[str, Any]], value: Any) -> str:
    try:
        index = int(value)
        state = states[index]
    except (TypeError, ValueError, IndexError) as exc:
        raise PackedViewerError(f"packed state index 无效: {value}") from exc
    name = state.get("state_name")
    return name if isinstance(name, str) and name else f"state_{index:06d}"


def _condition_point_count(value: Any) -> Optional[int]:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _packed_pair_item(
    record: PackedSample, states: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    sample = record.sample
    metadata = sample.get("canonical_metadata")
    if not isinstance(metadata, Mapping):
        raise PackedViewerError("packed sample 缺少 canonical_metadata")
    validation = sample.get("validation")
    validation_ok = (
        validation.get("canonical_roundtrip")
        if isinstance(validation, Mapping)
        else None
    )
    pair_id = sample.get("pair_name")
    if not isinstance(pair_id, str) or not pair_id:
        raise PackedViewerError("packed sample 缺少 pair_name")
    change_kind = metadata.get("change_kind")
    return {
        "pairId": pair_id,
        "pairLocator": record.locator,
        "sourceState": _packed_state_name(states, sample.get("source_state_index")),
        "targetState": _packed_state_name(states, sample.get("target_state_index")),
        "changeKind": change_kind,
        "pairHash": metadata.get("pair_hash"),
        "isDemolitionPair": change_kind == "demolition",
        "includeDemolition": True,
        "validationOk": validation_ok,
        "conditionPointCount": _condition_point_count(sample.get("condition")),
    }


def _packed_records_page(
    dataset_dir: pathlib.Path,
    split_names: Sequence[str],
    offset: int,
    limit: int,
) -> List[PackedSample]:
    output: List[PackedSample] = []
    remaining_offset = offset
    for split_name in split_names:
        split_count = packed_sample_count(dataset_dir, split_name)
        if remaining_offset >= split_count:
            remaining_offset -= split_count
            continue
        take = min(limit - len(output), split_count - remaining_offset)
        output.extend(
            iter_packed_samples(
                dataset_dir,
                split_name,
                start=remaining_offset,
                stop=remaining_offset + take,
            )
        )
        remaining_offset = 0
        if len(output) >= limit:
            break
    return output


def packed_pair_list(
    dataset_dir: pathlib.Path, split: str, query: str, offset: int, limit: int
) -> Dict[str, Any]:
    if split != "all" and split not in SPLITS:
        raise PackedViewerError(f"不支持的 split: {split}")
    split_names: Sequence[str] = SPLITS if split == "all" else (split,)
    offset = max(0, offset)
    limit = min(max(1, limit), 500)
    states = load_packed_states(dataset_dir)
    normalized_query = query.strip().lower()
    search_truncated = False
    scanned = 0
    if normalized_query:
        matches: List[PackedSample] = []
        candidate_total = sum(
            packed_sample_count(dataset_dir, name) for name in split_names
        )
        for split_name in split_names:
            for record in iter_packed_samples(dataset_dir, split_name):
                scanned += 1
                item = _packed_pair_item(record, states)
                haystack = " ".join(
                    str(item.get(key, ""))
                    for key in (
                        "pairId",
                        "sourceState",
                        "targetState",
                        "changeKind",
                        "pairHash",
                    )
                ).lower()
                if normalized_query in haystack:
                    matches.append(record)
                if scanned >= PACKED_QUERY_SCAN_LIMIT:
                    search_truncated = candidate_total > scanned
                    break
            if scanned >= PACKED_QUERY_SCAN_LIMIT:
                break
        total = len(matches)
        records = matches[offset : offset + limit]
    else:
        total = sum(packed_sample_count(dataset_dir, name) for name in split_names)
        records = _packed_records_page(dataset_dir, split_names, offset, limit)
    return {
        "datasetDir": str(dataset_dir),
        "datasetFormat": "packed",
        "split": split,
        "query": query,
        "offset": offset,
        "limit": limit,
        "total": total,
        "searchScanned": scanned,
        "searchTruncated": search_truncated,
        "pairs": [_packed_pair_item(record, states) for record in records],
    }


def pair_list(
    dataset_dir: pathlib.Path, split: str, query: str, offset: int, limit: int
) -> Dict[str, Any]:
    if is_packed_dataset(dataset_dir):
        return packed_pair_list(dataset_dir, split, query, offset, limit)
    rows = base_pair_rows(dataset_dir, split)
    normalized_query = query.strip().lower()
    if normalized_query:
        rows = [
            row
            for row in rows
            if normalized_query
            in " ".join(
                str(row.get(key, ""))
                for key in ("pair_id", "source_state", "target_state")
            ).lower()
        ]
    total = len(rows)
    offset = max(0, offset)
    limit = min(max(1, limit), 500)
    page = rows[offset : offset + limit]
    enriched = []
    for row in page:
        pair_id = str(row["pair_id"])
        meta = load_pair_meta(dataset_dir, pair_id)
        validation_ok = None
        if meta:
            layer_ok = meta.get("reconstructed_layer_count_match")
            point_ok = meta.get("reconstructed_point_count_match")
            validation_ok = bool(layer_ok and point_ok)
        enriched.append(
            {
                "pairId": pair_id,
                "sourceState": row.get("source_state") or meta.get("source_state"),
                "targetState": row.get("target_state") or meta.get("target_state"),
                "isDemolitionPair": meta.get("is_demolition_pair"),
                "includeDemolition": meta.get("include_demolition"),
                "validationOk": validation_ok,
                "conditionPointCount": meta.get("condition_point_count"),
            }
        )
    return {
        "datasetDir": str(dataset_dir),
        "datasetFormat": "raw",
        "split": split,
        "query": query,
        "offset": offset,
        "limit": limit,
        "total": total,
        "pairs": enriched,
    }


def find_condition_path(
    dataset_dir: pathlib.Path, pair_id: str
) -> Optional[pathlib.Path]:
    meta = load_pair_meta(dataset_dir, pair_id)
    for key in ("condition", "condition_ply"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            path = pathlib.Path(value)
            if path.exists():
                return path
    condition_dir = dataset_dir / "conditions"
    for suffix in (".pt", ".ply"):
        for candidate in (
            condition_dir / f"{pair_id}_r0{suffix}",
            condition_dir / f"{pair_id}{suffix}",
        ):
            if candidate.exists():
                return candidate
    matches = (
        sorted(
            condition_dir.glob(f"{pair_id}*"), key=lambda path: natural_key(path.name)
        )
        if condition_dir.exists()
        else []
    )
    return matches[0] if matches else None


def load_pt_points(path: pathlib.Path) -> List[List[float]]:
    import torch

    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict):
        for key in ("points", "condition", "change_point_clouds", "xyz"):
            if key in tensor:
                tensor = tensor[key]
                break
    tensor = tensor.float().reshape(-1, tensor.shape[-1])
    if tensor.shape[-1] < 3:
        raise RuntimeError(f"condition tensor must have at least 3 columns: {path}")
    return tensor[:, :3].tolist()


def load_ascii_ply_points(path: pathlib.Path) -> List[List[float]]:
    points: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        in_header = True
        vertex_count: Optional[int] = None
        for line in handle:
            stripped = line.strip()
            if in_header:
                if stripped.startswith("element vertex"):
                    vertex_count = int(stripped.split()[-1])
                if stripped == "end_header":
                    in_header = False
                continue
            if vertex_count is not None and len(points) >= vertex_count:
                break
            parts = stripped.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return points


def sample_points(
    points: Sequence[Sequence[float]], max_points: int
) -> List[List[float]]:
    if max_points <= 0 or len(points) <= max_points:
        return [[float(point[0]), float(point[1]), float(point[2])] for point in points]
    if max_points == 1:
        return [[float(points[0][0]), float(points[0][1]), float(points[0][2])]]
    step = (len(points) - 1) / (max_points - 1)
    return [
        [
            float(points[round(i * step)][0]),
            float(points[round(i * step)][1]),
            float(points[round(i * step)][2]),
        ]
        for i in range(max_points)
    ]


def packed_condition(
    dataset_dir: pathlib.Path,
    pair_id: str,
    max_points: int,
    pair_locator: Optional[str],
) -> Dict[str, Any]:
    record = find_packed_sample(dataset_dir, pair_name=pair_id, locator=pair_locator)
    points = packed_condition_points(record.sample.get("condition"))
    sampled = sample_points(points, max_points)
    return {
        "pairId": pair_id,
        "pairLocator": record.locator,
        "available": True,
        "path": f"{record.shard_path}#samples[{record.sample_index}].condition",
        "totalPoints": len(points),
        "sampledPoints": len(sampled),
        "stride": 3,
        "points": [coord for point in sampled for coord in point],
    }


def condition(
    dataset_dir: pathlib.Path,
    pair_id: str,
    max_points: int,
    pair_locator: Optional[str] = None,
) -> Dict[str, Any]:
    if is_packed_dataset(dataset_dir):
        return packed_condition(dataset_dir, pair_id, max_points, pair_locator)
    if pair_locator:
        raise RuntimeError("raw 数据不接受 packed pair locator")
    path = find_condition_path(dataset_dir, pair_id)
    if not path:
        return {
            "pairId": pair_id,
            "available": False,
            "path": None,
            "totalPoints": 0,
            "sampledPoints": 0,
            "stride": 3,
            "points": [],
        }
    points = (
        load_pt_points(path)
        if path.suffix.lower() == ".pt"
        else load_ascii_ply_points(path)
    )
    sampled = sample_points(points, max_points)
    flat = [coord for point in sampled for coord in point]
    return {
        "pairId": pair_id,
        "available": True,
        "path": str(path),
        "totalPoints": len(points),
        "sampledPoints": len(sampled),
        "stride": 3,
        "points": flat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dataset browser API helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)

    pairs_parser = subparsers.add_parser("pairs")
    pairs_parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    pairs_parser.add_argument("--split", default="all")
    pairs_parser.add_argument("--query", default="")
    pairs_parser.add_argument("--offset", type=int, default=0)
    pairs_parser.add_argument("--limit", type=int, default=100)

    condition_parser = subparsers.add_parser("condition")
    condition_parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    condition_parser.add_argument("--pair-id", required=True)
    condition_parser.add_argument("--pair-locator", default=None)
    condition_parser.add_argument("--max-points", type=int, default=8192)

    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"dataset-dir does not exist: {dataset_dir}")

    if args.command == "summary":
        payload = dataset_summary(dataset_dir)
    elif args.command == "pairs":
        payload = pair_list(
            dataset_dir, args.split, args.query, args.offset, args.limit
        )
    elif args.command == "condition":
        payload = condition(
            dataset_dir, args.pair_id, args.max_points, args.pair_locator
        )
    else:
        raise SystemExit(f"unsupported command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

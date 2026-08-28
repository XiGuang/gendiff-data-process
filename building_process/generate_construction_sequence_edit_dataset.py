from __future__ import annotations

import argparse
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import yaml
from shapely.geometry import MultiPolygon, Polygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_process import generate_construction_proxy as proxy_stage

POLICIES = ("vertical", "footprint", "hybrid")


def _import_dataset_stage():
    try:
        from building_process import build_layer_edit_dataset_from_sequence as dataset_stage
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import build_layer_edit_dataset_from_sequence.py dependencies. "
            "This script requires the same runtime stack as the dataset builder, including torch."
        ) from exc
    return dataset_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate monotonic construction-stage sequences from flat proxy YAML files and "
            "build matching layer-edit training data in one pass."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input flat proxy YAML file or a directory containing flat proxy YAML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for generated sequence subdirectories.",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=None,
        help=(
            "Optional dataset output root. Defaults to <sequence_dir>/layer_edit_dataset for each sequence."
        ),
    )
    parser.add_argument(
        "--stage-count",
        type=int,
        default=5,
        help="Number of stages in each sequence. The last stage is always complete.",
    )
    parser.add_argument(
        "--policy",
        choices=("auto",) + POLICIES,
        default="auto",
        help="Sequence policy. auto randomly selects one of vertical, footprint, hybrid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--min-progress",
        type=float,
        default=0.15,
        help="Minimum progress ratio for the first sampled stage, in (0, 1].",
    )
    parser.add_argument(
        "--axis",
        choices=("random", "x", "z"),
        default="random",
        help="Footprint clipping axis. random picks x or z per source YAML.",
    )
    parser.add_argument(
        "--keep-side",
        choices=("random", "min", "max"),
        default="random",
        help="Which side of the bounds to keep for footprint clipping.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="Discard clipped polygon fragments smaller than this XZ area.",
    )
    parser.add_argument(
        "--export-obj",
        action="store_true",
        help="Also export a reconstructed OBJ for each generated stage.",
    )
    parser.add_argument(
        "--pair-mode",
        type=str,
        choices=("consecutive", "all_forward"),
        default="consecutive",
        help="Use only consecutive forward transitions or all i<j forward transitions.",
    )
    parser.add_argument(
        "--condition-point-count",
        type=int,
        default=8192,
        help="Number of surface points to sample for each condition point cloud.",
    )
    parser.add_argument(
        "--change-tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for deciding whether a target layer differs from the history layer.",
    )
    parser.add_argument(
        "--copy-objs",
        action="store_true",
        help="Also copy stage OBJ files into the normalized stage directories.",
    )
    parser.add_argument(
        "--save-condition-ply",
        action="store_true",
        help="Also export each condition point cloud as a .ply file next to the .pt file.",
    )
    return parser.parse_args()


def _sequence_seed(base_seed: int, yaml_path: Path) -> int:
    path_hash = zlib.crc32(str(yaml_path).encode("utf-8")) & 0xFFFFFFFF
    return (base_seed + path_hash) % (2**32)


def _choose_policy(policy: str, rng: random.Random) -> str:
    if policy != "auto":
        return policy
    return rng.choice(POLICIES)


def _choose_axis(axis: str, rng: random.Random) -> str:
    if axis != "random":
        return axis
    return rng.choice(("x", "z"))


def _choose_keep_side(keep_side: str, rng: random.Random) -> str:
    if keep_side != "random":
        return keep_side
    return rng.choice(("min", "max"))


def _build_progression(stage_count: int, min_progress: float, rng: random.Random) -> list[float]:
    if stage_count < 1:
        raise ValueError(f"stage-count must be >= 1, got {stage_count}")

    min_progress = proxy_stage._validate_ratio(min_progress)
    if stage_count == 1:
        return [1.0]

    sampled = [rng.uniform(min_progress, 0.98) for _ in range(stage_count - 1)]
    sampled.sort()
    progression = [float(value) for value in sampled]
    progression[-1] = max(progression[-1], min_progress)
    progression.append(1.0)
    return progression


def _stage_dir_name(index: int, total: int, policy: str) -> str:
    width = max(2, len(str(total)))
    return f"stage_{index:0{width}d}_{policy}"


def _sequence_dir_name(policy: str, seed: int) -> str:
    return f"sequence_{policy}_seed{seed}"


def _generate_hybrid_stage(
    entries: list[dict],
    vertical_ratio: float,
    footprint_ratio: float,
    axis: str,
    keep_side: str,
    min_area: float,
) -> tuple[list[dict], dict]:
    global_min, global_max = proxy_stage._global_height_bounds(entries)
    target_height = global_min + vertical_ratio * (global_max - global_min)
    bounds = proxy_stage._global_planar_bounds(entries)
    clip_region = proxy_stage._build_clip_box(bounds, axis, keep_side, footprint_ratio)

    staged: list[dict] = []
    for entry in entries:
        entry_min = float(entry["min_height"])
        entry_max = float(entry["max_height"])
        if entry_min >= target_height - proxy_stage.EPS:
            continue

        clipped_max = min(entry_max, target_height)
        if clipped_max - entry_min <= proxy_stage.EPS:
            continue

        clipped_geometry = proxy_stage._entry_polygon(entry).intersection(clip_region)
        for polygon in proxy_stage._iter_polygons(clipped_geometry):
            contour = proxy_stage._normalize_polygon(polygon, min_area)
            if not contour:
                continue
            staged.append(
                {
                    "source_proxy_id": int(entry["proxy_id"]),
                    "level_index": int(entry["level_index"]),
                    "min_height": entry_min,
                    "max_height": float(clipped_max),
                    "footprint": contour,
                }
            )

    min_x, max_x, min_z, max_z = bounds
    metadata = {
        "mode": "hybrid",
        "vertical_ratio": float(vertical_ratio),
        "footprint_ratio": float(footprint_ratio),
        "axis": axis,
        "keep_side": keep_side,
        "global_min_height": global_min,
        "global_max_height": global_max,
        "target_height": float(target_height),
        "global_bounds": {
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
        },
        "clip_bounds": {
            "min_x": float(clip_region.bounds[0]),
            "min_z": float(clip_region.bounds[1]),
            "max_x": float(clip_region.bounds[2]),
            "max_z": float(clip_region.bounds[3]),
        },
        "min_area": float(min_area),
    }
    return staged, metadata


def _dataset_dir_for_sequence(
    dataset_output_root: Path | None,
    sequence_dir: Path,
    yaml_path: Path,
) -> Path:
    if dataset_output_root is None:
        return sequence_dir / "layer_edit_dataset"
    return dataset_output_root / yaml_path.stem / sequence_dir.name


def _safe_polygon(entry: dict) -> Polygon | None:
    footprint = entry.get("footprint") or []
    if len(footprint) < 3:
        return None
    polygon = Polygon([(float(point[0]), float(point[1])) for point in footprint])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return None
    return polygon


def _footprint_signature(entry: dict) -> tuple[float, float, float, float, float]:
    polygon = _safe_polygon(entry)
    if polygon is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    min_x, min_z, max_x, max_z = polygon.bounds
    return (
        round(float(min_x), 6),
        round(float(min_z), 6),
        round(float(max_x), 6),
        round(float(max_z), 6),
        round(float(polygon.area), 6),
    )


def _ordered_group(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda entry: (
            int(entry["level_index"]),
            float(entry["min_height"]),
            float(entry["max_height"]),
            int(entry["source_proxy_id"]),
            *_footprint_signature(entry),
        ),
    )


def _lineage_match_score(previous: dict, current: dict) -> float:
    prev_polygon = _safe_polygon(previous)
    curr_polygon = _safe_polygon(current)
    if prev_polygon is None or curr_polygon is None:
        return 0.0

    intersection_area = float(prev_polygon.intersection(curr_polygon).area)
    if intersection_area <= proxy_stage.EPS:
        return 0.0

    prev_area = max(float(prev_polygon.area), proxy_stage.EPS)
    curr_area = max(float(curr_polygon.area), proxy_stage.EPS)
    area_score = intersection_area / min(prev_area, curr_area)

    prev_min = float(previous["min_height"])
    prev_max = float(previous["max_height"])
    curr_min = float(current["min_height"])
    curr_max = float(current["max_height"])
    overlap = max(0.0, min(prev_max, curr_max) - max(prev_min, curr_min))
    height_scale = max(prev_max - prev_min, curr_max - curr_min, proxy_stage.EPS)
    height_score = overlap / height_scale
    return 4.0 * area_score + height_score


def _match_lineages(previous_entries: list[dict], current_entries: list[dict]) -> dict[int, int]:
    if not previous_entries or not current_entries:
        return {}

    candidates: list[tuple[float, int, int]] = []
    for prev_index, previous in enumerate(previous_entries):
        for curr_index, current in enumerate(current_entries):
            score = _lineage_match_score(previous, current)
            if score <= 0.0:
                continue
            candidates.append((score, prev_index, curr_index))

    candidates.sort(
        key=lambda item: (
            -item[0],
            int(previous_entries[item[1]]["proxy_id"]),
            int(current_entries[item[2]].get("proxy_id", item[2])),
        )
    )

    assigned_previous: set[int] = set()
    assigned_current: dict[int, int] = {}
    for _, prev_index, curr_index in candidates:
        if prev_index in assigned_previous or curr_index in assigned_current:
            continue
        assigned_previous.add(prev_index)
        assigned_current[curr_index] = int(previous_entries[prev_index]["proxy_id"])
    return assigned_current


def _stabilize_stage_entries(
    stage_entries: list[dict],
    previous_entries: list[dict],
    used_proxy_ids: set[int],
    next_proxy_id: int,
) -> tuple[list[dict], int]:
    previous_by_source: dict[int, list[dict]] = defaultdict(list)
    for entry in previous_entries:
        previous_by_source[int(entry["source_proxy_id"])].append(dict(entry))

    current_by_source: dict[int, list[dict]] = defaultdict(list)
    for entry in stage_entries:
        source_proxy_id = int(entry.get("source_proxy_id", entry.get("proxy_id", 0)))
        current_by_source[source_proxy_id].append(
            {
                "source_proxy_id": source_proxy_id,
                "level_index": int(entry["level_index"]),
                "min_height": float(entry["min_height"]),
                "max_height": float(entry["max_height"]),
                "footprint": entry["footprint"],
            }
        )

    stabilized: list[dict] = []
    for source_proxy_id in sorted(current_by_source):
        previous_group = _ordered_group(previous_by_source.get(source_proxy_id, []))
        current_group = _ordered_group(current_by_source[source_proxy_id])
        matched_proxy_ids = _match_lineages(previous_group, current_group)

        for curr_index, current_entry in enumerate(current_group):
            proxy_id = matched_proxy_ids.get(curr_index)
            if proxy_id is None:
                if source_proxy_id not in used_proxy_ids:
                    proxy_id = source_proxy_id
                    used_proxy_ids.add(proxy_id)
                    next_proxy_id = max(next_proxy_id, proxy_id + 1)
                else:
                    while next_proxy_id in used_proxy_ids:
                        next_proxy_id += 1
                    proxy_id = next_proxy_id
                    used_proxy_ids.add(proxy_id)
                    next_proxy_id += 1

            stabilized.append(
                {
                    "proxy_id": int(proxy_id),
                    "source_proxy_id": int(source_proxy_id),
                    "level_index": int(current_entry["level_index"]),
                    "min_height": float(current_entry["min_height"]),
                    "max_height": float(current_entry["max_height"]),
                    "footprint": current_entry["footprint"],
                }
            )

    stabilized.sort(
        key=lambda entry: (
            int(entry["level_index"]),
            float(entry["min_height"]),
            float(entry["max_height"]),
            int(entry["source_proxy_id"]),
            int(entry["proxy_id"]),
            *_footprint_signature(entry),
        )
    )
    return stabilized, next_proxy_id


def _write_stage_outputs(
    entries: list[dict],
    source_yaml: Path,
    output_dir: Path,
    metadata: dict,
    export_obj: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "building1.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(entries, handle, sort_keys=False)

    meta_payload = {
        "source_yaml": str(source_yaml),
        "source_entry_count": int(metadata["source_entry_count"]),
        "stage_entry_count": len(entries),
        **metadata,
    }
    with (output_dir / "construction_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta_payload, handle, sort_keys=False)

    if export_obj and entries:
        proxy_stage._export_stage_obj(entries, output_dir / "building1.obj")


def _write_sequence_meta(
    sequence_dir: Path,
    *,
    source_yaml: Path,
    sequence_seed: int,
    policy: str,
    axis: str | None,
    keep_side: str | None,
    stage_records: list[dict],
) -> None:
    payload = {
        "source_yaml": str(source_yaml),
        "sequence_seed": int(sequence_seed),
        "policy": policy,
        "axis": axis,
        "keep_side": keep_side,
        "stage_count": len(stage_records),
        "stable_proxy_ids": True,
        "stages": stage_records,
    }
    with (sequence_dir / "sequence_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _generate_stage_entries(
    *,
    policy: str,
    stage_index: int,
    entries: list[dict],
    vertical_progress: list[float] | None,
    footprint_progress: list[float] | None,
    axis: str | None,
    keep_side: str | None,
    min_area: float,
    stage_count: int,
) -> tuple[list[dict], dict, dict]:
    if policy == "vertical":
        ratio = float(vertical_progress[stage_index])
        stage_entries, metadata = proxy_stage._generate_vertical(entries, ratio)
        stage_record = {
            "stage_index": stage_index + 1,
            "stage_name": _stage_dir_name(stage_index + 1, stage_count, policy),
            "vertical_ratio": ratio,
        }
        return stage_entries, metadata, stage_record

    if policy == "footprint":
        ratio = float(footprint_progress[stage_index])
        stage_entries, metadata = proxy_stage._generate_footprint(
            entries,
            ratio,
            axis,
            keep_side,
            min_area,
        )
        stage_record = {
            "stage_index": stage_index + 1,
            "stage_name": _stage_dir_name(stage_index + 1, stage_count, policy),
            "footprint_ratio": ratio,
            "axis": axis,
            "keep_side": keep_side,
        }
        return stage_entries, metadata, stage_record

    vertical_ratio = float(vertical_progress[stage_index])
    footprint_ratio = float(footprint_progress[stage_index])
    stage_entries, metadata = _generate_hybrid_stage(
        entries,
        vertical_ratio,
        footprint_ratio,
        axis,
        keep_side,
        min_area,
    )
    stage_record = {
        "stage_index": stage_index + 1,
        "stage_name": _stage_dir_name(stage_index + 1, stage_count, policy),
        "vertical_ratio": vertical_ratio,
        "footprint_ratio": footprint_ratio,
        "axis": axis,
        "keep_side": keep_side,
    }
    return stage_entries, metadata, stage_record


def _generate_sequence_for_yaml(
    yaml_path: Path,
    sequence_output_root: Path,
    dataset_output_root: Path | None,
    args: argparse.Namespace,
) -> int:
    sequence_seed = _sequence_seed(args.seed, yaml_path)
    rng = random.Random(sequence_seed)
    policy = _choose_policy(args.policy, rng)
    axis = _choose_axis(args.axis, rng) if policy in {"footprint", "hybrid"} else None
    keep_side = (
        _choose_keep_side(args.keep_side, rng) if policy in {"footprint", "hybrid"} else None
    )

    entries = proxy_stage._load_yaml_entries(yaml_path)
    if not entries:
        print(f"[SKIP] {yaml_path}: no entries")
        return 0

    vertical_progress = (
        _build_progression(args.stage_count, args.min_progress, rng)
        if policy in {"vertical", "hybrid"}
        else None
    )
    footprint_progress = (
        _build_progression(args.stage_count, args.min_progress, rng)
        if policy in {"footprint", "hybrid"}
        else None
    )

    source_root = sequence_output_root / yaml_path.stem
    sequence_dir = source_root / _sequence_dir_name(policy, sequence_seed)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    next_proxy_id = max(int(entry["proxy_id"]) for entry in entries) + 1
    used_proxy_ids: set[int] = set()
    previous_entries: list[dict] = []
    stage_records: list[dict] = []

    for stage_index in range(args.stage_count):
        raw_stage_entries, metadata, stage_record = _generate_stage_entries(
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
        stage_entries, next_proxy_id = _stabilize_stage_entries(
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
        }
        stage_dir = sequence_dir / stage_record["stage_name"]
        _write_stage_outputs(
            stage_entries,
            yaml_path,
            stage_dir,
            metadata,
            args.export_obj,
        )

        previous_entries = [dict(entry) for entry in stage_entries]
        stage_record["entry_count"] = len(stage_entries)
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
    )

    dataset_dir = _dataset_dir_for_sequence(dataset_output_root, sequence_dir, yaml_path)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_stage = _import_dataset_stage()
    dataset_stage.build_pair_dataset(
        sequence_dir=sequence_dir,
        output_dir=dataset_dir,
        pair_mode=args.pair_mode,
        condition_point_count=args.condition_point_count,
        change_tolerance=args.change_tolerance,
        copy_objs=args.copy_objs,
        save_condition_ply_flag=args.save_condition_ply,
    )
    print(f"[DATASET] {yaml_path.name} -> {dataset_dir}")
    return len(stage_records)


def main() -> int:
    args = parse_args()
    yaml_files = proxy_stage._discover_yaml_files(args.input.resolve())
    if not yaml_files:
        print("No input YAML files found.")
        return 1

    generated = 0
    sequence_output_root = args.output.resolve()
    dataset_output_root = args.dataset_output.resolve() if args.dataset_output is not None else None
    for yaml_path in yaml_files:
        generated += _generate_sequence_for_yaml(
            yaml_path,
            sequence_output_root,
            dataset_output_root,
            args,
        )

    print(f"Generated {generated} stage(s) under {sequence_output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

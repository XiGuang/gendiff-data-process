from __future__ import annotations

import argparse
import random
import sys
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_process import generate_construction_proxy as proxy_stage

POLICIES = ("vertical", "footprint", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random but monotonic construction-stage sequences from flat proxy YAML files."
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
                    "source_proxy_id": entry["proxy_id"],
                    "level_index": entry["level_index"],
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
    return proxy_stage._renumber_entries(staged), metadata


def _stage_dir_name(index: int, total: int, policy: str) -> str:
    width = max(2, len(str(total)))
    return f"stage_{index:0{width}d}_{policy}"


def _sequence_dir_name(policy: str, seed: int) -> str:
    return f"sequence_{policy}_seed{seed}"


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
        "stages": stage_records,
    }
    with (sequence_dir / "sequence_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _generate_sequence_for_yaml(
    yaml_path: Path,
    output_root: Path,
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

    source_root = output_root / yaml_path.stem
    sequence_dir = source_root / _sequence_dir_name(policy, sequence_seed)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    stage_records: list[dict] = []
    for stage_index in range(args.stage_count):
        if policy == "vertical":
            ratio = float(vertical_progress[stage_index])
            stage_entries, metadata = proxy_stage._generate_vertical(entries, ratio)
            stage_record = {
                "stage_index": stage_index + 1,
                "stage_name": _stage_dir_name(stage_index + 1, args.stage_count, policy),
                "vertical_ratio": ratio,
            }
        elif policy == "footprint":
            ratio = float(footprint_progress[stage_index])
            stage_entries, metadata = proxy_stage._generate_footprint(
                entries,
                ratio,
                axis,
                keep_side,
                args.min_area,
            )
            stage_record = {
                "stage_index": stage_index + 1,
                "stage_name": _stage_dir_name(stage_index + 1, args.stage_count, policy),
                "footprint_ratio": ratio,
                "axis": axis,
                "keep_side": keep_side,
            }
        else:
            vertical_ratio = float(vertical_progress[stage_index])
            footprint_ratio = float(footprint_progress[stage_index])
            stage_entries, metadata = _generate_hybrid_stage(
                entries,
                vertical_ratio,
                footprint_ratio,
                axis,
                keep_side,
                args.min_area,
            )
            stage_record = {
                "stage_index": stage_index + 1,
                "stage_name": _stage_dir_name(stage_index + 1, args.stage_count, policy),
                "vertical_ratio": vertical_ratio,
                "footprint_ratio": footprint_ratio,
                "axis": axis,
                "keep_side": keep_side,
            }

        metadata["policy"] = policy
        metadata["sequence_seed"] = int(sequence_seed)
        metadata["stage_index"] = stage_index + 1
        metadata["stage_count"] = args.stage_count
        metadata["source_entry_count"] = len(entries)
        stage_dir = sequence_dir / stage_record["stage_name"]
        proxy_stage._write_stage_outputs(
            stage_entries,
            yaml_path,
            stage_dir,
            metadata.copy(),
            args.export_obj,
        )

        stage_record["entry_count"] = len(stage_entries)
        stage_records.append(stage_record)
        print(
            f"[OK] {yaml_path.name} -> {stage_dir.relative_to(output_root)}: "
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
    return len(stage_records)


def main() -> int:
    args = parse_args()
    yaml_files = proxy_stage._discover_yaml_files(args.input.resolve())
    if not yaml_files:
        print("No input YAML files found.")
        return 1

    generated = 0
    for yaml_path in yaml_files:
        generated += _generate_sequence_for_yaml(
            yaml_path,
            args.output.resolve(),
            args,
        )

    print(f"Generated {generated} stage(s) under {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

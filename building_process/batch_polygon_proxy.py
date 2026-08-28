from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_process.polygon_proxy import ProxyConfig, process_obj_to_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate stepped polygon proxies from yingrenshi building tiles."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/block/yingrenshi_building_simple"),
        help="Input root that contains per-tile OBJ folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/block/yingrenshi_building_polygon_proxy"),
        help="Output root for generated polygon proxies.",
    )
    parser.add_argument("--grid-pitch", type=float, default=0.35)
    parser.add_argument("--height-bin", type=float, default=0.5)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--min-stage-height", type=float, default=1.5)
    parser.add_argument("--min-component-area", type=float, default=4.0)
    parser.add_argument(
        "--edge-smooth-radius-factor",
        type=float,
        default=0.75,
        help="Polygon smoothing radius as a multiple of grid pitch.",
    )
    parser.add_argument(
        "--keypoint-snap-tolerance-factor",
        type=float,
        default=0.75,
        help="Remove near-collinear points whose deviation is below this multiple of grid pitch.",
    )
    parser.add_argument(
        "--curve-preserve-angle-deg",
        type=float,
        default=18.0,
        help="Keep more points when the local turning angle exceeds this threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker process count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tiles after filtering.",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="",
        help="Comma-separated tile names to process, e.g. 1_1_1,2_2_1.",
    )
    return parser.parse_args()


def discover_tiles(input_root: Path, selected_tiles: set[str]) -> list[tuple[Path, str, Path]]:
    jobs: list[tuple[Path, str, Path]] = []
    for tile_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        tile_name = tile_dir.name
        if selected_tiles and tile_name not in selected_tiles:
            continue
        obj_path = tile_dir / f"bs_{tile_name}.obj"
        if not obj_path.exists():
            continue
        jobs.append((tile_dir, tile_name, obj_path))
    return jobs


def _run_single(job: tuple[str, str, str, dict]) -> tuple[str, int]:
    obj_path_str, output_dir_str, tile_name, config_payload = job
    config = ProxyConfig(**config_payload)
    artifact = process_obj_to_directory(Path(obj_path_str), Path(output_dir_str), config)
    return tile_name, len(artifact.entries)


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    selected_tiles = {item.strip() for item in args.tiles.split(",") if item.strip()}

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    config = ProxyConfig(
        grid_pitch=args.grid_pitch,
        height_bin=args.height_bin,
        max_levels=args.max_levels,
        min_stage_height=args.min_stage_height,
        min_component_area=args.min_component_area,
        edge_smooth_radius_factor=args.edge_smooth_radius_factor,
        keypoint_snap_tolerance_factor=args.keypoint_snap_tolerance_factor,
        curve_preserve_angle_deg=args.curve_preserve_angle_deg,
    )

    jobs = discover_tiles(input_root, selected_tiles)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        print("No matching tiles found.")
        return 1

    payload = []
    config_payload = config.__dict__
    for tile_dir, tile_name, obj_path in jobs:
        output_dir = output_root / tile_name
        payload.append((str(obj_path), str(output_dir), tile_name, config_payload))

    print(f"Processing {len(payload)} tile(s) into {output_root}")
    completed = 0
    total_entries = 0
    failures: list[str] = []

    if args.workers <= 1:
        iterable = (_run_single(job) for job in payload)
        for tile_name, entry_count in iterable:
            completed += 1
            total_entries += entry_count
            print(f"[{completed}/{len(payload)}] {tile_name}: {entry_count} proxy entries")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_run_single, job): job[2] for job in payload}
            for future in as_completed(future_map):
                tile_name = future_map[future]
                try:
                    _, entry_count = future.result()
                    completed += 1
                    total_entries += entry_count
                    print(f"[{completed}/{len(payload)}] {tile_name}: {entry_count} proxy entries")
                except Exception as exc:
                    failures.append(f"{tile_name}: {exc}")

    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}")
        print(f"Finished with {len(failures)} failure(s).")
        return 1

    print(f"Done. Generated {total_entries} proxy entries across {completed} tile(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

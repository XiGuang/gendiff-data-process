from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_process.polygon_proxy import ProxyArtifact, ProxyConfig, build_proxy_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate polygon proxy outputs for OBJ files in a flat directory."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/component/yingrenshi_change/building"),
        help="Input directory containing OBJ files such as building1.obj.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/polygon_proxy_components_flat"),
        help="Output root. Each OBJ is written into its own subdirectory.",
    )
    parser.add_argument(
        "--objs",
        type=str,
        default="",
        help="Comma-separated OBJ stems or file names to process, e.g. building1,building2.obj.",
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
        help="Process only the first N OBJ files after filtering.",
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
    return parser.parse_args()


def _normalize_obj_name(value: str) -> str:
    name = value.strip()
    if not name:
        return ""
    return Path(name).stem


def discover_objs(input_root: Path, selected_objs: set[str]) -> list[Path]:
    jobs: list[Path] = []
    for obj_path in sorted(input_root.glob("*.obj")):
        if selected_objs and obj_path.stem not in selected_objs:
            continue
        jobs.append(obj_path)
    return jobs


def _flatten_entries(entries: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for entry in entries:
        base_height = float(entry["base_height"])
        height = float(entry["height"])
        contour = [[float(point[0]), float(point[2])] for point in entry["footprint"]]
        flattened.append(
            {
                "proxy_id": entry["proxy_id"],
                "level_index": entry["level_index"],
                "min_height": base_height,
                "max_height": base_height + height,
                "footprint": contour,
            }
        )
    return flattened


def write_flat_proxy_outputs(artifact: ProxyArtifact, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{artifact.source_obj.stem}.yaml"
    flat_entries = _flatten_entries(artifact.entries)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(flat_entries, handle, sort_keys=False)

    if artifact.mesh is not None:
        artifact.mesh.export(output_dir / artifact.source_obj.name)

    metadata = {
        "source_obj": str(artifact.source_obj),
        "levels": artifact.levels,
        "config": config_to_dict(artifact.config),
        "proxy_count": len(flat_entries),
        "output_format": {
            "footprint": "xz_2d",
            "height_fields": ["min_height", "max_height"],
        },
    }
    with (output_dir / "polygon_proxy_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def config_to_dict(config: ProxyConfig) -> dict:
    return {
        "grid_pitch": config.grid_pitch,
        "height_bin": config.height_bin,
        "max_levels": config.max_levels,
        "min_stage_height": config.min_stage_height,
        "min_component_area": config.min_component_area,
        "min_component_faces": config.min_component_faces,
        "min_fragment_projected_area": config.min_fragment_projected_area,
        "stage_support_ratio": config.stage_support_ratio,
        "cluster_gap": config.cluster_gap,
        "padding_cells": config.padding_cells,
        "simplify_tolerance_factor": config.simplify_tolerance_factor,
        "edge_smooth_radius_factor": config.edge_smooth_radius_factor,
        "keypoint_snap_tolerance_factor": config.keypoint_snap_tolerance_factor,
        "curve_preserve_angle_deg": config.curve_preserve_angle_deg,
        "ray_batch_size": config.ray_batch_size,
    }


def _run_single(job: tuple[str, str, str, dict]) -> tuple[str, int]:
    obj_path_str, output_dir_str, obj_name, config_payload = job
    config = ProxyConfig(**config_payload)
    artifact = build_proxy_artifact(Path(obj_path_str), config)
    write_flat_proxy_outputs(artifact, Path(output_dir_str))
    return obj_name, len(artifact.entries)


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    selected_objs = {_normalize_obj_name(item) for item in args.objs.split(",") if item.strip()}

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_root}")

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

    sources = discover_objs(input_root, selected_objs)
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        print("No matching OBJ files found.")
        return 1

    payload = []
    config_payload = config.__dict__
    for obj_path in sources:
        obj_name = obj_path.stem
        output_dir = output_root / obj_name
        payload.append((str(obj_path), str(output_dir), obj_name, config_payload))

    print(f"Processing {len(payload)} OBJ file(s) into {output_root}")
    completed = 0
    total_entries = 0
    failures: list[str] = []

    if args.workers <= 1:
        iterable = (_run_single(job) for job in payload)
        for obj_name, entry_count in iterable:
            completed += 1
            total_entries += entry_count
            print(f"[{completed}/{len(payload)}] {obj_name}: {entry_count} proxy entries")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_run_single, job): job[2] for job in payload}
            for future in as_completed(future_map):
                obj_name = future_map[future]
                try:
                    _, entry_count = future.result()
                    completed += 1
                    total_entries += entry_count
                    print(f"[{completed}/{len(payload)}] {obj_name}: {entry_count} proxy entries")
                except Exception as exc:
                    failures.append(f"{obj_name}: {exc}")

    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}")
        print(f"Finished with {len(failures)} failure(s).")
        return 1

    print(f"Done. Generated {total_entries} proxy entries across {completed} OBJ file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

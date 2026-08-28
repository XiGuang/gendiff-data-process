from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_process.polygon_proxy import (
    ProxyConfig,
    compare_meshes_topdown,
    load_obj_mesh,
    process_obj_to_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate polygon proxies for a few tiles and validate top-down metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/block/yingrenshi_building_simple"),
        help="Input tile root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/polygon_proxy_smoke"),
        help="Temporary output root for generated proxies.",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=["1_1_1", "2_2_1", "10_4_4"],
        help="Tile names to process.",
    )
    parser.add_argument("--grid-pitch", type=float, default=0.35)
    parser.add_argument("--height-bin", type=float, default=0.5)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--min-stage-height", type=float, default=1.5)
    parser.add_argument("--min-component-area", type=float, default=4.0)
    parser.add_argument("--edge-smooth-radius-factor", type=float, default=0.75)
    parser.add_argument("--keypoint-snap-tolerance-factor", type=float, default=0.75)
    parser.add_argument("--curve-preserve-angle-deg", type=float, default=18.0)
    parser.add_argument("--max-bbox-delta-xz", type=float, default=0.35)
    parser.add_argument("--max-bbox-delta-y", type=float, default=2.0)
    parser.add_argument("--min-footprint-iou", type=float, default=0.75)
    parser.add_argument("--max-top-height-mae", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    failures: list[str] = []
    for tile in args.tiles:
        source_dir = args.input / tile
        source_obj = source_dir / f"bs_{tile}.obj"
        output_dir = args.output / tile
        artifact = process_obj_to_directory(source_obj, output_dir, config)
        if artifact.mesh is None:
            failures.append(f"{tile}: generated proxy mesh is empty")
            continue

        source_mesh = load_obj_mesh(source_obj)
        metrics = compare_meshes_topdown(source_mesh, artifact.mesh, config.grid_pitch)
        print(
            f"{tile}: proxies={len(artifact.entries)} "
            f"iou={metrics['footprint_iou']:.3f} "
            f"mae={metrics['top_height_mae']:.3f} "
            f"bbox_xz={metrics['bbox_delta_xz']:.3f} "
            f"bbox_y={metrics['bbox_delta_y']:.3f}"
        )

        if metrics["bbox_delta_xz"] > args.max_bbox_delta_xz + 1e-6:
            failures.append(
                f"{tile}: bbox_delta_xz {metrics['bbox_delta_xz']:.3f} > {args.max_bbox_delta_xz:.3f}"
            )
        if metrics["bbox_delta_y"] > args.max_bbox_delta_y + 1e-6:
            failures.append(
                f"{tile}: bbox_delta_y {metrics['bbox_delta_y']:.3f} > {args.max_bbox_delta_y:.3f}"
            )
        if metrics["footprint_iou"] + 1e-6 < args.min_footprint_iou:
            failures.append(
                f"{tile}: footprint_iou {metrics['footprint_iou']:.3f} < {args.min_footprint_iou:.3f}"
            )
        if metrics["top_height_mae"] > args.max_top_height_mae + 1e-6:
            failures.append(
                f"{tile}: top_height_mae {metrics['top_height_mae']:.3f} > {args.max_top_height_mae:.3f}"
            )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

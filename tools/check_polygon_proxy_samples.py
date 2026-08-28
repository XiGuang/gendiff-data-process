from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polygon_proxy.core import (
    ProxyConfig,
    approximation_error,
    batch_convert,
    load_config_file,
    load_obj_mesh,
)


DEFAULT_TILES = ["0_1_1", "3_2_16", "4_4_12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sample polygon proxy validation sweep.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./data/block/yingrenshi_building_simple"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./output/polygon_proxy_samples"),
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default=",".join(DEFAULT_TILES),
        help="Comma-separated sample tile ids.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON or YAML config file overriding ProxyConfig thresholds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-mean-error",
        type=float,
        default=4.0,
        help="Maximum allowed mean sampled source-to-proxy distance.",
    )
    parser.add_argument(
        "--max-face-ratio",
        type=float,
        default=0.15,
        help="Maximum allowed proxy-face / input-triangle ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config_file(args.config) if args.config else ProxyConfig()
    tiles = [tile.strip() for tile in args.tiles.split(",") if tile.strip()]
    results = batch_convert(
        input_root=args.input_root,
        output_root=args.output_root,
        config=config,
        tiles=tiles,
        overwrite=True,
        workers=max(args.workers, 1),
    )

    report: list[dict[str, object]] = []
    failed = False
    for result in results:
        tile_id = str(result["tile_id"])
        source_obj = args.input_root / tile_id / f"bs_{tile_id}.obj"
        proxy_obj = args.output_root / tile_id / f"{tile_id}.proxy.obj"
        source_mesh = load_obj_mesh(source_obj, config)
        proxy_mesh = load_obj_mesh(proxy_obj, config)
        mean_error = approximation_error(source_mesh, proxy_mesh, config.max_sample_points)
        face_ratio = float(result["proxy_faces"]) / max(float(result["input_triangles"]), 1.0)
        tile_report = {
            "tile_id": tile_id,
            "input_triangles": int(result["input_triangles"]),
            "proxy_faces": int(result["proxy_faces"]),
            "buildings_detected": int(result["buildings_detected"]),
            "mean_sample_error": round(mean_error, 6),
            "face_ratio": round(face_ratio, 6),
        }
        report.append(tile_report)
        if mean_error > args.max_mean_error or face_ratio > args.max_face_ratio:
            failed = True

    print(json.dumps({"tiles": report, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

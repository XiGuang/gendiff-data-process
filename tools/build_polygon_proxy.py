from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polygon_proxy.core import batch_convert, load_config_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-convert tiled building meshes into polygon proxy outputs.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./data/block/yingrenshi_building_simple"),
        help="Root directory containing tile folders and bs_<tile>.obj files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./output/polygon_proxy"),
        help="Output directory for generated proxy JSON/OBJ artifacts.",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default="",
        help="Comma-separated tile ids to process. Leave empty to process all discovered tiles.",
    )
    parser.add_argument(
        "--tiles-file",
        type=Path,
        default=None,
        help="Optional text file listing one tile id per line.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs in the target directory.",
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
        help="Number of worker processes to use.",
    )
    return parser.parse_args()


def resolve_tiles(args: argparse.Namespace) -> list[str]:
    tiles: list[str] = []
    if args.tiles:
        tiles.extend([tile.strip() for tile in args.tiles.split(",") if tile.strip()])
    if args.tiles_file is not None:
        text = args.tiles_file.read_text(encoding="utf-8")
        tiles.extend([line.strip() for line in text.splitlines() if line.strip()])
    return sorted(set(tiles))


def main() -> int:
    args = parse_args()
    config = load_config_file(args.config)
    tiles = resolve_tiles(args)
    results = batch_convert(
        input_root=args.input_root,
        output_root=args.output_root,
        config=config,
        tiles=tiles or None,
        overwrite=args.overwrite,
        workers=max(args.workers, 1),
    )
    summary = {
        "tiles_requested": tiles if tiles else "ALL",
        "tiles_processed": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

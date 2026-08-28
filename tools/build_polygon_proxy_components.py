from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polygon_proxy.core import (
    ProxyConfig,
    build_proxy_artifact,
    load_config_file,
    write_proxy_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert a flat directory of OBJ meshes into polygon proxy outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./data/component/yingrenshi_change/yingrenshi_building_simple"),
        help="Directory containing OBJ files such as building1.obj, building2.obj.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./output/polygon_proxy_components"),
        help="Output directory for generated proxy JSON/OBJ artifacts.",
    )
    parser.add_argument(
        "--objs",
        type=str,
        default="",
        help="Comma-separated OBJ stem names or file names to process. Leave empty to process all OBJ files.",
    )
    parser.add_argument(
        "--objs-file",
        type=Path,
        default=None,
        help="Optional text file listing one OBJ stem or file name per line.",
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


def _normalize_obj_name(value: str) -> str:
    name = value.strip()
    if not name:
        return ""
    return Path(name).stem


def resolve_requested_objs(args: argparse.Namespace) -> list[str]:
    obj_names: list[str] = []
    if args.objs:
        obj_names.extend(_normalize_obj_name(item) for item in args.objs.split(","))
    if args.objs_file is not None:
        text = args.objs_file.read_text(encoding="utf-8")
        obj_names.extend(_normalize_obj_name(line) for line in text.splitlines())
    return sorted({name for name in obj_names if name})


def discover_obj_sources(input_root: Path, selected: Iterable[str] | None) -> list[Path]:
    selected_set = set(selected or [])
    sources: list[Path] = []
    for obj_path in sorted(input_root.glob("*.obj")):
        if selected_set and obj_path.stem not in selected_set:
            continue
        sources.append(obj_path)
    return sources


def _worker(job: tuple[str, str, ProxyConfig, bool]) -> dict[str, object]:
    source_obj, output_root, config, overwrite = job
    artifact = build_proxy_artifact(source_obj, config)
    json_path, obj_path, metrics_path = write_proxy_outputs(artifact, output_root, overwrite=overwrite)
    result = artifact.metrics.copy()
    result.update(
        {
            "tile_id": artifact.tile_id,
            "json_path": str(json_path),
            "obj_path": str(obj_path),
            "metrics_path": str(metrics_path),
        }
    )
    return result


def batch_convert_components(
    input_root: Path,
    output_root: Path,
    config: ProxyConfig,
    objs: Iterable[str] | None = None,
    overwrite: bool = False,
    workers: int = 1,
) -> list[dict[str, object]]:
    sources = discover_obj_sources(input_root, objs)
    jobs = [(str(source), str(output_root), config, overwrite) for source in sources]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_worker, jobs))
    return [_worker(job) for job in jobs]


def main() -> int:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if not args.input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {args.input_root}")

    config = load_config_file(args.config)
    requested_objs = resolve_requested_objs(args)
    results = batch_convert_components(
        input_root=args.input_root,
        output_root=args.output_root,
        config=config,
        objs=requested_objs or None,
        overwrite=args.overwrite,
        workers=max(args.workers, 1),
    )
    summary = {
        "objs_requested": requested_objs if requested_objs else "ALL",
        "objs_processed": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

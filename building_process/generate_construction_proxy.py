from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import trimesh
import yaml
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

EPS = 1e-6
SKIP_FILENAMES = {"polygon_proxy_meta.yaml", "construction_meta.yaml", "data.yaml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate construction-stage proxy YAMLs from flat polygon proxy YAML files."
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
        help="Output directory for generated stage subdirectories.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("vertical", "footprint"),
        default=("vertical", "footprint"),
        help="Construction-stage generation modes to run.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=(0.33, 0.5, 0.66),
        help="Progress ratios in (0, 1], e.g. 0.33 0.5 0.66.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "z"),
        default="x",
        help="Axis used by footprint clipping.",
    )
    parser.add_argument(
        "--keep-side",
        choices=("min", "max"),
        default="min",
        help="Which side of the footprint bounds to keep during footprint clipping.",
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


def _validate_ratio(value: float) -> float:
    ratio = float(value)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    return ratio


def _ratio_tag(ratio: float) -> str:
    text = f"{ratio:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _load_yaml_entries(yaml_path: Path) -> list[dict]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list):
        raise ValueError(f"YAML root must be a list: {yaml_path}")

    entries: list[dict] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"Entry {index} must be a mapping in {yaml_path}")
        footprint = raw.get("footprint")
        if not isinstance(footprint, list):
            raise ValueError(f"Entry {index} footprint must be a list in {yaml_path}")
        points = [[float(point[0]), float(point[1])] for point in footprint]
        if len(points) < 3:
            raise ValueError(f"Entry {index} footprint must contain at least 3 points in {yaml_path}")

        min_height = float(raw["min_height"])
        max_height = float(raw["max_height"])
        if max_height - min_height <= EPS:
            raise ValueError(f"Entry {index} has non-positive height in {yaml_path}")

        entries.append(
            {
                "proxy_id": int(raw.get("proxy_id", index)),
                "level_index": int(raw.get("level_index", 0)),
                "min_height": min_height,
                "max_height": max_height,
                "footprint": points,
            }
        )
    return entries


def _iter_polygons(geometry: Polygon | MultiPolygon | GeometryCollection) -> Iterable[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    polygons: list[Polygon] = []
    for geom in geometry.geoms:
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, (MultiPolygon, GeometryCollection)):
            polygons.extend(_iter_polygons(geom))
    return polygons


def _entry_polygon(entry: dict) -> Polygon:
    polygon = Polygon(entry["footprint"])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError(f"Invalid or empty footprint for proxy_id={entry['proxy_id']}")
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    return polygon


def _global_height_bounds(entries: list[dict]) -> tuple[float, float]:
    min_height = min(entry["min_height"] for entry in entries)
    max_height = max(entry["max_height"] for entry in entries)
    return float(min_height), float(max_height)


def _global_planar_bounds(entries: list[dict]) -> tuple[float, float, float, float]:
    xs = [point[0] for entry in entries for point in entry["footprint"]]
    zs = [point[1] for entry in entries for point in entry["footprint"]]
    return float(min(xs)), float(max(xs)), float(min(zs)), float(max(zs))


def _normalize_polygon(polygon: Polygon, min_area: float) -> list[list[float]]:
    cleaned = polygon.buffer(0)
    if cleaned.is_empty:
        return []
    if isinstance(cleaned, MultiPolygon):
        cleaned = max(cleaned.geoms, key=lambda geom: geom.area)
    if cleaned.area < min_area:
        return []
    coords = list(cleaned.exterior.coords)[:-1]
    if len(coords) < 3:
        return []
    result = [[float(x), float(z)] for x, z in coords]
    if not cleaned.exterior.is_ccw:
        result.reverse()
    return result


def _renumber_entries(entries: list[dict]) -> list[dict]:
    ordered = sorted(
        entries,
        key=lambda item: (
            item["level_index"],
            item["min_height"],
            item["max_height"],
            item["source_proxy_id"],
        ),
    )
    normalized: list[dict] = []
    for proxy_id, entry in enumerate(ordered):
        normalized.append(
            {
                "proxy_id": proxy_id,
                "source_proxy_id": entry["source_proxy_id"],
                "level_index": entry["level_index"],
                "min_height": float(entry["min_height"]),
                "max_height": float(entry["max_height"]),
                "footprint": entry["footprint"],
            }
        )
    return normalized


def _generate_vertical(entries: list[dict], ratio: float) -> tuple[list[dict], dict]:
    global_min, global_max = _global_height_bounds(entries)
    target_height = global_min + ratio * (global_max - global_min)
    staged: list[dict] = []
    for entry in entries:
        entry_min = float(entry["min_height"])
        entry_max = float(entry["max_height"])
        if entry_min >= target_height - EPS:
            continue
        clipped_max = min(entry_max, target_height)
        if clipped_max - entry_min <= EPS:
            continue
        staged.append(
            {
                "source_proxy_id": entry["proxy_id"],
                "level_index": entry["level_index"],
                "min_height": entry_min,
                "max_height": float(clipped_max),
                "footprint": [[float(x), float(z)] for x, z in entry["footprint"]],
            }
        )
    metadata = {
        "mode": "vertical",
        "ratio": float(ratio),
        "global_min_height": global_min,
        "global_max_height": global_max,
        "target_height": float(target_height),
    }
    return _renumber_entries(staged), metadata


def _build_clip_box(
    bounds: tuple[float, float, float, float],
    axis: str,
    keep_side: str,
    ratio: float,
) -> Polygon:
    min_x, max_x, min_z, max_z = bounds
    if axis == "x":
        span = max_x - min_x
        if span <= EPS:
            return box(min_x, min_z, max_x, max_z)
        if keep_side == "min":
            clip_max = min_x + span * ratio
            return box(min_x - EPS, min_z - EPS, clip_max + EPS, max_z + EPS)
        clip_min = max_x - span * ratio
        return box(clip_min - EPS, min_z - EPS, max_x + EPS, max_z + EPS)

    span = max_z - min_z
    if span <= EPS:
        return box(min_x, min_z, max_x, max_z)
    if keep_side == "min":
        clip_max = min_z + span * ratio
        return box(min_x - EPS, min_z - EPS, max_x + EPS, clip_max + EPS)
    clip_min = max_z - span * ratio
    return box(min_x - EPS, clip_min - EPS, max_x + EPS, max_z + EPS)


def _generate_footprint(
    entries: list[dict],
    ratio: float,
    axis: str,
    keep_side: str,
    min_area: float,
) -> tuple[list[dict], dict]:
    bounds = _global_planar_bounds(entries)
    clip_region = _build_clip_box(bounds, axis, keep_side, ratio)
    staged: list[dict] = []

    for entry in entries:
        clipped = _entry_polygon(entry).intersection(clip_region)
        for polygon in _iter_polygons(clipped):
            contour = _normalize_polygon(polygon, min_area)
            if not contour:
                continue
            staged.append(
                {
                    "source_proxy_id": entry["proxy_id"],
                    "level_index": entry["level_index"],
                    "min_height": float(entry["min_height"]),
                    "max_height": float(entry["max_height"]),
                    "footprint": contour,
                }
            )

    min_x, max_x, min_z, max_z = bounds
    metadata = {
        "mode": "footprint",
        "ratio": float(ratio),
        "axis": axis,
        "keep_side": keep_side,
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
    return _renumber_entries(staged), metadata


def _mesh_from_entry(entry: dict) -> trimesh.Trimesh:
    polygon = Polygon(entry["footprint"])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError(f"Cannot export empty polygon for proxy_id={entry['proxy_id']}")
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda geom: geom.area)
    if not polygon.exterior.is_ccw:
        polygon = Polygon(list(polygon.exterior.coords)[::-1])

    height = float(entry["max_height"] - entry["min_height"])
    mesh = trimesh.creation.extrude_polygon(polygon, height=height)
    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, float(entry["min_height"])],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    mesh.apply_transform(transform)
    return mesh


def _export_stage_obj(entries: list[dict], obj_path: Path) -> None:
    meshes = [_mesh_from_entry(entry) for entry in entries]
    if not meshes:
        return
    merged = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    merged.export(obj_path)


def _stage_name(mode: str, ratio: float, axis: str, keep_side: str) -> str:
    if mode == "vertical":
        return f"vertical_{_ratio_tag(ratio)}"
    return f"footprint_{axis}_{keep_side}_{_ratio_tag(ratio)}"


def _write_stage_outputs(
    entries: list[dict],
    source_yaml: Path,
    output_dir: Path,
    metadata: dict,
    export_obj: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / source_yaml.name
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(entries, handle, sort_keys=False)

    meta_payload = {
        "source_yaml": str(source_yaml),
        "source_entry_count": int(metadata.pop("source_entry_count")),
        "stage_entry_count": len(entries),
        **metadata,
    }
    with (output_dir / "construction_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta_payload, handle, sort_keys=False)

    if export_obj and entries:
        _export_stage_obj(entries, output_dir / f"{source_yaml.stem}.obj")


def _discover_yaml_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*.yaml")
            if path.name not in SKIP_FILENAMES
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output.resolve()
    ratios = [_validate_ratio(value) for value in args.ratios]
    yaml_files = _discover_yaml_files(input_path)
    if not yaml_files:
        print("No input YAML files found.")
        return 1

    generated_count = 0
    for yaml_path in yaml_files:
        entries = _load_yaml_entries(yaml_path)
        if not entries:
            print(f"[SKIP] {yaml_path}: no entries")
            continue

        source_output_root = output_root / yaml_path.stem
        for mode in args.modes:
            for ratio in ratios:
                if mode == "vertical":
                    stage_entries, metadata = _generate_vertical(entries, ratio)
                else:
                    stage_entries, metadata = _generate_footprint(
                        entries,
                        ratio,
                        args.axis,
                        args.keep_side,
                        args.min_area,
                    )
                metadata["source_entry_count"] = len(entries)
                stage_dir = source_output_root / _stage_name(
                    mode,
                    ratio,
                    args.axis,
                    args.keep_side,
                )
                _write_stage_outputs(
                    stage_entries,
                    yaml_path,
                    stage_dir,
                    metadata,
                    args.export_obj,
                )
                generated_count += 1
                print(
                    f"[OK] {yaml_path.name} -> {stage_dir.name}: "
                    f"{len(stage_entries)} entries"
                )

    print(f"Generated {generated_count} stage output(s) under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

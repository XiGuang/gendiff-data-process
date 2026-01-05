#!/usr/bin/env python3
"""Recursively compute bounding boxes of OBJ files in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

BBox = Tuple[float, float, float, float, float, float]


def compute_bbox(obj_path: Path) -> BBox:
    """Return the bounding box (x_min, x_max, y_min, y_max, z_min, z_max)."""
    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")
    vertex_found = False

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.lstrip()
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertex_found = True
            x, y, z = map(float, parts[1:4])
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
            if z < z_min:
                z_min = z
            if z > z_max:
                z_max = z

    if not vertex_found:
        raise ValueError(f"No vertices found in OBJ: {obj_path}")

    return x_min, x_max, y_min, y_max, z_min, z_max


def format_bbox(bbox: BBox) -> str:
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return (
        f"x_min={x_min:.6f} x_max={x_max:.6f} "
        f"y_min={y_min:.6f} y_max={y_max:.6f} "
        f"z_min={z_min:.6f} z_max={z_max:.6f}"
    )


def combine(bboxes: Iterable[BBox]) -> BBox:
    iterator = iter(bboxes)
    try:
        x_min, x_max, y_min, y_max, z_min, z_max = next(iterator)
    except StopIteration:
        raise ValueError("Cannot combine empty bounding box list") from None

    for box in iterator:
        bx_min, bx_max, by_min, by_max, bz_min, bz_max = box
        if bx_min < x_min:
            x_min = bx_min
        if bx_max > x_max:
            x_max = bx_max
        if by_min < y_min:
            y_min = by_min
        if by_max > y_max:
            y_max = by_max
        if bz_min < z_min:
            z_min = bz_min
        if bz_max > z_max:
            z_max = bz_max
    return x_min, x_max, y_min, y_max, z_min, z_max


def collect_obj_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.obj"):
        if path.is_file():
            yield path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively compute OBJ bounding boxes.",
    )
    parser.add_argument("root", type=Path, help="Folder that contains OBJ files")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the combined bounding box over all files.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.root

    if not root.exists():
        print(f"Folder does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Path is not a directory: {root}", file=sys.stderr)
        return 1

    obj_files = list(collect_obj_files(root))
    if not obj_files:
        print("No OBJ files found", file=sys.stderr)
        return 1

    per_file_results = []
    for obj_path in obj_files:
        try:
            bbox = compute_bbox(obj_path)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        per_file_results.append((obj_path, bbox))
        print(f"{obj_path}: {format_bbox(bbox)}")

    if not per_file_results:
        print("No usable OBJ files found", file=sys.stderr)
        return 1

    if not args.no_summary:
        summary_bbox = combine(bbox for _, bbox in per_file_results)
        print("SUMMARY:", format_bbox(summary_bbox))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

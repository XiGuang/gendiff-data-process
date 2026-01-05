#!/usr/bin/env python3
"""Count OBJ files whose Y-axis height exceeds a threshold."""

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional


def iter_obj_files(root: Path, recursive: bool) -> Iterable[Path]:
    searcher = root.rglob if recursive else root.glob
    yield from searcher("*.obj")


def compute_y_height(obj_path: Path) -> Optional[float]:
    min_y = math.inf
    max_y = -math.inf
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    y_coord = float(parts[2])
                except ValueError:
                    continue
                min_y = min(min_y, y_coord)
                max_y = max(max_y, y_coord)
    except OSError as exc:  # file may be unreadable
        raise RuntimeError(f"Failed to read {obj_path}: {exc}") from exc

    if min_y is math.inf:
        return None
    return max_y - min_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count OBJ meshes whose Y-axis height exceeds a threshold."
    )
    parser.add_argument("folder", type=Path, help="Directory containing OBJ files")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=True,
        help="Height threshold along the Y axis (same units as OBJ coordinates)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a valid directory")

    total = 0
    exceeding = 0
    skipped = 0

    for obj_path in iter_obj_files(args.folder, args.recursive):
        total += 1
        try:
            height = compute_y_height(obj_path)
        except RuntimeError as err:
            print(err)
            skipped += 1
            continue

        if height is None:
            skipped += 1
            continue

        if height > args.threshold:
            exceeding += 1

    print(f"Total OBJ files scanned: {total}")
    print(f"Files without vertex data or failed reads: {skipped}")
    print(
        f"Files with Y-axis height > {args.threshold}: {exceeding}"
    )


if __name__ == "__main__":
    main()

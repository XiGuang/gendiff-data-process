#!/usr/bin/env python3
"""Check OBJ files for axis-aligned size thresholds using multiprocessing."""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan OBJ files under a directory, measure their axis-aligned bounds, "
            "and list the files whose extents exceed the given threshold along X, Y, and Z."
        )
    )
    parser.add_argument("root", type=Path, help="Directory whose subfolders contain OBJ files.")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=True,
        help="Minimum required size along each axis (same value applied to X, Y, and Z).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("obj_over_threshold.txt"),
        help="Path to the output txt file (default: ./obj_over_threshold.txt).",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=None, help="Number of worker processes (default: CPU count)."
    )
    parser.add_argument(
        "--suffix",
        default=".obj",
        help="File extension to scan (default: .obj).",
    )
    return parser.parse_args()


def iter_obj_files(root: Path, suffix: str) -> Iterable[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {root}")
    suffix = suffix.lower()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == suffix:
            yield path


def compute_extents(obj_path: Path) -> Tuple[float, float, float]:
    min_xyz = [math.inf, math.inf, math.inf]
    max_xyz = [-math.inf, -math.inf, -math.inf]
    found_vertex = False

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                continue
            found_vertex = True
            for idx, value in enumerate((x, y, z)):
                if value < min_xyz[idx]:
                    min_xyz[idx] = value
                if value > max_xyz[idx]:
                    max_xyz[idx] = value

    if not found_vertex:
        raise ValueError(f"No valid vertex lines found in {obj_path}")

    return tuple(max_xyz[i] - min_xyz[i] for i in range(3))


def worker_task(obj_path: str, threshold: float) -> Tuple[str, Tuple[float, float, float], bool, str | None]:
    path = Path(obj_path)
    try:
        extents = compute_extents(path)
        fits = all(extent > threshold for extent in extents)
        return (obj_path, extents, fits, None)
    except Exception as exc:  # noqa: BLE001 - collect any per-file issue for reporting
        return (obj_path, (0.0, 0.0, 0.0), False, str(exc))


def write_output(output_path: Path, threshold: float, qualifying: list[tuple[str, Tuple[float, float, float]]], total: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"threshold\t{threshold}\n")
        handle.write(f"total_files\t{total}\n")
        handle.write(f"qualified\t{len(qualifying)}\n")
        handle.write("path\textent_x\textent_y\textent_z\n")
        for path, extents in qualifying:
            handle.write(f"{path}\t{extents[0]:.6f}\t{extents[1]:.6f}\t{extents[2]:.6f}\n")


def main() -> None:
    args = parse_args()
    obj_files = list(iter_obj_files(args.root, args.suffix))
    if not obj_files:
        print(f"No files ending with '{args.suffix}' found under {args.root}.")
        return

    qualifying: list[tuple[str, Tuple[float, float, float]]] = []
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker_task, str(path), args.threshold): path for path in obj_files
        }
        for future in as_completed(futures):
            obj_path_str, extents, fits, error = future.result()
            if error:
                failures.append((obj_path_str, error))
            elif fits:
                qualifying.append((obj_path_str, extents))

    write_output(args.output, args.threshold, qualifying, len(obj_files))

    print(f"Processed {len(obj_files)} OBJ files under {args.root}.")
    print(f"Qualified files (> {args.threshold} on X/Y/Z): {len(qualifying)}.")
    if failures:
        print(f"Files with errors: {len(failures)}.")
        for path, message in failures[:10]:  # cap to avoid noisy output
            print(f"  {path}: {message}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rotate point clouds around the Y axis and save angle-specific copies."""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

ROTATIONS: List[Tuple[float, str]] = [
    (0.0, "r0"),
    (90.0, "r90"),
    (180.0, "r180"),
    (270.0, "r270"),
]

# Precompute trig values to avoid repeating math in each worker
_ROTATION_CACHE = {
    angle: (math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle, _ in ROTATIONS
}


def _rotate_y(points: torch.Tensor, cos_val: float, sin_val: float) -> torch.Tensor:
    """Rotate a (N, 3) tensor around the Y axis by using cached trig values."""
    x = points[:, 0] * cos_val + points[:, 2] * sin_val
    y = points[:, 1]
    z = -points[:, 0] * sin_val + points[:, 2] * cos_val
    return torch.stack((x, y, z), dim=1)


def _process_file(file_path: str, output_dir: str, overwrite: bool) -> Tuple[str, bool, str]:
    """Worker entry point executed inside a process."""
    path = Path(file_path)
    out_dir = Path(output_dir)

    try:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if 'points' in data:
                tensor = data['points']
            else:
                # 如果字典里没有 points key，抛出异常
                raise KeyError(f"Loaded dict but 'points' key is missing. Keys found: {list(data.keys())}")
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            raise TypeError(f"Expected torch.Tensor or dict, got {type(data)!r}")
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError(f"Tensor {path.name} must have shape (N, 3), found {tuple(tensor.shape)}")

        tensor = tensor.to(dtype=torch.float32).contiguous()
        saved_files = []

        for angle, suffix in ROTATIONS:
            out_path = out_dir / f"{path.stem}_{suffix}.pt"
            if out_path.exists() and not overwrite:
                saved_files.append(out_path.name)
                continue

            if angle % 360 == 0:
                rotated = tensor.clone()
            else:
                cos_val, sin_val = _ROTATION_CACHE[angle]
                rotated = _rotate_y(tensor, cos_val, sin_val)

            torch.save(rotated, out_path)
            saved_files.append(out_path.name)

        return path.name, True, f"Generated: {', '.join(saved_files)}"
    except Exception as exc:  # pragma: no cover - defensive logging
        return path.name, False, f"Failed: {exc}"


def _iter_pt_files(input_dir: Path) -> Iterable[Path]:
    for item in sorted(input_dir.iterdir()):
        if item.is_file() and item.suffix == ".pt":
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate .pt point clouds around the Y axis.")
    parser.add_argument("--input_dir", required=True, help="Directory that contains source .pt files.")
    parser.add_argument("--output_dir", required=True, help="Directory to store rotated .pt files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of processes to use (default: cpu_count).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rotated files instead of skipping them.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = list(_iter_pt_files(input_dir))
    if not files:
        print(f"No .pt files found under {input_dir}")
        return

    worker_count = max(1, args.workers)
    print(f"Processing {len(files)} files with {worker_count} workers...")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_file, str(path), str(output_dir), args.overwrite): path for path in files
        }

        completed = 0
        for future in as_completed(futures):
            filename = futures[future].name
            completed += 1
            ok, msg = future.result()[1:]
            status = "OK" if ok else "ERR"
            if status == "ERR":
                print(f"[{completed}/{len(files)}][{status}] {filename}: {msg}")


if __name__ == "__main__":
    main()

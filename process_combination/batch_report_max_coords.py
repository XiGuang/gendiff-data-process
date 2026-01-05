#!/usr/bin/env python3
"""Report per-axis maxima for .pt point clouds and flag oversized files."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import torch


def _load_points(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if "points" not in data:
            raise KeyError(
                f"Loaded dict but 'points' key is missing. Keys found: {list(data.keys())}"
            )
        tensor = data["points"]
    elif isinstance(data, torch.Tensor):
        tensor = data
    else:
        raise TypeError(f"Expected torch.Tensor or dict, got {type(data)!r}")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"Tensor {path.name} must have shape (N, 3), found {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32).contiguous()


def _inspect_file(file_path: str, threshold: float | None) -> Tuple[str, bool, Tuple[float, float, float], bool, str]:
    path = Path(file_path)
    try:
        tensor = _load_points(path)
        max_vals = tensor.max(dim=0).values.tolist()
        maxima = tuple(float(val) for val in max_vals)
        exceeds = threshold is not None and any(val > threshold for val in maxima)
        return path.name, True, maxima, exceeds, ""
    except Exception as exc:  # pragma: no cover - defensive logging
        return path.name, False, (float("nan"),) * 3, False, str(exc)


def _iter_pt_files(input_dir: Path) -> Iterable[Path]:
    for item in sorted(input_dir.iterdir()):
        if item.is_file() and item.suffix == ".pt":
            yield item


def _format_maxima(maxima: Tuple[float, float, float], precision: int) -> str:
    return (
        f"max_x={maxima[0]:.{precision}f}, "
        f"max_y={maxima[1]:.{precision}f}, "
        f"max_z={maxima[2]:.{precision}f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-axis maxima for .pt point clouds and flag large coordinates."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source .pt files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of processes to use (default: cpu_count).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Flag files whose max coordinate exceeds this value (any axis).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for reporting maxima (default: 4).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = list(_iter_pt_files(input_dir))
    if not files:
        print(f"No .pt files found under {input_dir}")
        return

    worker_count = max(1, args.workers)
    print(f"Processing {len(files)} files with {worker_count} workers...")

    flagged: List[str] = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_inspect_file, str(path), args.threshold): path for path in files
        }

        completed = 0
        for future in as_completed(futures):
            filename = futures[future].name
            completed += 1
            ok, maxima, exceeds, err = future.result()[1:]
            if ok:
                msg = _format_maxima(maxima, args.precision)
                if exceeds:
                    flagged.append(filename)
                    msg += " [FLAGGED]"
                status = "OK"
            else:
                msg = f"Failed: {err}"
                status = "ERR"
            # print(f"[{completed}/{len(files)}][{status}] {filename}: {msg}")
            if status == "ERR":
                print(f"[{completed}/{len(files)}][{status}] {filename}: {msg}")

    if args.threshold is not None:
        if flagged:
            print("\nFiles exceeding threshold:")
            for name in flagged:
                print(f"- {name}")
        else:
            print("\nNo files exceeded the threshold.")


if __name__ == "__main__":
    main()

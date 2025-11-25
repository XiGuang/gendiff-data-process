#!/usr/bin/env python3
"""
Convert a point cloud saved as a .pt file (PyTorch tensor of shape [N, 3])
into a .ply file.

Usage:
  python convert_pt_to_ply.py --input path/to/points.pt --output path/to/points.ply

Notes:
- The input .pt is expected to be a torch Tensor (or numpy array) with shape (N, >=3).
- If the .pt contains a dict, the script will try common keys like 'points', 'xyz', or 'pos'.
- Output is ASCII PLY by default; use --binary for binary little-endian.
"""
from __future__ import annotations
import argparse
import os
import sys
import struct
from typing import Optional

import numpy as np
import torch


def _extract_points(obj) -> np.ndarray:
    """Extract Nx3 float32 numpy array from possible .pt content shapes.

    Accepts:
    - torch.Tensor
    - numpy.ndarray
    - dict-like with keys in {'points','xyz','pos'} pointing to any of the above
    """
    if isinstance(obj, dict):
        for key in ("points", "xyz", "pos"):
            if key in obj:
                return _extract_points(obj[key])
        raise ValueError("Unsupported dict structure in .pt: expected keys 'points'/'xyz'/'pos'.")

    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        raise TypeError(f"Unsupported .pt content type: {type(obj)}")

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected array of shape (N, >=3), got {arr.shape}.")

    pts = arr[:, :3].astype(np.float32, copy=False)

    # Filter invalid rows (NaN/Inf)
    mask = np.isfinite(pts).all(axis=1)
    if not mask.all():
        pts = pts[mask]
    return pts


def write_ply_ascii(points: np.ndarray, out_path: str) -> None:
    n = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        # Write lines: x y z
        for x, y, z in points:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")


def write_ply_binary_little_endian(points: np.ndarray, out_path: str) -> None:
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with open(out_path, "wb") as f:
        f.write(header)
        # pack as little-endian floats
        for x, y, z in points:
            f.write(struct.pack("<fff", float(x), float(y), float(z)))


def convert_pt_to_ply(input_pt: str, output_ply: Optional[str] = None, binary: bool = False) -> str:
    """Convert a .pt file containing a point cloud to a .ply file.

    Args:
        input_pt: Path to .pt file (torch.save format)
        output_ply: Optional output path; if None, replace .pt with .ply
        binary: If True, write binary_little_endian; otherwise ASCII
    Returns:
        The output .ply file path
    """
    if output_ply is None:
        base, _ = os.path.splitext(input_pt)
        output_ply = base + ".ply"

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_ply) or ".", exist_ok=True)

    # Prefer safer loading when available (PyTorch >= 2.4 supports weights_only)
    try:
        data = torch.load(input_pt, map_location="cpu", weights_only=True)
    except TypeError:
        # Fallback for older PyTorch versions
        data = torch.load(input_pt, map_location="cpu")
    points = _extract_points(data)

    if points.size == 0:
        raise ValueError("No valid points to write (after filtering NaN/Inf).")

    if binary:
        write_ply_binary_little_endian(points, output_ply)
    else:
        write_ply_ascii(points, output_ply)

    return output_ply


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a .pt point cloud to .ply")
    parser.add_argument("--input", required=True, help="Path to input .pt file")
    parser.add_argument("--output", default=None, help="Path to output .ply file (default: replace .pt suffix)")
    parser.add_argument("--binary", action="store_true", help="Write binary little-endian PLY instead of ASCII")
    args = parser.parse_args(argv)

    out = convert_pt_to_ply(args.input, args.output, args.binary)
    print(f"Wrote PLY: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

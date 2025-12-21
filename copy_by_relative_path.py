#!/usr/bin/env python3
"""Copy files from an input directory to an output directory, preserving relative paths.

Example:
  python copy_by_relative_path.py \
    --input data/condition/yuehai_building_and_ground_combinations \
    --output /tmp/filtered \
    --ext .ply .npz

Notes:
- Relative path is computed against --input.
- By default, existing files are skipped; use --overwrite to replace.
- Extensions are matched case-insensitively; you can pass with or without leading dot.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CopyStats:
    scanned: int = 0
    matched: int = 0
    copied: int = 0
    skipped_exists: int = 0
    failed: int = 0


def _normalize_exts(exts: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for ext in exts:
        e = ext.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        normalized.add(e.lower())
    return normalized


def _is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def copy_preserve_relative(
    input_dir: Path,
    output_dir: Path,
    exts: set[str],
    overwrite: bool,
    follow_symlinks: bool,
    include_hidden: bool,
    dry_run: bool,
) -> CopyStats:
    stats = CopyStats()

    for root, dirs, files in os.walk(input_dir, followlinks=follow_symlinks):
        root_path = Path(root)

        if not include_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

        for filename in files:
            stats.scanned += 1
            src = root_path / filename

            if not include_hidden:
                try:
                    rel_check = src.relative_to(input_dir)
                except ValueError:
                    rel_check = None
                if rel_check is not None and _is_hidden_path(rel_check):
                    continue

            if exts and src.suffix.lower() not in exts:
                continue

            stats.matched += 1

            rel = src.relative_to(input_dir)
            dst = output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists() and not overwrite:
                stats.skipped_exists += 1
                continue

            try:
                if not dry_run:
                    shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
                stats.copied += 1
            except Exception:
                stats.failed += 1

    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy selected file types while preserving relative paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input directory")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--ext",
        nargs="*",
        default=[],
        help="Extensions to include, e.g. .ply .npz .yaml (empty means copy all files)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks during walk and copy (default: do not follow)",
    )
    p.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files/dirs (dot-prefixed)",
    )
    p.add_argument("--dry-run", action="store_true", help="Only print stats; do not copy")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"--input is not a directory: {input_dir}")

    if input_dir == output_dir:
        raise SystemExit("--output must be different from --input")

    exts = _normalize_exts(args.ext)

    stats = copy_preserve_relative(
        input_dir=input_dir,
        output_dir=output_dir,
        exts=exts,
        overwrite=bool(args.overwrite),
        follow_symlinks=bool(args.follow_symlinks),
        include_hidden=bool(args.include_hidden),
        dry_run=bool(args.dry_run),
    )

    ext_text = ", ".join(sorted(exts)) if exts else "(all)"
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Exts  : {ext_text}")
    print(
        "Stats : "
        f"scanned={stats.scanned} matched={stats.matched} "
        f"copied={stats.copied} skipped_exists={stats.skipped_exists} failed={stats.failed}"
    )

    if stats.failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

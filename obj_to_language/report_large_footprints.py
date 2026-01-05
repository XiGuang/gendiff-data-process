import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan rotated YAML files and report ones whose footprint coordinates exceed a given"
            " absolute threshold on the X/Z axes."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing rotated YAML files")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Maximum allowed absolute coordinate value in meters (evaluated on X/Z)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show at most this many offending files when printing details",
    )
    return parser.parse_args()


def iter_yaml_files(root: Path) -> Iterable[Path]:
    for yaml_path in sorted(root.rglob("*.yaml")):
        if yaml_path.is_file():
            yield yaml_path


def load_yaml_entries(yaml_path: Path) -> List[Dict]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"Unexpected YAML content in {yaml_path}")
    return data


def find_worst_coordinate(entries: List[Dict]) -> Tuple[float, int]:
    max_abs = 0.0
    worst_idx = -1
    for idx, entry in enumerate(entries):
        if "footprint" in entry:
            contour = entry["footprint"]
        elif "bottom_contour" in entry:
            contour = entry["bottom_contour"]
        else:
            continue
        for point in contour:
            if len(point) < 3:
                raise ValueError(f"Malformed point in entry {idx}")
            x, _, z = map(float, point[:3])
            current = max(abs(x), abs(z))
            if current > max_abs:
                max_abs = current
                worst_idx = idx
    return max_abs, worst_idx


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    threshold = args.threshold
    offenders: List[Tuple[Path, float, int]] = []
    yaml_paths = list(iter_yaml_files(input_dir))

    for yaml_path in yaml_paths:
        try:
            entries = load_yaml_entries(yaml_path)
            max_abs, entry_idx = find_worst_coordinate(entries)
        except Exception as exc:
            print(f"[ERROR] Failed to inspect {yaml_path}: {exc}")
            continue
        if max_abs > threshold:
            offenders.append((yaml_path, max_abs, entry_idx))

    offenders.sort(key=lambda item: item[1], reverse=True)

    print(f"Checked {len(yaml_paths)} YAML file(s) under {input_dir}")
    print(f"Found {len(offenders)} file(s) exceeding threshold {threshold}.")

    for yaml_path, max_abs, entry_idx in offenders[: args.top]:
        print(f"- {yaml_path}: max |x/z| = {max_abs:.4f} (entry #{entry_idx})")

    if len(offenders) > args.top:
        print(f"... skipped {len(offenders) - args.top} additional offending file(s)")


if __name__ == "__main__":
    main()

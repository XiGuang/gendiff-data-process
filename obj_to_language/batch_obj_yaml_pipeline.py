import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import yaml

from obj_to_language.obj_to_yaml_new import extract_bottom_contour, split_with_merge
from obj_to_language.yaml_to_obj_new import extrude_polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize OBJ meshes based on per-folder data.yaml, convert them to "
            "YAML descriptions, and reconstruct OBJ files from those YAMLs."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Root directory containing folders with obj + data.yaml")
    parser.add_argument("output_dir", type=Path, help="Root directory to mirror outputs into")
    return parser.parse_args()


def load_cube_meta(data_yaml: Path) -> Tuple[np.ndarray, float]:
    with data_yaml.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    position = meta.get("position")
    size = meta.get("size")

    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"Invalid position in {data_yaml}")
    if size is None:
        raise ValueError(f"Missing size in {data_yaml}")

    center = np.asarray(position, dtype=float)
    size_val = float(size)
    if size_val <= 0:
        raise ValueError(f"Size must be positive in {data_yaml}")
    return center, size_val


def normalize_mesh(mesh: trimesh.Trimesh, center: np.ndarray, size: float) -> None:
    mesh.apply_translation(-center)
    mesh.apply_scale(2.0 / size)


def mesh_to_yaml_entries(mesh: trimesh.Trimesh) -> List[dict]:
    entries = []
    for i, submesh in enumerate(split_with_merge(mesh.copy())):
        contour, y_min = extract_bottom_contour(submesh)
        if contour is None:
            continue
        height = float(submesh.vertices[:, 1].max() - y_min)
        entries.append({
            "mesh_id": i,
            "height": height,
            "footprint": contour,
        })
    return entries


def yaml_entries_to_mesh(entries: List[dict]) -> Optional[trimesh.Trimesh]:
    meshes = []
    for i, entry in enumerate(entries):
        contour = entry["footprint"]
        height = entry["height"]
        part = extrude_polygon(contour, height)
        part.metadata["name"] = f"mesh_{i}"
        meshes.append(part)
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def process_obj_file(obj_path: Path, center: np.ndarray, size: float, output_dir: Path) -> None:
    mesh = trimesh.load(obj_path, force="mesh")
    normalize_mesh(mesh, center, size)

    yaml_entries = mesh_to_yaml_entries(mesh)
    if not yaml_entries:
        print(f"[WARN] No valid components found in {obj_path}")
        return

    yaml_path = output_dir / f"{obj_path.stem}.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_entries, f, sort_keys=False)

    reconstructed = yaml_entries_to_mesh(yaml_entries)
    if reconstructed is None:
        print(f"[WARN] Reconstruction skipped for {obj_path} (no meshes)")
        return

    obj_out = output_dir / f"{obj_path.stem}_from_yaml.obj"
    reconstructed.export(obj_out)

    print(f"Processed {obj_path} -> {yaml_path} & {obj_out}")


def iter_target_directories(input_dir: Path):
    for root, _dirs, files in os.walk(input_dir):
        if "data.yaml" not in files:
            continue
        obj_files = [Path(root) / f for f in files if f.lower().endswith(".obj")]
        yield Path(root), obj_files


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    total_objs = 0
    for folder, obj_files in iter_target_directories(input_dir):
        rel = folder.relative_to(input_dir)
        out_dir = output_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        center, size = load_cube_meta(folder / "data.yaml")

        if not obj_files:
            placeholder = out_dir / f"bs_{folder.name}.yaml"
            with placeholder.open("w", encoding="utf-8") as f:
                yaml.safe_dump([], f, sort_keys=False)
            print(f"[INFO] No OBJ files found in {folder}, wrote {placeholder}")
            continue

        for obj_path in obj_files:
            try:
                process_obj_file(obj_path, center, size, out_dir)
                total_objs += 1
            except Exception as exc:
                print(f"[ERROR] Failed to process {obj_path}: {exc}")

    print(f"Done. Processed {total_objs} obj file(s).")


if __name__ == "__main__":
    main()

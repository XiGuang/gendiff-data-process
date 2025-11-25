import os
import sys
import argparse
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import trimesh

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - provide friendly message if tqdm missing
    tqdm = None  # type: ignore


def _obj_has_vertices(path: str) -> bool:
    """Quickly check if an OBJ file contains any vertex lines (starts with 'v ').

    Returns True if there's at least one vertex; False if none found.
    If the file can't be read, return True (treat as non-empty to be safe).
    """
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                # Exact vertex line begins with 'v ' (not 'vn'/'vt')
                if line.startswith("v "):
                    return True
        return False
    except Exception:
        return True


def _load_mesh_safely(mesh_path: str) -> Optional[trimesh.Trimesh]:
    """Load a mesh file and return a single Trimesh instance.

    Handles cases where trimesh.load might return a Scene by concatenating geometries.
    Returns None if loading fails or no geometry is present.
    """
    try:
        loaded = trimesh.load(mesh_path, force=None)
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        # Scene: merge geometries into a single mesh
        if hasattr(loaded, "geometry") and loaded.geometry:
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return None
            return trimesh.util.concatenate(geoms)
        return None
    except Exception:
        return None


def merge_building_meshes(building_dir: str, output_path: str, verbose: bool = False) -> Tuple[str, str]:
    """Merge all .obj meshes inside one building directory into a single OBJ.

    Returns a tuple (status, message):
    - status in {"merged", "skipped_no_obj", "skipped_exists", "error"}
    - message is a short description
    """
    # Existing output handling
    if os.path.exists(output_path):
        if _obj_has_vertices(output_path):
            return "skipped_exists", f"Output exists and not empty: {output_path}"
        # Otherwise overwrite the small/empty file
        if verbose:
            print(f"Overwriting empty output: {output_path}")

    try:
        mesh_files = sorted([f for f in os.listdir(building_dir) if f.lower().endswith(".obj")])
    except FileNotFoundError:
        return "error", f"Building dir not found: {building_dir}"

    if not mesh_files:
        return "skipped_no_obj", f"No .obj files in {building_dir}"

    meshes: List[trimesh.Trimesh] = []
    for mesh_file in mesh_files:
        mesh_path = os.path.join(building_dir, mesh_file)
        m = _load_mesh_safely(mesh_path)
        if m is None:
            if verbose:
                print(f"Warning: failed to load mesh: {mesh_path}")
            continue
        meshes.append(m)

    if not meshes:
        return "error", f"Failed to load any meshes in {building_dir}"

    try:
        merged = trimesh.util.concatenate(meshes)
        # Drop material to avoid MTL references
        try:
            if hasattr(merged, "visual") and hasattr(merged.visual, "material"):
                merged.visual.material = None  # type: ignore[attr-defined]
        except Exception:
            pass

        # Ensure parent dir exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.export(output_path)
        if verbose:
            print(f"Merged mesh saved to {output_path}")
        return "merged", f"Saved: {output_path}"
    except Exception as e:
        return "error", f"Merge/export failed: {e}"


def _discover_building_jobs(input_directory: str, output_name: str) -> List[Tuple[str, str, str]]:
    """Find subdirectories under input_directory that contain a 'building' folder.

    Returns list of tuples: (subdir_name, building_dir, output_path)
    """
    jobs: List[Tuple[str, str, str]] = []
    for subdir in os.listdir(input_directory):
        building_dir = os.path.join(input_directory, subdir, "building")
        if not os.path.isdir(building_dir):
            continue
        output_mesh_path = os.path.join(input_directory, subdir, output_name)
        jobs.append((subdir, building_dir, output_mesh_path))
    return jobs


def _worker(args: Tuple[str, str, str, bool]) -> Tuple[str, str, str]:
    """Worker for multiprocessing: returns (subdir, status, message)."""
    subdir, building_dir, output_path, verbose = args
    status, msg = merge_building_meshes(building_dir, output_path, verbose=verbose)
    return subdir, status, msg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge building meshes into single OBJ per subdir (multiprocess)")
    p.add_argument("--input-dir", default="yingrenshi_fixed", help="Root directory containing subdirs with building/ folders")
    p.add_argument("--output-name", default="building_normalized.obj", help="Output OBJ file name placed in each subdir")
    p.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1), help="Number of parallel workers")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if tqdm is None:
        print("tqdm is not installed. Install it to see progress bar: pip install tqdm", file=sys.stderr)

    if not os.path.isdir(args.input_dir):
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    jobs = _discover_building_jobs(args.input_dir, args.output_name)
    total = len(jobs)
    if total == 0:
        print("No building directories found.")
        return 0

    # Build worker arguments
    tasks = [(subdir, bdir, out, args.verbose) for (subdir, bdir, out) in jobs]

    merged = skipped_exists = skipped_no_obj = errors = 0

    if args.workers <= 1:
        iterator = tasks
        bar = tqdm(total=total, desc="Merging buildings", dynamic_ncols=True) if tqdm else None
        for t in iterator:
            subdir, status, msg = _worker(t)
            if bar:
                bar.update(1)
            if status == "merged":
                merged += 1
            elif status == "skipped_exists":
                skipped_exists += 1
            elif status == "skipped_no_obj":
                skipped_no_obj += 1
            else:
                errors += 1
            if args.verbose:
                print(f"[{subdir}] {status}: {msg}")
        if bar:
            bar.close()
    else:
        with Pool(processes=args.workers) as pool:
            imap_it = pool.imap_unordered(_worker, tasks)
            if tqdm:
                with tqdm(total=total, desc="Merging buildings", dynamic_ncols=True) as bar:
                    for subdir, status, msg in imap_it:
                        bar.update(1)
                        if status == "merged":
                            merged += 1
                        elif status == "skipped_exists":
                            skipped_exists += 1
                        elif status == "skipped_no_obj":
                            skipped_no_obj += 1
                        else:
                            errors += 1
                        if args.verbose:
                            print(f"[{subdir}] {status}: {msg}")
            else:
                for subdir, status, msg in imap_it:
                    if status == "merged":
                        merged += 1
                    elif status == "skipped_exists":
                        skipped_exists += 1
                    elif status == "skipped_no_obj":
                        skipped_no_obj += 1
                    else:
                        errors += 1
                    if args.verbose:
                        print(f"[{subdir}] {status}: {msg}")

    print(
        f"Summary: merged={merged}, skipped_exists={skipped_exists}, "
        f"skipped_no_obj={skipped_no_obj}, errors={errors}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import bpy
import yaml
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EPS = 1e-4


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Cut a textured dense mesh using generated construction sequence stages."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Input dense mesh path, typically OBJ with MTL and textures.",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Sequence directory produced by generate_construction_sequence.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory. Stage subdirectories will mirror the sequence layout.",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        type=int,
        default=None,
        help="Optional 1-based stage indices to export.",
    )
    parser.add_argument(
        "--export-format",
        choices=("glb", "obj"),
        default="glb",
        help="Export format for each stage output.",
    )
    parser.add_argument(
        "--cap-mode",
        choices=("solid",),
        default="solid",
        help="How to shade newly created cut faces.",
    )
    parser.add_argument(
        "--cap-color",
        nargs=4,
        type=float,
        default=(0.72, 0.72, 0.72, 1.0),
        help="RGBA color for cut faces when --cap-mode solid is used.",
    )
    parser.add_argument(
        "--cap-roughness",
        type=float,
        default=0.9,
        help="Roughness for the cap material.",
    )
    parser.add_argument(
        "--plane-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for detecting faces on cut planes in world coordinates.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=2.0,
        help="Extra padding added to the boolean cutter bounds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage outputs.",
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_obj(mesh_path: Path) -> list[bpy.types.Object]:
    bpy.ops.wm.obj_import(filepath=str(mesh_path), use_split_objects=False)
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    restore_matrix = Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    for obj in objects:
        obj.matrix_world = restore_matrix @ obj.matrix_world
    bpy.context.view_layer.update()
    return objects


def merge_mesh_objects(objects: list[bpy.types.Object], name: str) -> bpy.types.Object:
    if not objects:
        raise ValueError("No mesh objects were imported")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = name
    return merged


def object_world_bounds(obj: bpy.types.Object) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = [min(corner[i] for corner in corners) for i in range(3)]
    maxs = [max(corner[i] for corner in corners) for i in range(3)]
    return mins, maxs


def discover_stage_dirs(sequence_dir: Path, selected_stages: set[int] | None) -> list[tuple[int, Path]]:
    discovered: list[tuple[int, Path]] = []
    for path in sorted(sequence_dir.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("stage_"):
            continue
        meta_path = path / "construction_meta.yaml"
        if not meta_path.exists():
            continue
        try:
            stage_index = int(path.name.split("_", 2)[1])
        except (IndexError, ValueError):
            continue
        if selected_stages and stage_index not in selected_stages:
            continue
        discovered.append((stage_index, path))
    return discovered


def load_stage_metadata(stage_dir: Path) -> dict:
    meta_path = stage_dir / "construction_meta.yaml"
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid stage metadata in {meta_path}")
    return payload


def load_flat_entries(yaml_path: Path) -> list[dict]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {yaml_path}")
    entries: list[dict] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Entry {index} is not a mapping in {yaml_path}")
        footprint = raw.get("footprint")
        if not isinstance(footprint, list):
            raise ValueError(f"Entry {index} has no footprint list in {yaml_path}")
        entries.append(
            {
                "proxy_id": int(raw.get("proxy_id", index)),
                "source_proxy_id": int(raw.get("source_proxy_id", raw.get("proxy_id", index))),
                "level_index": int(raw.get("level_index", 0)),
                "min_height": float(raw["min_height"]),
                "max_height": float(raw["max_height"]),
                "footprint": [[float(point[0]), float(point[1])] for point in footprint],
            }
        )
    return entries


def discover_stage_yaml(stage_dir: Path) -> Path:
    candidates = [
        path
        for path in sorted(stage_dir.glob("*.yaml"))
        if path.name not in {"construction_meta.yaml", "cap_meta.yaml"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No stage YAML found in {stage_dir}")
    return candidates[0]


def active_cut_planes(stage_meta: dict) -> list[dict]:
    planes: list[dict] = []
    mode = str(stage_meta.get("mode", "")).lower()

    if mode in {"vertical", "hybrid"}:
        ratio_key = "vertical_ratio" if mode == "hybrid" else "ratio"
        ratio = float(stage_meta.get(ratio_key, 1.0))
        if ratio < 1.0 - EPS and "target_height" in stage_meta:
            planes.append(
                {
                    "axis": 1,
                    "value": float(stage_meta["target_height"]),
                    "name": "top",
                    "uv_axes": (0, 2),
                }
            )

    if mode in {"footprint", "hybrid"}:
        ratio_key = "footprint_ratio" if mode == "hybrid" else "ratio"
        ratio = float(stage_meta.get(ratio_key, 1.0))
        axis = str(stage_meta.get("axis", "x"))
        keep_side = str(stage_meta.get("keep_side", "min"))
        clip_bounds = stage_meta.get("clip_bounds") or {}
        if ratio < 1.0 - EPS:
            if axis == "x":
                value = float(clip_bounds["max_x"] if keep_side == "min" else clip_bounds["min_x"])
                planes.append(
                    {
                        "axis": 0,
                        "value": value,
                        "name": f"x_{keep_side}",
                        "uv_axes": (2, 1),
                    }
                )
            else:
                value = float(clip_bounds["max_z"] if keep_side == "min" else clip_bounds["min_z"])
                planes.append(
                    {
                        "axis": 2,
                        "value": value,
                        "name": f"z_{keep_side}",
                        "uv_axes": (0, 1),
                    }
                )
    return planes


def cutter_bounds(mesh_obj: bpy.types.Object, stage_meta: dict, padding: float) -> tuple[list[float], list[float]]:
    mesh_min, mesh_max = object_world_bounds(mesh_obj)
    min_corner = [mesh_min[0] - padding, mesh_min[1] - padding, mesh_min[2] - padding]
    max_corner = [mesh_max[0] + padding, mesh_max[1] + padding, mesh_max[2] + padding]
    mode = str(stage_meta.get("mode", "")).lower()

    if mode in {"vertical", "hybrid"} and "target_height" in stage_meta:
        max_corner[1] = min(max_corner[1], float(stage_meta["target_height"]) + padding)

    if mode in {"footprint", "hybrid"}:
        clip_bounds = stage_meta.get("clip_bounds") or {}
        min_corner[0] = max(min_corner[0], float(clip_bounds.get("min_x", min_corner[0])) - padding)
        max_corner[0] = min(max_corner[0], float(clip_bounds.get("max_x", max_corner[0])) + padding)
        min_corner[2] = max(min_corner[2], float(clip_bounds.get("min_z", min_corner[2])) - padding)
        max_corner[2] = min(max_corner[2], float(clip_bounds.get("max_z", max_corner[2])) + padding)

    return min_corner, max_corner


def create_cutter(min_corner: list[float], max_corner: list[float], name: str) -> bpy.types.Object:
    center = tuple((low + high) * 0.5 for low, high in zip(min_corner, max_corner))
    size = max(max_corner[i] - min_corner[i] for i in range(3))
    bpy.ops.mesh.primitive_cube_add(size=size, location=center)
    cube = bpy.context.active_object
    cube.name = name
    dims = [max(max_corner[i] - min_corner[i], EPS) for i in range(3)]
    cube.scale = (dims[0] / size, dims[1] / size, dims[2] / size)
    bpy.context.view_layer.update()
    return cube


def boolean_intersect(target_obj: bpy.types.Object, cutter_obj: bpy.types.Object) -> None:
    modifier = target_obj.modifiers.new(name="ConstructionBoolean", type="BOOLEAN")
    modifier.operation = "INTERSECT"
    modifier.solver = "EXACT"
    modifier.double_threshold = 1e-6
    modifier.object = cutter_obj
    bpy.context.view_layer.objects.active = target_obj
    target_obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def ensure_cap_material(obj: bpy.types.Object, color: tuple[float, float, float, float], roughness: float) -> int:
    material_name = "ConstructionCap"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = float(roughness)

    for index, slot in enumerate(obj.data.materials):
        if slot == material:
            return index
    obj.data.materials.append(material)
    return len(obj.data.materials) - 1


def get_cap_material(color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    material_name = "ConstructionCap"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = float(roughness)
    return material


def assign_cap_material(
    obj: bpy.types.Object,
    planes: list[dict],
    *,
    cap_material_index: int,
    tolerance: float,
) -> dict:
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        mesh.uv_layers.active = uv_layer

    cap_faces = 0
    per_plane_counts: dict[str, int] = {}
    world_matrix = obj.matrix_world

    for face in mesh.polygons:
        matched_plane = None
        world_positions = [world_matrix @ mesh.vertices[index].co for index in face.vertices]
        for plane in planes:
            axis = int(plane["axis"])
            value = float(plane["value"])
            if all(abs(position[axis] - value) <= tolerance for position in world_positions):
                matched_plane = plane
                break
        if matched_plane is None:
            continue

        face.material_index = cap_material_index
        uv_u_axis, uv_v_axis = matched_plane["uv_axes"]
        for loop_index in face.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            world_pos = world_matrix @ mesh.vertices[vertex_index].co
            uv_layer.data[loop_index].uv = (world_pos[uv_u_axis], world_pos[uv_v_axis])
        cap_faces += 1
        per_plane_counts[matched_plane["name"]] = per_plane_counts.get(matched_plane["name"], 0) + 1

    mesh.update()
    return {"cap_face_count": cap_faces, "per_plane_face_count": per_plane_counts}


def export_stage_mesh(obj: bpy.types.Object, output_path: Path, export_format: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if export_format == "glb":
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            use_selection=True,
            export_materials="EXPORT",
        )
        return

    bpy.ops.wm.obj_export(
        filepath=str(output_path),
        export_selected_objects=True,
        export_materials=True,
    )


def export_stage_objects(objects: list[bpy.types.Object], output_path: Path, export_format: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    if export_format == "glb":
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            use_selection=True,
            export_materials="EXPORT",
        )
        return

    bpy.ops.wm.obj_export(
        filepath=str(output_path),
        export_selected_objects=True,
        export_materials=True,
    )


def build_cap_faces(stage_entries: list[dict], source_entries: dict[int, dict], stage_meta: dict) -> tuple[list[list[tuple[float, float, float]]], dict]:
    faces: list[list[tuple[float, float, float]]] = []
    summary = {"top_face_count": 0, "side_face_count": 0}
    mode = str(stage_meta.get("mode", "")).lower()

    if mode in {"vertical", "hybrid"}:
        ratio_key = "vertical_ratio" if mode == "hybrid" else "ratio"
        ratio = float(stage_meta.get(ratio_key, 1.0))
        target_height = float(stage_meta.get("target_height", 0.0))
        if ratio < 1.0 - EPS:
            for entry in stage_entries:
                source = source_entries.get(entry["source_proxy_id"])
                if source is None:
                    continue
                if source["max_height"] <= target_height + EPS:
                    continue
                if abs(entry["max_height"] - target_height) > EPS:
                    continue
                top_face = [
                    (float(point[0]), target_height, float(point[1]))
                    for point in entry["footprint"]
                ]
                if len(top_face) >= 3:
                    faces.append(top_face)
                    summary["top_face_count"] += 1

    if mode in {"footprint", "hybrid"}:
        ratio_key = "footprint_ratio" if mode == "hybrid" else "ratio"
        ratio = float(stage_meta.get(ratio_key, 1.0))
        axis = str(stage_meta.get("axis", "x"))
        keep_side = str(stage_meta.get("keep_side", "min"))
        clip_bounds = stage_meta.get("clip_bounds") or {}
        if ratio < 1.0 - EPS:
            plane_value = float(
                clip_bounds["max_x"] if axis == "x" and keep_side == "min"
                else clip_bounds["min_x"] if axis == "x"
                else clip_bounds["max_z"] if keep_side == "min"
                else clip_bounds["min_z"]
            )
            plane_axis = 0 if axis == "x" else 1
            for entry in stage_entries:
                points = entry["footprint"]
                for idx, start in enumerate(points):
                    end = points[(idx + 1) % len(points)]
                    if (
                        abs(start[plane_axis] - plane_value) <= EPS
                        and abs(end[plane_axis] - plane_value) <= EPS
                    ):
                        if axis == "x":
                            quad = [
                                (plane_value, entry["min_height"], start[1]),
                                (plane_value, entry["max_height"], start[1]),
                                (plane_value, entry["max_height"], end[1]),
                                (plane_value, entry["min_height"], end[1]),
                            ]
                        else:
                            quad = [
                                (start[0], entry["min_height"], plane_value),
                                (start[0], entry["max_height"], plane_value),
                                (end[0], entry["max_height"], plane_value),
                                (end[0], entry["min_height"], plane_value),
                            ]
                        faces.append(quad)
                        summary["side_face_count"] += 1

    return faces, summary


def create_cap_object(
    name: str,
    faces: list[list[tuple[float, float, float]]],
    material: bpy.types.Material,
) -> bpy.types.Object | None:
    if not faces:
        return None

    vertices: list[tuple[float, float, float]] = []
    face_indices: list[list[int]] = []
    for face in faces:
        start = len(vertices)
        vertices.extend(face)
        face_indices.append(list(range(start, start + len(face))))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], face_indices)
    mesh.update()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def write_cap_meta(output_dir: Path, payload: dict) -> None:
    with (output_dir / "cap_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def stage_output_path(output_dir: Path, mesh_path: Path, export_format: str) -> Path:
    if export_format == "glb":
        return output_dir / f"{mesh_path.stem}.glb"
    return output_dir / mesh_path.name


def process_stage(
    mesh_path: Path,
    stage_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_meta = load_stage_metadata(stage_dir)
    stage_yaml = discover_stage_yaml(stage_dir)
    stage_entries = load_flat_entries(stage_yaml)
    source_yaml = Path(str(stage_meta["source_yaml"]))
    source_entries = {entry["proxy_id"]: entry for entry in load_flat_entries(source_yaml)}
    out_path = stage_output_path(output_dir, mesh_path, args.export_format)
    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] {stage_dir.name}: output exists at {out_path}")
        return

    clear_scene()
    imported = import_obj(mesh_path)
    dense_obj = merge_mesh_objects(imported, mesh_path.stem)
    min_corner, max_corner = cutter_bounds(dense_obj, stage_meta, args.padding)
    cutter = create_cutter(min_corner, max_corner, f"{stage_dir.name}_cutter")

    boolean_intersect(dense_obj, cutter)
    bpy.data.objects.remove(cutter, do_unlink=True)

    planes = active_cut_planes(stage_meta)
    cap_faces, cap_breakdown = build_cap_faces(stage_entries, source_entries, stage_meta)
    cap_obj = None
    if planes and args.cap_mode == "solid":
        cap_material = get_cap_material(
            tuple(float(channel) for channel in args.cap_color),
            args.cap_roughness,
        )
        cap_obj = create_cap_object(f"{stage_dir.name}_cap", cap_faces, cap_material)

    export_objects = [dense_obj] + ([cap_obj] if cap_obj is not None else [])
    export_stage_objects(export_objects, out_path, args.export_format)
    shutil.copy2(stage_dir / "construction_meta.yaml", output_dir / "construction_meta.yaml")
    write_cap_meta(
        output_dir,
        {
            "source_mesh": str(mesh_path),
            "stage_dir": str(stage_dir),
            "export_path": str(out_path),
            "export_format": args.export_format,
            "cap_mode": args.cap_mode,
            "cap_material_name": "ConstructionCap" if planes else None,
            "cap_color": [float(channel) for channel in args.cap_color],
            "cut_planes": planes,
            "cap_face_count": len(cap_faces),
            **cap_breakdown,
        },
    )
    print(
        f"[OK] {stage_dir.name}: exported {out_path.name} "
        f"(cap_faces={len(cap_faces)})"
    )


def main() -> int:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    sequence_dir = args.sequence_dir.resolve()
    output_root = args.output.resolve()

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory does not exist: {sequence_dir}")

    selected_stages = set(args.stages) if args.stages else None
    stage_dirs = discover_stage_dirs(sequence_dir, selected_stages)
    if not stage_dirs:
        print("No matching stage directories found.")
        return 1

    for stage_index, stage_dir in stage_dirs:
        process_stage(
            mesh_path,
            stage_dir,
            output_root / stage_dir.name,
            args,
        )

    print(f"Finished {len(stage_dirs)} stage(s) under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

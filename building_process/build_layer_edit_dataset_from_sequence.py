import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
import yaml
from shapely.geometry import MultiPolygon, Polygon


EDIT_TOKEN_NAMES: Dict[int, str] = {
    0: "PAD",
    1: "BOS",
    2: "EOS",
    3: "LAYER_START",
    4: "LAYER_END",
    5: "ADD_POINT",
    6: "DELETE_POINT",
    7: "MOVE_POINT",
    8: "MIN_HEIGHT",
    9: "MAX_HEIGHT",
    10: "ADD_MIN_HEIGHT",
    11: "ADD_MAX_HEIGHT",
}


class LayerToken:
    def __init__(self, token_type: int, values: Tuple[float, float] = (0.0, 0.0)) -> None:
        self.token_type = int(token_type)
        self.values = (float(values[0]), float(values[1]))


def normalize_layers_data(data: Sequence[dict]) -> List[dict]:
    normalized: List[dict] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        proxy_id = entry.get("proxy_id", idx)
        source_proxy_id = entry.get("source_proxy_id", proxy_id)
        level_index = entry.get("level_index", idx)
        footprint = entry.get("footprint") or []
        normalized_footprint: List[List[float]] = []
        for point in footprint:
            try:
                coords = np.asarray(point, dtype=float).reshape(-1).tolist()
            except (TypeError, ValueError):
                continue
            if len(coords) < 2:
                continue
            normalized_footprint.append([float(coords[0]), float(coords[1])])
        normalized.append(
            {
                "proxy_id": int(proxy_id if proxy_id is not None else idx),
                "source_proxy_id": int(source_proxy_id if source_proxy_id is not None else proxy_id if proxy_id is not None else idx),
                "level_index": int(level_index if level_index is not None else idx),
                "min_height": float(entry.get("min_height", 0.0)),
                "max_height": float(entry.get("max_height", 0.0)),
                "footprint": normalized_footprint,
            }
        )
    return normalized


def edit_sequence_to_yaml(token_types: Sequence[int], token_values: Sequence[Sequence[float]]) -> str:
    serializable: List[dict] = []
    payload_dims = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 2,
        6: 0,
        7: 2,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
    }
    for token_type, values in zip(token_types, token_values):
        t = int(token_type)
        if t == 0:
            continue
        entry = {"type": EDIT_TOKEN_NAMES.get(t, f"UNKNOWN_{t}")}
        dims = payload_dims[t]
        if dims == 1:
            entry["value"] = float(values[0])
        elif dims == 2:
            entry["value"] = [float(values[0]), float(values[1])]
        serializable.append(entry)
    return yaml.safe_dump(
        serializable,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def match_target_layers(history_layers: Sequence[dict], target_layers: Sequence[dict]) -> List[Optional[int]]:
    history_key_to_idx: Dict[int, int] = {}
    for idx, layer in enumerate(history_layers):
        history_key_to_idx.setdefault(int(layer.get("proxy_id", idx)), idx)
        history_key_to_idx.setdefault(int(layer.get("source_proxy_id", layer.get("proxy_id", idx))), idx)

    matches: List[Optional[int]] = []
    min_allowed_idx = 0
    for target_idx, layer in enumerate(target_layers):
        candidates: List[int] = []
        for key_name in ("source_proxy_id", "proxy_id"):
            key = layer.get(key_name)
            if key is None:
                continue
            matched = history_key_to_idx.get(int(key))
            if matched is not None and matched >= min_allowed_idx:
                candidates.append(matched)
        if target_idx >= min_allowed_idx and target_idx < len(history_layers):
            candidates.append(target_idx)
        if candidates:
            choice = min(candidates)
            matches.append(choice)
            min_allowed_idx = choice + 1
        else:
            matches.append(None)
    return matches


def align_points(source_points: Sequence[Sequence[float]], target_points: Sequence[Sequence[float]]) -> List[LayerToken]:
    src = np.asarray(source_points, dtype=float).reshape(-1, 2) if source_points else np.zeros((0, 2), dtype=float)
    tgt = np.asarray(target_points, dtype=float).reshape(-1, 2) if target_points else np.zeros((0, 2), dtype=float)
    n_src, n_tgt = src.shape[0], tgt.shape[0]
    if n_src == 0 and n_tgt == 0:
        return []

    stacked = np.concatenate([src, tgt], axis=0) if (n_src > 0 and n_tgt > 0) else (src if n_src > 0 else tgt)
    diag = stacked.max(axis=0) - stacked.min(axis=0)
    scale = max(float(np.linalg.norm(diag)), 1.0)

    add_penalty = 1.0
    delete_penalty = 1.0
    dp = np.zeros((n_src + 1, n_tgt + 1), dtype=float)
    for i in range(n_src - 1, -1, -1):
        dp[i, n_tgt] = dp[i + 1, n_tgt] + delete_penalty
    for j in range(n_tgt - 1, -1, -1):
        dp[n_src, j] = dp[n_src, j + 1] + add_penalty

    for i in range(n_src - 1, -1, -1):
        for j in range(n_tgt - 1, -1, -1):
            move_cost = float(np.linalg.norm(src[i] - tgt[j])) / scale + dp[i + 1, j + 1]
            delete_cost = delete_penalty + dp[i + 1, j]
            add_cost = add_penalty + dp[i, j + 1]
            dp[i, j] = min(move_cost, delete_cost, add_cost)

    i = 0
    j = 0
    ops: List[LayerToken] = []
    eps = 1e-6
    while i < n_src or j < n_tgt:
        if i < n_src and j < n_tgt:
            move_cost = float(np.linalg.norm(src[i] - tgt[j])) / scale + dp[i + 1, j + 1]
            if abs(dp[i, j] - move_cost) <= eps:
                delta = tgt[j] - src[i]
                ops.append(LayerToken(7, (float(delta[0]), float(delta[1]))))
                i += 1
                j += 1
                continue
        if i < n_src:
            delete_cost = delete_penalty + dp[i + 1, j]
            if abs(dp[i, j] - delete_cost) <= eps:
                ops.append(LayerToken(6))
                i += 1
                continue
        if j < n_tgt:
            ops.append(LayerToken(5, (float(tgt[j, 0]), float(tgt[j, 1]))))
            j += 1
    return ops


def encode_existing_layer(history_layer: dict, target_layer: dict) -> List[LayerToken]:
    tokens = [
        LayerToken(3),
        LayerToken(8, (float(target_layer["min_height"]) - float(history_layer["min_height"]), 0.0)),
        LayerToken(9, (float(target_layer["max_height"]) - float(history_layer["max_height"]), 0.0)),
    ]
    tokens.extend(align_points(history_layer["footprint"], target_layer["footprint"]))
    tokens.append(LayerToken(4))
    return tokens


def encode_new_layer(target_layer: dict) -> List[LayerToken]:
    tokens = [
        LayerToken(3),
        LayerToken(10, (float(target_layer["min_height"]), 0.0)),
        LayerToken(11, (float(target_layer["max_height"]), 0.0)),
    ]
    for point in target_layer["footprint"]:
        tokens.append(LayerToken(5, (float(point[0]), float(point[1]))))
    tokens.append(LayerToken(4))
    return tokens


def encode_deleted_layer(history_layer: dict) -> List[LayerToken]:
    tokens = [LayerToken(3), LayerToken(8, (0.0, 0.0)), LayerToken(9, (0.0, 0.0))]
    for _ in history_layer["footprint"]:
        tokens.append(LayerToken(6))
    tokens.append(LayerToken(4))
    return tokens


def encode_edit_sequence(history_layers: Sequence[dict], target_layers: Sequence[dict]) -> Tuple[List[int], List[List[float]]]:
    history = normalize_layers_data(history_layers)
    target = normalize_layers_data(target_layers)
    matches = match_target_layers(history, target)
    tokens: List[LayerToken] = [LayerToken(1)]
    history_cursor = 0

    for target_layer, matched_history_idx in zip(target, matches):
        if matched_history_idx is None:
            tokens.extend(encode_new_layer(target_layer))
            continue
        while history_cursor < matched_history_idx:
            tokens.extend(encode_deleted_layer(history[history_cursor]))
            history_cursor += 1
        tokens.extend(encode_existing_layer(history[matched_history_idx], target_layer))
        history_cursor = matched_history_idx + 1

    while history_cursor < len(history):
        tokens.extend(encode_deleted_layer(history[history_cursor]))
        history_cursor += 1

    tokens.append(LayerToken(2))
    token_types = [token.token_type for token in tokens]
    token_values = [[token.values[0], token.values[1]] for token in tokens]
    return token_types, token_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a layer-edit training dataset from a staged building sequence.")
    parser.add_argument(
        "--sequence-dir",
        type=str,
        default="/mnt/d/projects/GenDiff/test_data/sequence_hybrid_seed2018348373",
        help="Directory containing stage_xx_* subdirectories and sequence_meta.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/d/projects/GenDiff/test_data/layer_edit_sequence_hybrid_seed2018348373_dataset",
        help="Output dataset root. train/val/test.yaml will be written here.",
    )
    parser.add_argument(
        "--pair-mode",
        type=str,
        choices=("consecutive", "all_forward"),
        default="consecutive",
        help="Use only consecutive forward transitions or all i<j forward transitions.",
    )
    parser.add_argument(
        "--condition-point-count",
        type=int,
        default=8192,
        help="Number of surface points to sample for each condition point cloud.",
    )
    parser.add_argument(
        "--change-tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for deciding whether a target layer differs from the history layer.",
    )
    parser.add_argument(
        "--copy-objs",
        action="store_true",
        help="Also copy stage OBJ files into the normalized stage directories.",
    )
    parser.add_argument(
        "--save-condition-ply",
        action="store_true",
        help="Also export each condition point cloud as a .ply file next to the .pt file.",
    )
    return parser.parse_args()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False, allow_unicode=False)


def normalize_layers(path: Path) -> List[dict]:
    data = load_yaml(path) or []
    if not isinstance(data, list):
        raise ValueError(f"Malformed layer yaml: {path}")
    return normalize_layers_data(data)


def normalize_stage_name(stage_name: str) -> str:
    return stage_name.rstrip("/").strip()


def enumerate_pairs(stage_names: Sequence[str], pair_mode: str) -> List[Tuple[str, str]]:
    if pair_mode == "consecutive":
        return list(zip(stage_names[:-1], stage_names[1:]))
    pairs: List[Tuple[str, str]] = []
    for i, src in enumerate(stage_names):
        for dst in stage_names[i + 1 :]:
            pairs.append((src, dst))
    return pairs


def ensure_stage_copy(sequence_dir: Path, output_dir: Path, stage_name: str, *, copy_objs: bool) -> Path:
    source_stage_dir = sequence_dir / stage_name
    stage_out_dir = output_dir / "stages" / stage_name
    stage_out_dir.mkdir(parents=True, exist_ok=True)

    yaml_src = source_stage_dir / "building1.yaml"
    yaml_dst = stage_out_dir / f"bs_{stage_name}_r0.yaml"
    shutil.copy2(yaml_src, yaml_dst)

    meta_src = source_stage_dir / "construction_meta.yaml"
    if meta_src.exists():
        shutil.copy2(meta_src, stage_out_dir / "construction_meta.yaml")

    if copy_objs:
        obj_src = source_stage_dir / "building1.obj"
        if obj_src.exists():
            shutil.copy2(obj_src, stage_out_dir / "building1.obj")

    return stage_out_dir.resolve()


def layer_changed(history_layer: dict, target_layer: dict, tol: float) -> bool:
    if abs(float(history_layer["min_height"]) - float(target_layer["min_height"])) > tol:
        return True
    if abs(float(history_layer["max_height"]) - float(target_layer["max_height"])) > tol:
        return True
    history_fp = history_layer.get("footprint") or []
    target_fp = target_layer.get("footprint") or []
    if len(history_fp) != len(target_fp):
        return True
    for src_pt, dst_pt in zip(history_fp, target_fp):
        if abs(float(src_pt[0]) - float(dst_pt[0])) > tol or abs(float(src_pt[1]) - float(dst_pt[1])) > tol:
            return True
    return False


def select_changed_target_layers(history_layers: Sequence[dict], target_layers: Sequence[dict], tol: float) -> List[dict]:
    history_by_source = {int(layer.get("source_proxy_id", layer.get("proxy_id", idx))): layer for idx, layer in enumerate(history_layers)}
    changed_layers: List[dict] = []
    for idx, target_layer in enumerate(target_layers):
        source_key = int(target_layer.get("source_proxy_id", target_layer.get("proxy_id", idx)))
        history_layer = history_by_source.get(source_key)
        if history_layer is None or layer_changed(history_layer, target_layer, tol):
            changed_layers.append(target_layer)
    return changed_layers


def extrude_layer_to_mesh(layer: dict) -> trimesh.Trimesh:
    footprint = np.asarray(layer["footprint"], dtype=float).reshape(-1, 2)
    if footprint.shape[0] < 3:
        raise ValueError("Footprint must contain at least three points.")
    polygon = Polygon([(pt[0], pt[1]) for pt in footprint])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda g: g.area)
    if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 1e-8:
        raise ValueError("Degenerate footprint polygon.")
    if not polygon.exterior.is_ccw:
        footprint = footprint[::-1]
        polygon = Polygon([(pt[0], pt[1]) for pt in footprint])
    height = float(layer["max_height"]) - float(layer["min_height"])
    if height <= 0.0:
        raise ValueError("Layer height must be positive.")
    mesh = trimesh.creation.extrude_polygon(polygon=polygon, height=height)
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, float(layer["min_height"])],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    mesh.apply_transform(transform)
    return mesh


def sample_condition_points(layers: Sequence[dict], point_count: int) -> np.ndarray:
    meshes: List[trimesh.Trimesh] = []
    for layer in layers:
        try:
            meshes.append(extrude_layer_to_mesh(layer))
        except ValueError:
            continue
    if not meshes:
        return np.zeros((1, 3), dtype=np.float32)
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    if len(mesh.faces) == 0:
        return np.zeros((1, 3), dtype=np.float32)
    samples, _ = trimesh.sample.sample_surface(mesh, point_count)
    return samples.astype(np.float32, copy=False)


def save_condition_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.points.PointCloud(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    cloud.export(path)


def build_pair_dataset(
    sequence_dir: Path,
    output_dir: Path,
    pair_mode: str,
    condition_point_count: int,
    change_tolerance: float,
    copy_objs: bool,
    save_condition_ply_flag: bool,
) -> None:
    sequence_meta = load_yaml(sequence_dir / "sequence_meta.yaml") or {}
    stage_names = [
        normalize_stage_name(stage["stage_name"])
        for stage in sequence_meta.get("stages", [])
        if isinstance(stage, dict) and "stage_name" in stage
    ]
    if not stage_names:
        stage_names = sorted(p.name for p in sequence_dir.iterdir() if p.is_dir() and p.name.startswith("stage_"))

    normalized_stage_dirs: Dict[str, Path] = {}
    for stage_name in stage_names:
        normalized_stage_dirs[stage_name] = ensure_stage_copy(sequence_dir, output_dir, stage_name, copy_objs=copy_objs)

    pairs = enumerate_pairs(stage_names, pair_mode)
    pair_records: List[dict] = []
    for src_stage, dst_stage in pairs:
        history_yaml = sequence_dir / src_stage / "building1.yaml"
        target_yaml = sequence_dir / dst_stage / "building1.yaml"
        history_layers = normalize_layers(history_yaml)
        target_layers = normalize_layers(target_yaml)

        token_types, token_values = encode_edit_sequence(history_layers, target_layers)
        pair_name = f"{src_stage}_to_{dst_stage}"

        edit_dir = output_dir / "edit_sequences"
        edit_path = edit_dir / f"{pair_name}.yaml"
        edit_yaml = edit_sequence_to_yaml(token_types, token_values)
        edit_path.parent.mkdir(parents=True, exist_ok=True)
        edit_path.write_text(edit_yaml, encoding="utf-8")

        changed_layers = select_changed_target_layers(history_layers, target_layers, change_tolerance)
        condition_points = sample_condition_points(changed_layers, condition_point_count)
        condition_dir = output_dir / "conditions"
        condition_path = condition_dir / f"{pair_name}_r0.pt"
        condition_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(condition_points), condition_path)
        condition_ply_path = condition_path.with_suffix(".ply")
        if save_condition_ply_flag:
            save_condition_ply(condition_ply_path, condition_points)

        pair_meta = {
            "pair_name": pair_name,
            "sequence_dir": str(sequence_dir.resolve()),
            "history_stage": src_stage,
            "target_stage": dst_stage,
            "history_yaml": str(history_yaml.resolve()),
            "target_yaml": str(target_yaml.resolve()),
            "edit_sequence": str(edit_path.resolve()),
            "condition": str(condition_path.resolve()),
            "condition_ply": str(condition_ply_path.resolve()) if save_condition_ply_flag else "",
            "changed_layer_count": len(changed_layers),
            "condition_point_count": int(condition_points.shape[0]),
        }
        save_yaml(output_dir / "pair_meta" / f"{pair_name}.yaml", pair_meta)

        pair_records.append(
            {
                "t1": str(normalized_stage_dirs[src_stage]),
                "t2": str(normalized_stage_dirs[dst_stage]),
                "condition": str(condition_path.resolve()),
                "edit_sequence": str(edit_path.resolve()),
            }
        )

    save_yaml(output_dir / "train.yaml", pair_records)
    save_yaml(output_dir / "val.yaml", pair_records)
    save_yaml(output_dir / "test.yaml", pair_records)
    save_yaml(
        output_dir / "dataset_meta.yaml",
        {
            "sequence_dir": str(sequence_dir.resolve()),
            "pair_mode": pair_mode,
            "pair_count": len(pair_records),
            "condition_point_count": condition_point_count,
            "change_tolerance": change_tolerance,
            "stage_names": stage_names,
        },
    )


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_pair_dataset(
        sequence_dir=sequence_dir,
        output_dir=output_dir,
        pair_mode=args.pair_mode,
        condition_point_count=args.condition_point_count,
        change_tolerance=args.change_tolerance,
        copy_objs=args.copy_objs,
        save_condition_ply_flag=args.save_condition_ply,
    )
    print(f"Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()

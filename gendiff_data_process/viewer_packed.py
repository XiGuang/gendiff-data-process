from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml  # type: ignore[import-untyped]

from .canonicalization.packed_contract import (
    PACK_SCHEMA,
    validate_packed_release_meta,
    validate_packed_sample,
)

SPLITS = ("train", "val", "test")


class PackedViewerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackedShard:
    split: str
    shard_index: int
    path: Path
    sample_offset: int
    sample_count: int


@dataclass(frozen=True)
class PackedSample:
    split: str
    shard_index: int
    sample_index: int
    global_index: int
    shard_path: Path
    sample: Mapping[str, Any]

    @property
    def locator(self) -> str:
        return f"packed:{self.split}:{self.shard_index}:{self.sample_index}"


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 取决于 viewer 运行环境
        raise PackedViewerError(
            "读取 packed 数据需要包含 PyTorch 的 Python 环境"
        ) from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PackedViewerError(f"无法读取 packed 文件 {path}: {exc}") from exc


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise PackedViewerError(f"无法读取 YAML {path}: {exc}") from exc


def is_packed_dataset(dataset_dir: Path) -> bool:
    root = dataset_dir.resolve()
    if not (root / "states.pt").is_file():
        return False
    yaml_meta = root / "dataset_meta.yaml"
    if yaml_meta.is_file():
        payload = _read_yaml(yaml_meta)
        return (
            isinstance(payload, Mapping)
            and payload.get("schema_version") == PACK_SCHEMA
        )
    pt_meta = root / "dataset_meta.pt"
    if not pt_meta.is_file():
        return False
    payload = _torch_load(pt_meta)
    return isinstance(payload, Mapping) and payload.get("schema_version") == PACK_SCHEMA


def load_packed_metadata(dataset_dir: Path) -> Mapping[str, Any]:
    root = dataset_dir.resolve()
    yaml_path = root / "dataset_meta.yaml"
    pt_path = root / "dataset_meta.pt"
    payload = _read_yaml(yaml_path) if yaml_path.is_file() else _torch_load(pt_path)
    if not isinstance(payload, Mapping):
        raise PackedViewerError("dataset_meta 必须是 mapping")
    validate_packed_release_meta(payload)
    return payload


def _prefer_local_path(dataset_dir: Path, recorded: Any, local_fallback: Path) -> Path:
    candidates = [local_fallback]
    if isinstance(recorded, (str, Path)) and str(recorded):
        recorded_path = Path(recorded)
        if not recorded_path.is_absolute():
            candidates.append(dataset_dir / recorded_path)
        candidates.append(recorded_path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def load_packed_states_payload(dataset_dir: Path) -> Mapping[str, Any]:
    root = dataset_dir.resolve()
    meta = load_packed_metadata(root)
    path = _prefer_local_path(root, meta.get("states_path"), root / "states.pt")
    payload = _torch_load(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("states"), list):
        raise PackedViewerError(f"states 文件格式不兼容: {path}")
    schema = payload.get("schema_version")
    if schema is not None and schema != PACK_SCHEMA:
        raise PackedViewerError(f"states schema 不兼容: {schema}")
    return payload


def load_packed_states(dataset_dir: Path) -> list[Mapping[str, Any]]:
    states = load_packed_states_payload(dataset_dir)["states"]
    if not all(isinstance(state, Mapping) for state in states):
        raise PackedViewerError("states 中存在非 mapping 项")
    return states


def _load_index_payload(dataset_dir: Path, split: str) -> Mapping[str, Any]:
    if split not in SPLITS:
        raise PackedViewerError(f"不支持的 split: {split}")
    root = dataset_dir.resolve()
    pt_path = root / f"{split}_index.pt"
    yaml_path = root / f"{split}_index.yaml"
    if pt_path.is_file():
        payload = _torch_load(pt_path)
    elif yaml_path.is_file():
        payload = _read_yaml(yaml_path)
    else:
        return {"shards": []}
    if not isinstance(payload, Mapping) or not isinstance(payload.get("shards"), list):
        raise PackedViewerError(f"{split} index 格式不兼容")
    return payload


def load_packed_shards(dataset_dir: Path, split: str) -> tuple[PackedShard, ...]:
    root = dataset_dir.resolve()
    payload = _load_index_payload(root, split)
    output: list[PackedShard] = []
    sample_offset = 0
    for shard_index, item in enumerate(payload["shards"]):
        if not isinstance(item, Mapping):
            raise PackedViewerError(
                f"{split} index 的 shard {shard_index} 不是 mapping"
            )
        try:
            sample_count = int(item.get("sample_count", 0))
        except (TypeError, ValueError) as exc:
            raise PackedViewerError(
                f"{split} shard {shard_index} 的 sample_count 无效"
            ) from exc
        if sample_count < 0:
            raise PackedViewerError(
                f"{split} shard {shard_index} 的 sample_count 为负数"
            )
        item_split = item.get("split")
        if item_split is not None and item_split != split:
            raise PackedViewerError(f"{split} index 引用了其他 split: {item_split}")
        recorded = item.get("path")
        basename = (
            Path(str(recorded)).name if recorded else f"{split}_{shard_index:05d}.pt"
        )
        local_path = root / "shards" / split / basename
        path = _prefer_local_path(root, recorded, local_path)
        if not path.is_file():
            raise PackedViewerError(f"找不到 shard: {path}")
        output.append(
            PackedShard(split, shard_index, path, sample_offset, sample_count)
        )
        sample_offset += sample_count
    return tuple(output)


def packed_sample_count(dataset_dir: Path, split: str) -> int:
    return sum(shard.sample_count for shard in load_packed_shards(dataset_dir, split))


def _load_shard_samples(shard: PackedShard) -> list[Mapping[str, Any]]:
    payload = _torch_load(shard.path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("samples"), list):
        raise PackedViewerError(f"shard 格式不兼容: {shard.path}")
    samples = payload["samples"]
    if len(samples) != shard.sample_count:
        raise PackedViewerError(
            f"shard sample_count 不一致: index={shard.sample_count}, actual={len(samples)}"
        )
    if not all(isinstance(sample, Mapping) for sample in samples):
        raise PackedViewerError(f"shard 中存在非 mapping sample: {shard.path}")
    return samples


def iter_packed_samples(
    dataset_dir: Path,
    split: str,
    *,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[PackedSample]:
    if start < 0 or (stop is not None and stop < start):
        raise ValueError("packed sample 范围无效")
    for shard in load_packed_shards(dataset_dir, split):
        shard_stop = shard.sample_offset + shard.sample_count
        if shard_stop <= start:
            continue
        if stop is not None and shard.sample_offset >= stop:
            break
        samples = _load_shard_samples(shard)
        local_start = max(0, start - shard.sample_offset)
        local_stop = min(
            shard.sample_count,
            shard.sample_count if stop is None else stop - shard.sample_offset,
        )
        for sample_index in range(local_start, local_stop):
            sample = samples[sample_index]
            validate_packed_sample(sample)
            yield PackedSample(
                split,
                shard.shard_index,
                sample_index,
                shard.sample_offset + sample_index,
                shard.path,
                sample,
            )


def _decode_locator(locator: str) -> tuple[str, int, int]:
    parts = locator.split(":")
    if len(parts) != 4 or parts[0] != "packed" or parts[1] not in SPLITS:
        raise PackedViewerError(f"packed locator 无效: {locator}")
    try:
        shard_index = int(parts[2])
        sample_index = int(parts[3])
    except ValueError as exc:
        raise PackedViewerError(f"packed locator 无效: {locator}") from exc
    if shard_index < 0 or sample_index < 0:
        raise PackedViewerError(f"packed locator 无效: {locator}")
    return parts[1], shard_index, sample_index


def find_packed_sample(
    dataset_dir: Path,
    *,
    pair_name: str | None = None,
    locator: str | None = None,
) -> PackedSample:
    if locator:
        split, shard_index, sample_index = _decode_locator(locator)
        shards = load_packed_shards(dataset_dir, split)
        if shard_index >= len(shards):
            raise PackedViewerError(f"locator shard 越界: {locator}")
        shard = shards[shard_index]
        samples = _load_shard_samples(shard)
        if sample_index >= len(samples):
            raise PackedViewerError(f"locator sample 越界: {locator}")
        sample = samples[sample_index]
        validate_packed_sample(sample)
        record = PackedSample(
            split,
            shard_index,
            sample_index,
            shard.sample_offset + sample_index,
            shard.path,
            sample,
        )
        if pair_name and sample.get("pair_name") != pair_name:
            raise PackedViewerError("locator 与 pair_name 不一致")
        return record
    if not pair_name:
        raise PackedViewerError("需要 pair_name 或 locator")
    for split in SPLITS:
        for record in iter_packed_samples(dataset_dir, split):
            if record.sample.get("pair_name") == pair_name:
                return record
    raise PackedViewerError(f"packed 数据中找不到 pair: {pair_name}")


def packed_condition_points(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        raise PackedViewerError("packed condition 不是点列表或 tensor")
    points: list[list[float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            raise PackedViewerError(f"packed condition 第 {index} 个点格式无效")
        points.append([float(point[0]), float(point[1]), float(point[2])])
    return points

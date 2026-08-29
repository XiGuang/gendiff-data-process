from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .types import RawBuildingSequence, RawLayer, RawStage


@dataclass(frozen=True)
class SourceFileEvidence:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LoadedHistorySequence:
    sequence: RawBuildingSequence
    source_files: tuple[SourceFileEvidence, ...]


def load_history_sequence(
    dataset_root: str | Path,
    building_key: str,
    stage_indices: tuple[int, ...],
    *,
    max_stage_yaml_bytes: int = 16 * 1024 * 1024,
) -> LoadedHistorySequence:
    root = Path(dataset_root).resolve()
    building_dir = (root / building_key).resolve()
    if not building_dir.is_relative_to(root) or not building_dir.is_dir():
        raise ValueError(f"building 目录不存在或越界: {building_key}")
    stages: list[RawStage] = []
    evidence: list[SourceFileEvidence] = []
    for stage_index in stage_indices:
        path = (building_dir / f"stage_{stage_index}" / f"stage_{stage_index}.yaml").resolve()
        if not path.is_relative_to(building_dir) or not path.is_file():
            raise ValueError(f"缺少预期 stage YAML: {path}")
        size_bytes = path.stat().st_size
        if size_bytes > max_stage_yaml_bytes:
            raise ValueError(f"stage YAML 超过有界读取上限: {path} ({size_bytes} bytes)")
        payload = path.read_bytes()
        mapping = yaml.safe_load(payload.decode("utf-8")) if payload else []
        if mapping is None:
            mapping = []
        if not isinstance(mapping, list):
            raise ValueError(f"stage YAML 顶层必须是 layer list: {path}")
        layers = tuple(RawLayer.from_mapping(layer) for layer in mapping)
        stages.append(RawStage(stage_index, f"stage_{stage_index}", layers))
        evidence.append(
            SourceFileEvidence(
                path=str(path),
                size_bytes=size_bytes,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return LoadedHistorySequence(
        RawBuildingSequence(
            building_key=building_key,
            coordinate_frame="world_xzy",
            stages=tuple(stages),
            expected_stage_indices=stage_indices,
        ),
        tuple(evidence),
    )

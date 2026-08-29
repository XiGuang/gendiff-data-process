from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

PointQ = tuple[int, int]


@dataclass(frozen=True)
class RawLayer:
    min_height: Any
    max_height: Any
    footprint: tuple[tuple[Any, Any], ...]
    raw_proxy_id: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RawLayer":
        points = tuple((point[0], point[1]) for point in value.get("footprint") or [])
        return cls(
            min_height=value.get("min_height"),
            max_height=value.get("max_height"),
            footprint=points,
            raw_proxy_id=value.get("raw_proxy_id", value.get("proxy_id")),
            provenance={"source": dict(value)},
        )


@dataclass(frozen=True)
class RawStage:
    stage_index: int
    stage_key: str
    layers: tuple[RawLayer, ...]

    @classmethod
    def from_layers(cls, stage_index: int, layers: Sequence[Mapping[str, Any]]) -> "RawStage":
        return cls(stage_index, f"stage_{stage_index}", tuple(RawLayer.from_mapping(item) for item in layers))


@dataclass(frozen=True)
class RawBuildingSequence:
    building_key: str
    coordinate_frame: str
    stages: tuple[RawStage, ...]
    expected_stage_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class QuantizedLayer:
    min_height_q: int
    max_height_q: int
    footprint_q: tuple[PointQ, ...]
    raw_proxy_id: Any = None


@dataclass(frozen=True)
class CanonicalLayer:
    canonical_layer_index: int
    min_height_q: int
    max_height_q: int
    footprint_q: tuple[PointQ, ...]
    geometry_hash: str
    layer_lineage_id: int | None = None
    point_lineage_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CanonicalStage:
    stage_index: int
    stage_key: str
    layers: tuple[CanonicalLayer, ...]
    stage_hash: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PointEdit:
    action: str
    point_lineage_id: int | None
    source_index: int | None
    target_index: int | None
    target_coord_q: PointQ | None = None


@dataclass(frozen=True)
class LayerEdit:
    action: str
    layer_lineage_id: int
    source_layer_index: int | None
    target_layer_index: int | None
    source_height_q: tuple[int, int] | None
    target_height_q: tuple[int, int] | None
    point_edits: tuple[PointEdit, ...]


@dataclass(frozen=True)
class CanonicalEdit:
    source_stage_hash: str
    target_stage_hash: str
    target_stage_index: int
    target_stage_key: str
    canonicalizer_config_hash: str
    layer_edits: tuple[LayerEdit, ...]
    edit_hash: str


@dataclass(frozen=True)
class CanonicalBuildingSequence:
    building_key: str
    building_uid: str
    canonicalizer_version: str
    geometry_version: str
    geometry_config_hash: str
    canonicalizer_config_hash: str
    stages: tuple[CanonicalStage, ...]
    adjacent_edits: tuple[CanonicalEdit, ...]
    sequence_hash: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerMatch:
    source_index: int
    target_index: int


@dataclass(frozen=True)
class PointAlignment:
    edits: tuple[PointEdit, ...]
    warning: str | None = None

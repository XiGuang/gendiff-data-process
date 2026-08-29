from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from ..config import CanonicalizerBundle
from ..errors import CanonicalizationError
from ..packed_contract import canonical_pair_hash
from ..quantize import dequantize_scalar
from ..solid_partition import classify_stage_change
from ..types import (
    CanonicalBuildingSequence,
    CanonicalEdit,
    CanonicalLayer,
    CanonicalStage,
    PointEdit,
)


@dataclass(frozen=True)
class AreaNormalizationStats:
    profile_id: str
    center_x: float
    center_z: float
    scale_xz: float
    center_y: float
    scale_y: float

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("normalization profile_id 不能为空")
        if self.scale_xz <= 0 or self.scale_y <= 0:
            raise ValueError("normalization scale 必须为正数")

    @property
    def tensor_order(self) -> tuple[float, ...]:
        return (
            self.center_x,
            self.center_z,
            self.scale_xz,
            self.center_y,
            self.scale_y,
        )

    def normalize_xz(self, point_q: tuple[int, int], grid_xz: str) -> list[float]:
        x = float(dequantize_scalar(point_q[0], grid_xz))
        z = float(dequantize_scalar(point_q[1], grid_xz))
        return [
            (x - self.center_x) / self.scale_xz,
            (z - self.center_z) / self.scale_xz,
        ]

    def normalize_y(self, height_q: int, grid_y: str) -> float:
        value = float(dequantize_scalar(height_q, grid_y))
        return (value - self.center_y) / self.scale_y

    def normalize_xyz(
        self, point_q: tuple[int, int, int], grid_xz: str, grid_y: str
    ) -> list[float]:
        x = float(dequantize_scalar(point_q[0], grid_xz))
        y = float(dequantize_scalar(point_q[1], grid_y))
        z = float(dequantize_scalar(point_q[2], grid_xz))
        return [
            (x - self.center_x) / self.scale_xz,
            (y - self.center_y) / self.scale_y,
            (z - self.center_z) / self.scale_xz,
        ]


@dataclass(frozen=True)
class AreaV2Capacity:
    max_layers: int = 64
    max_points_per_layer: int = 32
    max_buildings: int = 16


def validate_area_v2_edit_capacity(
    source: CanonicalStage,
    edit: CanonicalEdit,
    capacity: AreaV2Capacity,
) -> None:
    insert_count = sum(
        layer_edit.action == "INSERT_LAYER" for layer_edit in edit.layer_edits
    )
    if len(source.layers) + insert_count > capacity.max_layers:
        raise CanonicalizationError(
            "E_LAYER_CAPACITY", "source slots 加 insert slots 超过模型容量"
        )
    for layer_edit in edit.layer_edits:
        point_count = sum(
            point_edit.action != "EOS" for point_edit in layer_edit.point_edits
        )
        if point_count > capacity.max_points_per_layer:
            raise CanonicalizationError(
                "E_POINT_CAPACITY", "adapter point edit 超过 AR token 容量"
            )


class AreaV2Adapter:
    PACK_SCHEMA = "area_v2_packed_v1"
    EDIT_SCHEMA = "area_v2_absolute_target_coord_no_anchor"

    def __init__(
        self,
        bundle: CanonicalizerBundle,
        normalization: AreaNormalizationStats,
        capacity: AreaV2Capacity | None = None,
    ) -> None:
        self.bundle = bundle
        self.normalization = normalization
        self.capacity = capacity or AreaV2Capacity(
            bundle.validation_profile.max_layers,
            bundle.validation_profile.max_points_per_layer,
            bundle.validation_profile.max_buildings_per_tile,
        )

    def canonical_contract(self) -> dict:
        task_contract_id = (
            "bidirectional_monotonic_v1"
            if self.bundle.validation_profile.mode == "bidirectional_monotonic"
            else "construction_only_v1"
        )
        return {
            "task_contract_id": task_contract_id,
            "validation_mode": self.bundle.validation_profile.mode,
            "condition_surface_mode": self.bundle.condition_sampling.surface_mode,
            "canonicalizer_version": self.bundle.canonicalizer.canonicalizer_version,
            "geometry_version": self.bundle.canonicalizer.geometry_version,
            "geometry_config_hash": self.bundle.canonicalizer.geometry_config_hash,
            "canonicalizer_config_hash": self.bundle.canonicalizer.canonicalizer_config_hash,
            "validation_profile_hash": self.bundle.validation_profile.config_hash,
            "condition_config_hash": self.bundle.condition_sampling.config_hash,
            "normalization_config_hash": self.bundle.normalization.config_hash,
            "split_config_hash": self.bundle.split.config_hash,
            "package_config_hash": self.bundle.package.config_hash,
            "normalization_profile_id": self.normalization.profile_id,
        }

    @staticmethod
    def _proxy_id(dense_building_id: int, layer_lineage_id: int) -> int:
        if not 0 <= layer_lineage_id < 10_000:
            raise CanonicalizationError(
                "E_ID_OVERFLOW", "layer lineage 超出 area-v2 stride"
            )
        return dense_building_id * 1_000_000_000 + layer_lineage_id

    @staticmethod
    def _point_id(
        dense_building_id: int, layer_lineage_id: int, point_lineage_id: int
    ) -> int:
        if not 0 <= point_lineage_id < 100_000:
            raise CanonicalizationError(
                "E_ID_OVERFLOW", "point lineage 超出 area-v2 stride"
            )
        return (
            dense_building_id * 1_000_000_000
            + layer_lineage_id * 100_000
            + point_lineage_id
        )

    def _layer(self, layer: CanonicalLayer, dense_building_id: int) -> dict:
        if layer.layer_lineage_id is None or len(layer.point_lineage_ids) != len(
            layer.footprint_q
        ):
            raise ValueError(
                "area-v2 adapter 只接受已分配完整 lineage 的 canonical layer"
            )
        if len(layer.footprint_q) > self.capacity.max_points_per_layer:
            raise CanonicalizationError("E_POINT_CAPACITY", "adapter 输入层超过点容量")
        lineage = int(layer.layer_lineage_id)
        return {
            "proxy_id": self._proxy_id(dense_building_id, lineage),
            "source_proxy_id": self._proxy_id(dense_building_id, lineage),
            "building_id": dense_building_id,
            "building_layer_index": layer.canonical_layer_index,
            "min_height": self.normalization.normalize_y(
                layer.min_height_q, self.bundle.canonicalizer.grid_y
            ),
            "max_height": self.normalization.normalize_y(
                layer.max_height_q, self.bundle.canonicalizer.grid_y
            ),
            "footprint": [
                self.normalization.normalize_xz(
                    point, self.bundle.canonicalizer.grid_xz
                )
                for point in layer.footprint_q
            ],
            "point_ids": [
                self._point_id(dense_building_id, lineage, point_id)
                for point_id in layer.point_lineage_ids
            ],
            "source_point_ids": [
                self._point_id(dense_building_id, lineage, point_id)
                for point_id in layer.point_lineage_ids
            ],
            "canonical_layer_index": layer.canonical_layer_index,
            "layer_lineage_id": lineage,
            "geometry_hash": layer.geometry_hash,
        }

    def adapt_state(
        self, stage: CanonicalStage, dense_building_id: int, building_key: str
    ) -> dict:
        if not 0 <= dense_building_id < self.capacity.max_buildings:
            raise CanonicalizationError(
                "E_BUILDING_CAPACITY", "dense building ID 超出 tile 容量"
            )
        if len(stage.layers) > self.capacity.max_layers:
            raise CanonicalizationError(
                "E_LAYER_CAPACITY", "adapter 输入 stage 超过层容量"
            )
        return {
            "state_name": f"{building_key}/{stage.stage_key}",
            "layers": [self._layer(layer, dense_building_id) for layer in stage.layers],
            "meta": {
                "building_key": building_key,
                "stage_index": stage.stage_index,
                "stage_hash": stage.stage_hash,
                "normalization_profile_id": self.normalization.profile_id,
            },
        }

    def _point_entry(
        self,
        point_edit: PointEdit,
        source_layer: CanonicalLayer | None,
        target_layer: CanonicalLayer | None,
    ) -> dict | None:
        if point_edit.action == "EOS":
            return None
        action = point_edit.action.removesuffix("_POINT")
        source_coord = None
        if source_layer is not None and point_edit.source_index is not None:
            source_coord = self.normalization.normalize_xz(
                source_layer.footprint_q[point_edit.source_index],
                self.bundle.canonicalizer.grid_xz,
            )
        target_coord = None
        if target_layer is not None and point_edit.target_index is not None:
            target_coord = self.normalization.normalize_xz(
                target_layer.footprint_q[point_edit.target_index],
                self.bundle.canonicalizer.grid_xz,
            )
        value = target_coord if action in {"MOVE", "INSERT"} else [0.0, 0.0]
        return {
            "action": action,
            "source_point_index": point_edit.source_index,
            "target_point_index": point_edit.target_index,
            "source_coord": source_coord,
            "target_coord": target_coord,
            "value": value,
            "point_lineage_id": point_edit.point_lineage_id,
        }

    def adapt_edit(
        self,
        source: CanonicalStage,
        target: CanonicalStage,
        edit: CanonicalEdit,
        dense_building_id: int,
    ) -> list[dict]:
        source_by_index = {
            layer.canonical_layer_index: layer for layer in source.layers
        }
        target_by_index = {
            layer.canonical_layer_index: layer for layer in target.layers
        }
        validate_area_v2_edit_capacity(source, edit, self.capacity)

        output: list[dict] = []
        for layer_edit in edit.layer_edits:
            source_layer = (
                source_by_index.get(layer_edit.source_layer_index)
                if layer_edit.source_layer_index is not None
                else None
            )
            target_layer = (
                target_by_index.get(layer_edit.target_layer_index)
                if layer_edit.target_layer_index is not None
                else None
            )
            point_entries = [
                entry
                for point_edit in layer_edit.point_edits
                if (entry := self._point_entry(point_edit, source_layer, target_layer))
                is not None
            ]
            action = layer_edit.action.removesuffix("_LAYER")
            source_proxy_id = (
                self._proxy_id(dense_building_id, layer_edit.layer_lineage_id)
                if source_layer is not None
                else None
            )
            target_proxy_id = (
                self._proxy_id(dense_building_id, layer_edit.layer_lineage_id)
                if target_layer is not None
                else None
            )
            source_height = (
                [
                    self.normalization.normalize_y(
                        source_layer.min_height_q, self.bundle.canonicalizer.grid_y
                    ),
                    self.normalization.normalize_y(
                        source_layer.max_height_q, self.bundle.canonicalizer.grid_y
                    ),
                ]
                if source_layer is not None
                else None
            )
            target_height = (
                [
                    self.normalization.normalize_y(
                        target_layer.min_height_q, self.bundle.canonicalizer.grid_y
                    ),
                    self.normalization.normalize_y(
                        target_layer.max_height_q, self.bundle.canonicalizer.grid_y
                    ),
                ]
                if target_layer is not None
                else None
            )
            output.append(
                {
                    "action": action,
                    "source_layer_index": layer_edit.source_layer_index,
                    "target_layer_index": layer_edit.target_layer_index,
                    "source_proxy_id": source_proxy_id,
                    "target_proxy_id": target_proxy_id,
                    "source_building_id": (
                        dense_building_id if source_layer is not None else None
                    ),
                    "target_building_id": (
                        dense_building_id if target_layer is not None else None
                    ),
                    "height_edit": {"source": source_height, "target": target_height},
                    "point_edits": point_entries,
                    "layer_lineage_id": layer_edit.layer_lineage_id,
                }
            )
        return output

    def adapt_transition(
        self,
        sequence: CanonicalBuildingSequence,
        source_position: int,
        target_position: int,
        edit: CanonicalEdit,
        condition_points_normalized: Iterable[Sequence[float]],
        *,
        dense_building_id: int = 0,
        condition_hash: str,
        split: str,
    ) -> dict:
        try:
            source = sequence.stages[source_position]
            target = sequence.stages[target_position]
        except IndexError as exc:
            raise ValueError("source/target stage position 越界") from exc
        if source_position == target_position:
            raise ValueError("source/target stage position 不能相同")
        if (
            edit.source_stage_hash != source.stage_hash
            or edit.target_stage_hash != target.stage_hash
        ):
            raise CanonicalizationError(
                "E_ROUNDTRIP_MISMATCH", "adapter edit 与 source/target stage 不匹配"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError("split 必须是 train、val 或 test")
        if not condition_hash:
            raise ValueError("condition_hash 不能为空")
        condition = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in condition_points_normalized
        ]
        if len(condition) != self.bundle.condition_sampling.point_count:
            raise CanonicalizationError(
                "E_CONDITION_SAMPLING",
                "adapter condition 点数与冻结配置不一致",
                expected=self.bundle.condition_sampling.point_count,
                actual=len(condition),
            )
        if any(
            len(point) != 3 or not all(math.isfinite(value) for value in point)
            for point in condition
        ):
            raise CanonicalizationError(
                "E_NONFINITE_VALUE", "adapter condition 必须是有限 XYZ 三元组"
            )
        change = classify_stage_change(
            source,
            target,
            volume_tolerance_q3=self.bundle.validation_profile.removed_volume_tolerance_q3,
        )
        if change.change_kind == "noop":
            raise CanonicalizationError(
                "E_CONDITION_EMPTY", "adapter 不接受 no-op pair"
            )
        canonical_metadata = {
            "task_contract_id": self.canonical_contract()["task_contract_id"],
            "validation_mode": self.bundle.validation_profile.mode,
            "condition_surface_mode": self.bundle.condition_sampling.surface_mode,
            "canonicalizer_version": sequence.canonicalizer_version,
            "geometry_version": sequence.geometry_version,
            "geometry_config_hash": sequence.geometry_config_hash,
            "canonicalizer_config_hash": sequence.canonicalizer_config_hash,
            "validation_profile_hash": self.bundle.validation_profile.config_hash,
            "condition_config_hash": self.bundle.condition_sampling.config_hash,
            "normalization_config_hash": self.bundle.normalization.config_hash,
            "split_config_hash": self.bundle.split.config_hash,
            "package_config_hash": self.bundle.package.config_hash,
            "source_stage_hash": source.stage_hash,
            "target_stage_hash": target.stage_hash,
            "edit_hash": edit.edit_hash,
            "condition_hash": condition_hash,
            "building_uid": sequence.building_uid,
            "split": split,
            "normalization_profile_id": self.normalization.profile_id,
            "change_kind": change.change_kind,
        }
        canonical_metadata["pair_hash"] = canonical_pair_hash(canonical_metadata)
        return {
            "source_state": self.adapt_state(
                source, dense_building_id, sequence.building_key
            ),
            "target_state": self.adapt_state(
                target, dense_building_id, sequence.building_key
            ),
            "sample": {
                "pair_name": f"{sequence.building_key}_{source.stage_key}_to_{target.stage_key}",
                "condition": condition,
                "edit_object": self.adapt_edit(source, target, edit, dense_building_id),
                "validation": {
                    "canonical_roundtrip": True,
                    "canonical_metadata": canonical_metadata,
                },
                "canonical_metadata": canonical_metadata,
            },
            "normalization_stats_tensor": list(self.normalization.tensor_order),
            "edit_schema_version": self.EDIT_SCHEMA,
            "pack_schema_version": self.PACK_SCHEMA,
        }

    def adapt_pair(
        self,
        sequence: CanonicalBuildingSequence,
        pair_index: int,
        condition_points_normalized: Iterable[Sequence[float]],
        *,
        dense_building_id: int = 0,
        condition_hash: str,
        split: str,
    ) -> dict:
        try:
            edit = sequence.adjacent_edits[pair_index]
        except IndexError as exc:
            raise ValueError("pair_index 越界") from exc
        return self.adapt_transition(
            sequence,
            pair_index,
            pair_index + 1,
            edit,
            condition_points_normalized,
            dense_building_id=dense_building_id,
            condition_hash=condition_hash,
            split=split,
        )

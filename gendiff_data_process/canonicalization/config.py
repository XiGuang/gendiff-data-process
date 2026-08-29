from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from .serialize import canonical_hash


@dataclass(frozen=True)
class PolygonConfig:
    remove_collinear: bool = True
    reject_self_intersection: bool = True
    reject_holes: bool = True
    multipolygon_policy: str = "split"
    raw_overlap_policy: str = "union_with_warning"


@dataclass(frozen=True)
class LayerMatchingConfig:
    require_positive_intersection: bool = True
    min_iou_q: int = 10_000
    min_smaller_coverage_q: int = 500_000
    metric_scale: int = 1_000_000
    optimal_tie_break: str = "lexicographic"


@dataclass(frozen=True)
class PointMatchingConfig:
    max_move_distance_ratio: str = "0.25"
    min_move_distance_q: int = 5
    fallback: str = "delete_then_insert"

    @property
    def move_ratio(self) -> Decimal:
        return Decimal(self.max_move_distance_ratio)


@dataclass(frozen=True)
class CanonicalizerConfig:
    canonicalizer_version: str = "canonicalizer_v1"
    geometry_version: str = "canonical_geometry_v1"
    coordinate_frame: str = "world_xzy"
    grid_xz: str = "0.001"
    grid_y: str = "0.001"
    rounding: str = "half_away_from_zero"
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    layer_matching: LayerMatchingConfig = field(default_factory=LayerMatchingConfig)
    point_matching: PointMatchingConfig = field(default_factory=PointMatchingConfig)

    @property
    def geometry_mapping(self) -> dict[str, Any]:
        return {
            "geometry_version": self.geometry_version,
            "coordinate_frame": self.coordinate_frame,
            "grid_xz": self.grid_xz,
            "grid_y": self.grid_y,
            "rounding": self.rounding,
            "polygon": asdict(self.polygon),
            "solid_partition": "height_events_union_cells_v1",
        }

    @property
    def geometry_config_hash(self) -> str:
        return canonical_hash(self.geometry_mapping)

    @property
    def canonicalizer_config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class ValidationProfile:
    mode: str = "construction_only"
    removed_volume_tolerance_q3: int = 0
    drop_noop_pairs: bool = True
    max_layers: int = 64
    max_points_per_layer: int = 32
    max_buildings_per_tile: int = 16
    overflow_policy: str = "error"

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class ConditionConfig:
    surface_mode: str = "addition_exterior"
    point_count: int = 2048
    sampler: str = "deterministic_stratified_fps"
    candidate_multiplier: int = 2
    allocation: str = "largest_remainder"
    low_discrepancy_sequence: str = "halton_2_3"
    fps_start: str = "lexicographic_minimum"
    final_order: str = "lexicographic_xyz"
    seed_material: str = "source_target_condition_config_hash"

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class NormalizationConfig:
    method: str = "train_bbox_uniform_v1"
    source_split: str = "train_only"
    coordinate_order: str = "world_xzy"
    min_scale: str = "0.001"
    scale_policy: str = "max_xyz_span"
    profile_id_prefix: str = "train_bbox_uniform_v1"

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class SplitConfig:
    algorithm: str = "sha256_threshold_v1"
    seed: str = "canonicalizer_split_20260829_v1"
    modulus: int = 10_000
    train_threshold: int = 8_000
    validation_threshold: int = 9_000

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class PackageConfig:
    distribution_name: str = "gendiff-data-process"
    import_name: str = "gendiff_data_process"
    version: str = "0.1.0"
    pin_mode: str = "git_commit_and_wheel_sha256"
    source_copy_policy: str = "forbidden"

    @property
    def config_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class CanonicalizerBundle:
    canonicalizer: CanonicalizerConfig = field(default_factory=CanonicalizerConfig)
    validation_profile: ValidationProfile = field(default_factory=ValidationProfile)
    condition_sampling: ConditionConfig = field(default_factory=ConditionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    package: PackageConfig = field(default_factory=PackageConfig)


def _construct(mapping: Mapping[str, Any]) -> CanonicalizerBundle:
    canonicalizer = dict(mapping.get("canonicalizer") or {})
    canonicalizer["polygon"] = PolygonConfig(**dict(canonicalizer.get("polygon") or {}))
    canonicalizer["layer_matching"] = LayerMatchingConfig(
        **dict(canonicalizer.get("layer_matching") or {})
    )
    canonicalizer["point_matching"] = PointMatchingConfig(
        **dict(canonicalizer.get("point_matching") or {})
    )
    bundle = CanonicalizerBundle(
        canonicalizer=CanonicalizerConfig(**canonicalizer),
        validation_profile=ValidationProfile(
            **dict(mapping.get("validation_profile") or {})
        ),
        condition_sampling=ConditionConfig(
            **dict(mapping.get("condition_sampling") or {})
        ),
        normalization=NormalizationConfig(**dict(mapping.get("normalization") or {})),
        split=SplitConfig(**dict(mapping.get("split") or {})),
        package=PackageConfig(**dict(mapping.get("package") or {})),
    )
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: CanonicalizerBundle) -> None:
    cfg = bundle.canonicalizer
    if cfg.coordinate_frame != "world_xzy":
        raise ValueError("canonicalizer_v1 只支持 world_xzy 坐标系")
    try:
        grids = (Decimal(cfg.grid_xz), Decimal(cfg.grid_y))
    except Exception as exc:
        raise ValueError("量化网格必须是有限十进制数") from exc
    if any(not grid.is_finite() or grid <= 0 for grid in grids):
        raise ValueError("量化网格必须是有限正数")
    if cfg.rounding != "half_away_from_zero":
        raise ValueError("canonicalizer_v1 只支持 half_away_from_zero")
    if not cfg.polygon.remove_collinear:
        raise ValueError("canonicalizer_v1 要求删除共线冗余点")
    if not cfg.polygon.reject_self_intersection or not cfg.polygon.reject_holes:
        raise ValueError("canonicalizer_v1 要求拒绝自交和 hole")
    if cfg.polygon.multipolygon_policy != "split":
        raise ValueError("canonicalizer_v1 的 MultiPolygon 策略必须是 split")
    if cfg.polygon.raw_overlap_policy != "union_with_warning":
        raise ValueError(
            "canonicalizer_v1 的 raw overlap 策略必须是 union_with_warning"
        )
    matching = cfg.layer_matching
    if matching.metric_scale <= 0:
        raise ValueError("layer matching metric_scale 必须为正数")
    if not 0 <= matching.min_iou_q <= matching.metric_scale:
        raise ValueError("min_iou_q 必须在 metric_scale 范围内")
    if not 0 <= matching.min_smaller_coverage_q <= matching.metric_scale:
        raise ValueError("min_smaller_coverage_q 必须在 metric_scale 范围内")
    if matching.optimal_tie_break != "lexicographic":
        raise ValueError("canonicalizer_v1 只支持 lexicographic layer tie-break")
    if cfg.point_matching.move_ratio < 0 or cfg.point_matching.min_move_distance_q < 0:
        raise ValueError("point matching 移动阈值不能为负数")
    if cfg.point_matching.fallback != "delete_then_insert":
        raise ValueError("canonicalizer_v1 只支持 delete_then_insert fallback")

    profile = bundle.validation_profile
    if profile.mode not in {"construction_only", "bidirectional_monotonic"}:
        raise ValueError(
            "validation profile 只支持 construction_only 或 bidirectional_monotonic"
        )
    if profile.removed_volume_tolerance_q3 < 0:
        raise ValueError("removed volume tolerance 不能为负数")
    if (
        min(
            profile.max_layers,
            profile.max_points_per_layer,
            profile.max_buildings_per_tile,
        )
        <= 0
    ):
        raise ValueError("capacity 必须为正数")
    if profile.overflow_policy != "error":
        raise ValueError("canonicalizer_v1 的 overflow policy 必须是 error")

    condition = bundle.condition_sampling
    if condition.surface_mode not in {
        "addition_exterior",
        "directional_delta_exterior",
    }:
        raise ValueError(
            "condition 只支持 addition_exterior 或 directional_delta_exterior"
        )
    if condition.point_count <= 0 or condition.candidate_multiplier < 1:
        raise ValueError("condition point_count 和 candidate_multiplier 必须为正数")
    if condition.sampler != "deterministic_stratified_fps":
        raise ValueError("canonicalizer_v1 只支持 deterministic_stratified_fps")
    if condition.allocation != "largest_remainder":
        raise ValueError("condition allocation 必须是 largest_remainder")
    if condition.low_discrepancy_sequence != "halton_2_3":
        raise ValueError("condition 低差异序列必须是 halton_2_3")
    if (
        condition.fps_start != "lexicographic_minimum"
        or condition.final_order != "lexicographic_xyz"
    ):
        raise ValueError("condition FPS 起点和最终排序必须使用字典序合同")
    expected_seed_material = {
        "addition_exterior": "source_target_condition_config_hash",
        "directional_delta_exterior": "unordered_stage_pair_condition_config_hash",
    }[condition.surface_mode]
    if condition.seed_material != expected_seed_material:
        raise ValueError("condition seed material 与 surface mode 合同不一致")
    if (
        profile.mode == "construction_only"
        and condition.surface_mode != "addition_exterior"
    ):
        raise ValueError("construction_only 必须使用 addition_exterior condition")
    if (
        profile.mode == "bidirectional_monotonic"
        and condition.surface_mode != "directional_delta_exterior"
    ):
        raise ValueError(
            "bidirectional_monotonic 必须使用 directional_delta_exterior condition"
        )

    normalization = bundle.normalization
    if (
        normalization.method != "train_bbox_uniform_v1"
        or normalization.source_split != "train_only"
        or normalization.coordinate_order != "world_xzy"
        or normalization.scale_policy != "max_xyz_span"
    ):
        raise ValueError("normalization 配置与 train-only bbox uniform v1 合同不一致")
    minimum_scale = Decimal(normalization.min_scale)
    if not minimum_scale.is_finite() or minimum_scale <= 0:
        raise ValueError("normalization min_scale 必须是有限正数")
    if not normalization.profile_id_prefix:
        raise ValueError("normalization profile_id_prefix 不能为空")

    split = bundle.split
    if split.algorithm != "sha256_threshold_v1" or not split.seed:
        raise ValueError("split 必须使用带固定 seed 的 sha256_threshold_v1")
    if not 0 < split.train_threshold < split.validation_threshold < split.modulus:
        raise ValueError("split threshold 必须满足 0 < train < validation < modulus")

    package = bundle.package
    if (
        package.pin_mode != "git_commit_and_wheel_sha256"
        or package.source_copy_policy != "forbidden"
    ):
        raise ValueError("package 必须按 commit+wheel SHA256 固定并禁止复制源码")
    if not package.distribution_name or not package.import_name or not package.version:
        raise ValueError("package identity 字段不能为空")


def load_bundle(path: str | Path) -> CanonicalizerBundle:
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(mapping, Mapping)
        or mapping.get("schema_version") != "canonicalizer_bundle_v1"
    ):
        raise ValueError("配置必须使用 canonicalizer_bundle_v1 schema")
    return _construct(mapping)

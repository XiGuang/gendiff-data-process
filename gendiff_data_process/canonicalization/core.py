from __future__ import annotations

import unicodedata
from dataclasses import replace

from .config import CanonicalizerBundle, CanonicalizerConfig
from .edit_v3 import apply_canonical_edit, build_canonical_edit
from .errors import CanonicalizationError
from .layer_matching import match_layers
from .point_matching import align_points
from .polygon import area2, cleanup_ring
from .quantize import quantize_points_with_collapse, quantize_scalar
from .release_contracts import building_uid_from_key
from .serialize import canonical_hash
from .solid_partition import SlabCell, build_slab_cells, removed_volume2
from .types import (
    CanonicalBuildingSequence,
    CanonicalLayer,
    CanonicalStage,
    QuantizedLayer,
    RawBuildingSequence,
    RawLayer,
    RawStage,
)


def _geometry_hash(min_height_q: int, max_height_q: int, footprint_q) -> str:
    return canonical_hash(
        {
            "min_height_q": min_height_q,
            "max_height_q": max_height_q,
            "footprint_q": footprint_q,
        }
    )


def _stage_hash(layers: tuple[CanonicalLayer, ...], cfg: CanonicalizerConfig) -> str:
    return canonical_hash(
        {
            "geometry_version": cfg.geometry_version,
            "geometry_config_hash": cfg.geometry_config_hash,
            "layers": [
                {
                    "min_height_q": layer.min_height_q,
                    "max_height_q": layer.max_height_q,
                    "footprint_q": layer.footprint_q,
                }
                for layer in layers
            ],
        }
    )


def stage_from_canonical_layers(
    stage_index: int,
    stage_key: str,
    layers: tuple[CanonicalLayer, ...],
    cfg: CanonicalizerConfig,
    warnings: tuple[str, ...] = (),
) -> CanonicalStage:
    sorted_layers = sorted(
        layers,
        key=lambda layer: (
            layer.min_height_q,
            layer.max_height_q,
            abs(area2(layer.footprint_q)),
            layer.footprint_q,
        ),
    )
    normalized = tuple(
        replace(
            layer,
            canonical_layer_index=index,
            geometry_hash=_geometry_hash(layer.min_height_q, layer.max_height_q, layer.footprint_q),
        )
        for index, layer in enumerate(sorted_layers)
    )
    return CanonicalStage(stage_index, stage_key, normalized, _stage_hash(normalized, cfg), tuple(sorted(set(warnings))))


def _quantize_layer(raw: RawLayer, cfg: CanonicalizerConfig) -> tuple[QuantizedLayer, tuple[str, ...]]:
    minimum = quantize_scalar(raw.min_height, cfg.grid_y)
    maximum = quantize_scalar(raw.max_height, cfg.grid_y)
    if maximum <= minimum:
        raise CanonicalizationError(
            "E_INVALID_HEIGHT",
            "量化后 max_height 必须大于 min_height",
            raw_proxy_id=raw.raw_proxy_id,
            min_height_q=minimum,
            max_height_q=maximum,
        )
    quantized_points, collapsed = quantize_points_with_collapse(raw.footprint, cfg.grid_xz)
    ring = cleanup_ring(quantized_points, remove_collinear=cfg.polygon.remove_collinear)
    warnings = ("W_QUANTIZATION_COLLAPSE",) if collapsed else ()
    return QuantizedLayer(minimum, maximum, ring, raw.raw_proxy_id), warnings


def canonicalize_stage(
    raw_stage: RawStage,
    bundle: CanonicalizerBundle,
) -> CanonicalStage:
    cfg = bundle.canonicalizer
    quantized_with_warnings = tuple(_quantize_layer(layer, cfg) for layer in raw_stage.layers)
    quantized = tuple(item[0] for item in quantized_with_warnings)
    cells, solid_warnings = build_slab_cells(quantized)
    warnings = tuple(
        sorted(
            set(solid_warnings).union(
                warning
                for _, layer_warnings in quantized_with_warnings
                for warning in layer_warnings
            )
        )
    )
    layers = tuple(
        CanonicalLayer(
            index,
            cell.min_height_q,
            cell.max_height_q,
            cell.footprint_q,
            _geometry_hash(cell.min_height_q, cell.max_height_q, cell.footprint_q),
        )
        for index, cell in enumerate(cells)
    )
    profile = bundle.validation_profile
    if len(layers) > profile.max_layers:
        raise CanonicalizationError("E_LAYER_CAPACITY", "canonical layer 超过 profile 上限", count=len(layers))
    for layer in layers:
        if len(layer.footprint_q) > profile.max_points_per_layer:
            raise CanonicalizationError(
                "E_POINT_CAPACITY",
                "canonical ring 超过 profile 上限",
                layer_index=layer.canonical_layer_index,
                count=len(layer.footprint_q),
            )
    return stage_from_canonical_layers(raw_stage.stage_index, raw_stage.stage_key, layers, cfg, warnings)


def _cells(stage: CanonicalStage) -> tuple[SlabCell, ...]:
    return tuple(SlabCell(layer.min_height_q, layer.max_height_q, layer.footprint_q) for layer in stage.layers)


def _initialize_lineage(stage: CanonicalStage) -> tuple[CanonicalStage, int, dict[int, int]]:
    layers: list[CanonicalLayer] = []
    next_point_by_layer: dict[int, int] = {}
    for layer_index, layer in enumerate(stage.layers):
        point_ids = tuple(range(len(layer.footprint_q)))
        layers.append(replace(layer, layer_lineage_id=layer_index, point_lineage_ids=point_ids))
        next_point_by_layer[layer_index] = len(point_ids)
    return replace(stage, layers=tuple(layers)), len(layers), next_point_by_layer


def _propagate_lineage(
    source: CanonicalStage,
    target: CanonicalStage,
    next_layer_id: int,
    next_point_by_layer: dict[int, int],
    cfg: CanonicalizerConfig,
) -> tuple[CanonicalStage, int, dict[int, int]]:
    matches, match_warnings = match_layers(source.layers, target.layers, cfg)
    source_for_target = {match.target_index: source.layers[match.source_index] for match in matches}
    warnings = set(target.warnings) | set(match_warnings)
    target_layers: list[CanonicalLayer] = []

    for target_layer in target.layers:
        source_layer = source_for_target.get(target_layer.canonical_layer_index)
        if source_layer is None:
            lineage = next_layer_id
            next_layer_id += 1
            new_point_ids = tuple(range(len(target_layer.footprint_q)))
            next_point_by_layer[lineage] = len(new_point_ids)
            target_layers.append(
                replace(target_layer, layer_lineage_id=lineage, point_lineage_ids=new_point_ids)
            )
            continue

        if source_layer.layer_lineage_id is None:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source layer 缺少 lineage")
        lineage = source_layer.layer_lineage_id
        alignment = align_points(source_layer, target_layer.footprint_q, cfg)
        if alignment.warning:
            warnings.add(alignment.warning)
        target_point_ids: list[int | None] = [None] * len(target_layer.footprint_q)
        for point_edit in alignment.edits:
            if point_edit.target_index is None:
                continue
            if point_edit.action in {"KEEP_POINT", "MOVE_POINT"}:
                target_point_ids[point_edit.target_index] = point_edit.point_lineage_id
            elif point_edit.action == "INSERT_POINT":
                target_point_ids[point_edit.target_index] = next_point_by_layer[lineage]
                next_point_by_layer[lineage] += 1
        if any(point_id is None for point_id in target_point_ids):
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "point alignment 未覆盖全部 target 点")
        complete_point_ids = tuple(
            point_id for point_id in target_point_ids if point_id is not None
        )
        target_layers.append(
            replace(
                target_layer,
                layer_lineage_id=lineage,
                point_lineage_ids=complete_point_ids,
            )
        )
    return replace(target, layers=tuple(target_layers), warnings=tuple(sorted(warnings))), next_layer_id, next_point_by_layer


def canonicalize_building_sequence(
    raw: RawBuildingSequence,
    bundle: CanonicalizerBundle,
) -> CanonicalBuildingSequence:
    cfg = bundle.canonicalizer
    if raw.coordinate_frame != cfg.coordinate_frame:
        raise CanonicalizationError(
            "E_COORDINATE_FRAME",
            "输入坐标系与 canonicalizer 配置不一致",
            actual=raw.coordinate_frame,
            expected=cfg.coordinate_frame,
        )
    normalized_key = unicodedata.normalize("NFC", raw.building_key)
    if not normalized_key:
        raise ValueError("building_key 不能为空")
    ordered_raw = tuple(sorted(raw.stages, key=lambda stage: stage.stage_index))
    indices = [stage.stage_index for stage in ordered_raw]
    if len(indices) != len(set(indices)):
        raise CanonicalizationError("E_MISSING_STAGE", "stage_index 重复")
    if raw.expected_stage_indices is not None:
        expected = tuple(sorted(raw.expected_stage_indices))
        if len(expected) != len(set(expected)):
            raise CanonicalizationError("E_MISSING_STAGE", "expected_stage_indices 包含重复值")
        if tuple(indices) != expected:
            raise CanonicalizationError(
                "E_MISSING_STAGE",
                "实际 stage_index 与显式预期不一致",
                actual=tuple(indices),
                expected=expected,
            )

    geometry_stages = [canonicalize_stage(stage, bundle) for stage in ordered_raw]
    if not geometry_stages:
        raise CanonicalizationError("E_MISSING_STAGE", "building sequence 不包含 stage")
    current, next_layer_id, next_point_by_layer = _initialize_lineage(geometry_stages[0])
    stages = [current]
    edits = []
    sequence_warnings: set[str] = set(current.warnings)

    for geometry_target in geometry_stages[1:]:
        removed = removed_volume2(_cells(current), _cells(geometry_target))
        if removed > bundle.validation_profile.removed_volume_tolerance_q3 * 2:
            raise CanonicalizationError(
                "E_CONSTRUCTION_REMOVAL",
                "construction-only sequence 删除了实体体积",
                source_stage=current.stage_index,
                target_stage=geometry_target.stage_index,
                removed_volume2_q3=removed,
            )
        if current.stage_hash == geometry_target.stage_hash:
            geometry_target = replace(
                geometry_target,
                warnings=tuple(sorted(set(geometry_target.warnings) | {"W_NOOP_STAGE"})),
            )
        target, next_layer_id, next_point_by_layer = _propagate_lineage(
            current,
            geometry_target,
            next_layer_id,
            next_point_by_layer,
            cfg,
        )
        edit = build_canonical_edit(current, target, cfg)
        applied = apply_canonical_edit(current, edit, cfg)
        if applied.stage_hash != target.stage_hash:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "sequence 内部 round-trip 失败")
        stages.append(target)
        edits.append(edit)
        sequence_warnings.update(target.warnings)
        current = target

    building_uid = building_uid_from_key(normalized_key)
    sequence_payload = {
        "canonicalizer_version": cfg.canonicalizer_version,
        "canonicalizer_config_hash": cfg.canonicalizer_config_hash,
        "building_key": normalized_key,
        "stage_hashes": [stage.stage_hash for stage in stages],
        "lineage": [
            [
                {
                    "layer_lineage_id": layer.layer_lineage_id,
                    "point_lineage_ids": layer.point_lineage_ids,
                }
                for layer in stage.layers
            ]
            for stage in stages
        ],
        "edit_hashes": [edit.edit_hash for edit in edits],
    }
    return CanonicalBuildingSequence(
        normalized_key,
        building_uid,
        cfg.canonicalizer_version,
        cfg.geometry_version,
        cfg.geometry_config_hash,
        cfg.canonicalizer_config_hash,
        tuple(stages),
        tuple(edits),
        canonical_hash(sequence_payload),
        tuple(sorted(sequence_warnings)),
    )

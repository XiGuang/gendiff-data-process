from __future__ import annotations

from dataclasses import replace

from .config import CanonicalizerConfig
from .errors import CanonicalizationError
from .polygon import cleanup_ring
from .serialize import canonical_hash
from .types import CanonicalEdit, CanonicalLayer, CanonicalStage, LayerEdit, PointEdit


def _point_edits(source: CanonicalLayer, target: CanonicalLayer) -> tuple[PointEdit, ...]:
    target_by_lineage = {lineage: index for index, lineage in enumerate(target.point_lineage_ids)}
    source_lineages = set(source.point_lineage_ids)
    edits: list[PointEdit] = []
    for source_index, lineage in enumerate(source.point_lineage_ids):
        target_index = target_by_lineage.get(lineage)
        if target_index is None:
            edits.append(PointEdit("DELETE_POINT", lineage, source_index, None))
            continue
        if source.footprint_q[source_index] == target.footprint_q[target_index]:
            edits.append(PointEdit("KEEP_POINT", lineage, source_index, target_index))
        else:
            edits.append(
                PointEdit(
                    "MOVE_POINT",
                    lineage,
                    source_index,
                    target_index,
                    target.footprint_q[target_index],
                )
            )
    for target_index, lineage in enumerate(target.point_lineage_ids):
        if lineage not in source_lineages:
            edits.append(PointEdit("INSERT_POINT", lineage, None, target_index, target.footprint_q[target_index]))
    edits.append(PointEdit("EOS", None, None, None))
    return tuple(edits)


def _edit_payload(
    source: CanonicalStage,
    target: CanonicalStage,
    cfg: CanonicalizerConfig,
    layer_edits: tuple[LayerEdit, ...],
) -> dict:
    return {
        "schema_version": "canonical_edit_v3",
        "canonicalizer_version": cfg.canonicalizer_version,
        "geometry_version": cfg.geometry_version,
        "geometry_config_hash": cfg.geometry_config_hash,
        "canonicalizer_config_hash": cfg.canonicalizer_config_hash,
        "source_stage_hash": source.stage_hash,
        "target_stage_hash": target.stage_hash,
        "target_stage_index": target.stage_index,
        "target_stage_key": target.stage_key,
        "layer_edits": layer_edits,
    }


def build_canonical_edit(
    source: CanonicalStage,
    target: CanonicalStage,
    cfg: CanonicalizerConfig,
) -> CanonicalEdit:
    source_by_lineage = {layer.layer_lineage_id: layer for layer in source.layers}
    target_by_lineage = {layer.layer_lineage_id: layer for layer in target.layers}
    if None in source_by_lineage or None in target_by_lineage:
        raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "构建 edit 前必须分配 layer lineage")

    layer_edits: list[LayerEdit] = []
    for source_layer in source.layers:
        if source_layer.layer_lineage_id is None:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source layer 缺少 lineage")
        lineage = source_layer.layer_lineage_id
        target_layer = target_by_lineage.get(lineage)
        if target_layer is None:
            point_edits = tuple(
                PointEdit("DELETE_POINT", point_lineage, index, None)
                for index, point_lineage in enumerate(source_layer.point_lineage_ids)
            ) + (PointEdit("EOS", None, None, None),)
            layer_edits.append(
                LayerEdit(
                    "DELETE_LAYER",
                    lineage,
                    source_layer.canonical_layer_index,
                    None,
                    (source_layer.min_height_q, source_layer.max_height_q),
                    None,
                    point_edits,
                )
            )
            continue
        action = "KEEP_LAYER" if source_layer.geometry_hash == target_layer.geometry_hash else "MODIFY_LAYER"
        layer_edits.append(
            LayerEdit(
                action,
                lineage,
                source_layer.canonical_layer_index,
                target_layer.canonical_layer_index,
                (source_layer.min_height_q, source_layer.max_height_q),
                (target_layer.min_height_q, target_layer.max_height_q),
                _point_edits(source_layer, target_layer),
            )
        )

    source_lineages = set(source_by_lineage)
    for target_layer in target.layers:
        if target_layer.layer_lineage_id is None:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target layer 缺少 lineage")
        lineage = target_layer.layer_lineage_id
        if lineage in source_lineages:
            continue
        point_edits = tuple(
            PointEdit("INSERT_POINT", point_lineage, None, index, target_layer.footprint_q[index])
            for index, point_lineage in enumerate(target_layer.point_lineage_ids)
        ) + (PointEdit("EOS", None, None, None),)
        layer_edits.append(
            LayerEdit(
                "INSERT_LAYER",
                lineage,
                None,
                target_layer.canonical_layer_index,
                None,
                (target_layer.min_height_q, target_layer.max_height_q),
                point_edits,
            )
        )

    ordered = tuple(layer_edits)
    edit_hash = canonical_hash(_edit_payload(source, target, cfg, ordered))
    return CanonicalEdit(
        source.stage_hash,
        target.stage_hash,
        target.stage_index,
        target.stage_key,
        cfg.canonicalizer_config_hash,
        ordered,
        edit_hash,
    )


def apply_canonical_edit(
    source: CanonicalStage,
    edit: CanonicalEdit,
    cfg: CanonicalizerConfig,
) -> CanonicalStage:
    if edit.canonicalizer_config_hash != cfg.canonicalizer_config_hash:
        raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "edit 的 canonicalizer config hash 不匹配")
    if edit.source_stage_hash != source.stage_hash:
        raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "edit 的 source hash 与输入不一致")
    expected_edit_hash = canonical_hash(
        {
            "schema_version": "canonical_edit_v3",
            "canonicalizer_version": cfg.canonicalizer_version,
            "geometry_version": cfg.geometry_version,
            "geometry_config_hash": cfg.geometry_config_hash,
            "canonicalizer_config_hash": cfg.canonicalizer_config_hash,
            "source_stage_hash": edit.source_stage_hash,
            "target_stage_hash": edit.target_stage_hash,
            "target_stage_index": edit.target_stage_index,
            "target_stage_key": edit.target_stage_key,
            "layer_edits": edit.layer_edits,
        }
    )
    if edit.edit_hash != expected_edit_hash:
        raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "edit hash 与内容不一致")

    source_by_index = {layer.canonical_layer_index: layer for layer in source.layers}
    reconstructed: list[CanonicalLayer] = []
    seen_source_layers: set[int] = set()
    seen_target_layers: set[int] = set()
    allowed_layer_actions = {"KEEP_LAYER", "MODIFY_LAYER", "DELETE_LAYER", "INSERT_LAYER"}
    allowed_point_actions = {"KEEP_POINT", "MOVE_POINT", "DELETE_POINT", "INSERT_POINT", "EOS"}

    for layer_edit in edit.layer_edits:
        if layer_edit.action not in allowed_layer_actions:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "未知 layer action", action=layer_edit.action)
        source_layer = None
        if layer_edit.source_layer_index is not None:
            if layer_edit.source_layer_index in seen_source_layers:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source layer index 重复")
            seen_source_layers.add(layer_edit.source_layer_index)
            source_layer = source_by_index.get(layer_edit.source_layer_index)
            if source_layer is None:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source layer index 越界")
            if source_layer.layer_lineage_id != layer_edit.layer_lineage_id:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source layer lineage 不匹配")
        if layer_edit.action == "INSERT_LAYER" and source_layer is not None:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "INSERT_LAYER 不得引用 source layer")
        if layer_edit.action != "INSERT_LAYER" and source_layer is None:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source-backed layer action 缺少 source layer")
        if layer_edit.target_layer_index is not None:
            if layer_edit.target_layer_index in seen_target_layers:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target layer index 重复")
            seen_target_layers.add(layer_edit.target_layer_index)

        point_actions = [point_edit.action for point_edit in layer_edit.point_edits]
        if any(action not in allowed_point_actions for action in point_actions):
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "未知 point action")
        if point_actions.count("EOS") != 1 or not point_actions or point_actions[-1] != "EOS":
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "每个 layer edit 必须以唯一 EOS 结尾")

        if layer_edit.action == "DELETE_LAYER":
            if layer_edit.target_layer_index is not None or layer_edit.target_height_q is not None:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "DELETE_LAYER 含 target metadata")
            if any(action not in {"DELETE_POINT", "EOS"} for action in point_actions):
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "DELETE_LAYER 含非删除 point action")
            continue
        if layer_edit.target_height_q is None or layer_edit.target_layer_index is None:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "非 DELETE layer 缺少 target metadata")
        if layer_edit.target_height_q[1] <= layer_edit.target_height_q[0]:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target layer 高度无效")

        target_points: dict[int, tuple[tuple[int, int], int]] = {}
        seen_source_points: set[int] = set()
        for point_edit in layer_edit.point_edits:
            if point_edit.action == "EOS":
                continue
            if point_edit.point_lineage_id is None:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "point edit 缺少 lineage")
            if point_edit.source_index is not None:
                if source_layer is None or not 0 <= point_edit.source_index < len(source_layer.footprint_q):
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source point index 越界")
                if point_edit.source_index in seen_source_points:
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source point index 重复")
                seen_source_points.add(point_edit.source_index)
                if source_layer.point_lineage_ids[point_edit.source_index] != point_edit.point_lineage_id:
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source point lineage 不匹配")
            if point_edit.action in {"KEEP_POINT", "MOVE_POINT", "DELETE_POINT"} and point_edit.source_index is None:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "source-backed point action 缺少 source index")
            if point_edit.action == "INSERT_POINT" and point_edit.source_index is not None:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "INSERT_POINT 不得引用 source point")
            if point_edit.action == "DELETE_POINT":
                if point_edit.target_index is not None or point_edit.target_coord_q is not None:
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "DELETE_POINT 含 target metadata")
                continue
            if point_edit.target_index is None:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target point 缺少 index")
            if point_edit.action == "KEEP_POINT":
                if point_edit.target_coord_q is not None:
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "KEEP_POINT 不得携带 target coordinate")
                if source_layer is None or point_edit.source_index is None:
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "KEEP_POINT 缺少 source point")
                coordinate = source_layer.footprint_q[point_edit.source_index]
            else:
                if point_edit.target_coord_q is None:
                    raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "MOVE/INSERT 缺少 target coordinate")
                coordinate = point_edit.target_coord_q
            if point_edit.target_index in target_points:
                raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target point index 重复")
            target_points[point_edit.target_index] = (coordinate, point_edit.point_lineage_id)

        ordered_indices = sorted(target_points)
        if ordered_indices != list(range(len(ordered_indices))):
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target point index 不连续")
        raw_ring = tuple(target_points[index][0] for index in ordered_indices)
        canonical_ring = cleanup_ring(raw_ring)
        lineage_by_coordinate = {target_points[index][0]: target_points[index][1] for index in ordered_indices}
        try:
            canonical_lineages = tuple(lineage_by_coordinate[point] for point in canonical_ring)
        except KeyError as exc:
            raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "规范环与 edit target 点不一致") from exc
        geometry_hash = canonical_hash(
            {
                "min_height_q": layer_edit.target_height_q[0],
                "max_height_q": layer_edit.target_height_q[1],
                "footprint_q": canonical_ring,
            }
        )
        reconstructed.append(
            CanonicalLayer(
                layer_edit.target_layer_index,
                layer_edit.target_height_q[0],
                layer_edit.target_height_q[1],
                canonical_ring,
                geometry_hash,
                layer_edit.layer_lineage_id,
                canonical_lineages,
            )
        )

    if seen_source_layers != set(source_by_index):
        raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "edit 未覆盖全部 source layer")
    if sorted(seen_target_layers) != list(range(len(seen_target_layers))):
        raise CanonicalizationError("E_ROUNDTRIP_MISMATCH", "target layer index 不连续")

    from .core import stage_from_canonical_layers

    stage = stage_from_canonical_layers(
        edit.target_stage_index,
        edit.target_stage_key,
        tuple(reconstructed),
        cfg,
    )
    if stage.stage_hash != edit.target_stage_hash:
        raise CanonicalizationError(
            "E_ROUNDTRIP_MISMATCH",
            "apply 后的 stage hash 与 target 不一致",
            actual=stage.stage_hash,
            expected=edit.target_stage_hash,
        )
    return replace(stage, warnings=())

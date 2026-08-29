from __future__ import annotations

from collections.abc import Mapping

from .errors import CanonicalizationError
from .serialize import canonical_hash

PACK_SCHEMA = "area_v2_packed_v1"
EDIT_SCHEMA = "area_v2_absolute_target_coord_no_anchor"
REQUIRED_CANONICAL_FIELDS = (
    "canonicalizer_version",
    "geometry_version",
    "geometry_config_hash",
    "canonicalizer_config_hash",
    "validation_profile_hash",
    "condition_config_hash",
    "normalization_config_hash",
    "split_config_hash",
    "package_config_hash",
    "normalization_profile_id",
)
REQUIRED_SAMPLE_FIELDS = REQUIRED_CANONICAL_FIELDS + (
    "source_stage_hash",
    "target_stage_hash",
    "edit_hash",
    "condition_hash",
    "building_uid",
    "split",
)
DIRECTIONAL_SAMPLE_FIELDS = ("change_kind", "pair_hash")
ALLOWED_CHANGE_KINDS = {"construction", "demolition", "mixed"}


def canonical_pair_hash(metadata: Mapping) -> str:
    return canonical_hash(
        {
            "schema_version": "canonical_pair_v1",
            "source_stage_hash": metadata.get("source_stage_hash"),
            "target_stage_hash": metadata.get("target_stage_hash"),
            "edit_hash": metadata.get("edit_hash"),
            "condition_hash": metadata.get("condition_hash"),
            "change_kind": metadata.get("change_kind"),
        }
    )


def _require_fields(mapping: Mapping, fields: tuple[str, ...], location: str) -> None:
    missing = [field for field in fields if not mapping.get(field)]
    if missing:
        raise CanonicalizationError(
            "E_PACKED_CANONICAL_METADATA",
            f"{location} 缺少 canonical metadata",
            missing=missing,
        )


def validate_packed_release_meta(meta: Mapping) -> None:
    if meta.get("schema_version") != PACK_SCHEMA:
        raise CanonicalizationError("E_PACKED_SCHEMA", "packed schema 不兼容")
    if meta.get("edit_schema_version") != EDIT_SCHEMA:
        raise CanonicalizationError("E_PACKED_SCHEMA", "edit schema 不兼容")
    contract = meta.get("canonical_contract")
    if not isinstance(contract, Mapping):
        raise CanonicalizationError(
            "E_PACKED_CANONICAL_METADATA", "缺少 canonical_contract"
        )
    _require_fields(contract, REQUIRED_CANONICAL_FIELDS, "dataset_meta")


def validate_packed_sample(sample: Mapping) -> None:
    metadata = sample.get("canonical_metadata")
    if not isinstance(metadata, Mapping):
        raise CanonicalizationError(
            "E_PACKED_CANONICAL_METADATA", "sample 缺少 canonical_metadata"
        )
    _require_fields(metadata, REQUIRED_SAMPLE_FIELDS, "sample")
    if not any(field in metadata for field in DIRECTIONAL_SAMPLE_FIELDS):
        return
    _require_fields(metadata, DIRECTIONAL_SAMPLE_FIELDS, "directional sample")
    if metadata.get("change_kind") not in ALLOWED_CHANGE_KINDS:
        raise CanonicalizationError(
            "E_PACKED_CANONICAL_METADATA",
            "sample change_kind 未被 canonical pair schema 定义",
            change_kind=metadata.get("change_kind"),
        )
    if metadata.get("pair_hash") != canonical_pair_hash(metadata):
        raise CanonicalizationError(
            "E_PACKED_CANONICAL_METADATA", "sample pair_hash 与内容不一致"
        )

"""Unified Canonicalizer v1 的公共接口。"""

from .config import CanonicalizerBundle, load_bundle
from .core import canonicalize_building_sequence, canonicalize_stage
from .edit_v3 import apply_canonical_edit, build_canonical_edit
from .errors import CanonicalizationError
from .release_contracts import (
    assign_building_splits,
    building_uid_from_key,
    compute_train_normalization_profile,
)
from .types import RawBuildingSequence, RawLayer, RawStage

__all__ = [
    "CanonicalizationError",
    "CanonicalizerBundle",
    "RawBuildingSequence",
    "RawLayer",
    "RawStage",
    "apply_canonical_edit",
    "assign_building_splits",
    "building_uid_from_key",
    "build_canonical_edit",
    "canonicalize_building_sequence",
    "canonicalize_stage",
    "compute_train_normalization_profile",
    "load_bundle",
]

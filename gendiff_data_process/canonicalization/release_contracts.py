from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Iterable, Mapping

from .config import NormalizationConfig, SplitConfig
from .errors import CanonicalizationError
from .serialize import canonical_hash
from .types import CanonicalBuildingSequence


def building_uid_from_key(building_key: str) -> str:
    normalized = unicodedata.normalize("NFC", building_key)
    if not normalized:
        raise ValueError("building_key 不能为空")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def split_for_building_uid(building_uid: str, config: SplitConfig) -> str:
    payload = f"{config.algorithm}\0{config.seed}\0{building_uid}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % config.modulus
    if bucket < config.train_threshold:
        return "train"
    if bucket < config.validation_threshold:
        return "val"
    return "test"


def assign_building_splits(
    building_keys: Iterable[str],
    config: SplitConfig,
) -> dict[str, str]:
    uid_to_key: dict[str, str] = {}
    assignments: dict[str, str] = {}
    for building_key in sorted(set(building_keys)):
        building_uid = building_uid_from_key(building_key)
        previous = uid_to_key.get(building_uid)
        if previous is not None and previous != building_key:
            raise CanonicalizationError(
                "E_BUILDING_UID_COLLISION",
                "split 前发现截断 building UID 冲突",
                building_uid=building_uid,
                previous_building_key=previous,
                current_building_key=building_key,
            )
        uid_to_key[building_uid] = building_key
        assignments[building_key] = split_for_building_uid(building_uid, config)
    return assignments


@dataclass(frozen=True)
class NormalizationProfile:
    profile_id: str
    method_config_hash: str
    train_building_uids: tuple[str, ...]
    min_x_q: int
    max_x_q: int
    min_z_q: int
    max_z_q: int
    min_y_q: int
    max_y_q: int
    center_x: float
    center_z: float
    scale_xz: float
    center_y: float
    scale_y: float

    @property
    def tensor_order(self) -> tuple[float, ...]:
        return (self.center_x, self.center_z, self.scale_xz, self.center_y, self.scale_y)

    def to_mapping(self) -> dict:
        return asdict(self)


def _world_center(minimum_q: int, maximum_q: int, grid: str) -> float:
    return float(Decimal(minimum_q + maximum_q) * Decimal(grid) / Decimal(2))


def compute_train_normalization_profile(
    sequences: Iterable[CanonicalBuildingSequence],
    config: NormalizationConfig,
    *,
    grid_xz: str,
    grid_y: str,
) -> NormalizationProfile:
    ordered = tuple(sorted(sequences, key=lambda sequence: sequence.building_uid))
    train_uids = tuple(sequence.building_uid for sequence in ordered)
    if len(train_uids) != len(set(train_uids)):
        raise CanonicalizationError("E_BUILDING_UID_COLLISION", "normalization train 集含重复 building UID")

    x_values: list[int] = []
    z_values: list[int] = []
    y_values: list[int] = []
    for sequence in ordered:
        for stage in sequence.stages:
            for layer in stage.layers:
                x_values.extend(point[0] for point in layer.footprint_q)
                z_values.extend(point[1] for point in layer.footprint_q)
                y_values.extend((layer.min_height_q, layer.max_height_q))
    if not x_values or not z_values or not y_values:
        raise CanonicalizationError(
            "E_NORMALIZATION_PROFILE",
            "train-only normalization 没有可用 canonical geometry",
        )

    min_x_q, max_x_q = min(x_values), max(x_values)
    min_z_q, max_z_q = min(z_values), max(z_values)
    min_y_q, max_y_q = min(y_values), max(y_values)
    minimum_scale_q = max(
        1,
        int((Decimal(config.min_scale) / min(Decimal(grid_xz), Decimal(grid_y))).to_integral_value(rounding=ROUND_CEILING)),
    )
    scale_q = max(
        max_x_q - min_x_q,
        max_z_q - min_z_q,
        max_y_q - min_y_q,
        minimum_scale_q,
    )
    if Decimal(grid_xz) != Decimal(grid_y):
        raise CanonicalizationError(
            "E_NORMALIZATION_PROFILE",
            "uniform v1 要求 XZ 和 Y 使用相同 grid",
            grid_xz=grid_xz,
            grid_y=grid_y,
        )
    uniform_scale = float(Decimal(scale_q) * Decimal(grid_xz))
    profile_payload: Mapping[str, object] = {
        "method_config_hash": config.config_hash,
        "train_building_uids": train_uids,
        "extrema_q": {
            "min_x_q": min_x_q,
            "max_x_q": max_x_q,
            "min_z_q": min_z_q,
            "max_z_q": max_z_q,
            "min_y_q": min_y_q,
            "max_y_q": max_y_q,
        },
        "grid_xz": grid_xz,
        "grid_y": grid_y,
        "scale_q": scale_q,
    }
    profile_hash = canonical_hash(profile_payload)
    return NormalizationProfile(
        profile_id=f"{config.profile_id_prefix}_{profile_hash[:16]}",
        method_config_hash=config.config_hash,
        train_building_uids=train_uids,
        min_x_q=min_x_q,
        max_x_q=max_x_q,
        min_z_q=min_z_q,
        max_z_q=max_z_q,
        min_y_q=min_y_q,
        max_y_q=max_y_q,
        center_x=_world_center(min_x_q, max_x_q, grid_xz),
        center_z=_world_center(min_z_q, max_z_q, grid_xz),
        scale_xz=uniform_scale,
        center_y=_world_center(min_y_q, max_y_q, grid_y),
        scale_y=uniform_scale,
    )

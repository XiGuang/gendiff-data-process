from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .errors import CanonicalizationError


@dataclass(frozen=True)
class CollisionReport:
    unique_key_count: int
    duplicate_row_count: int
    conflicting_key_count: int


@dataclass(frozen=True)
class BuildingUidReport:
    unique_uid_count: int
    duplicate_building_count: int


def audit_supervision_collisions(records: Iterable[Mapping[str, str]]) -> CollisionReport:
    observed: dict[tuple[str, str], tuple[str, str]] = {}
    duplicate_count = 0
    conflicts: list[dict[str, object]] = []
    for index, record in enumerate(records):
        key = (str(record["source_stage_hash"]), str(record["condition_hash"]))
        supervision = (str(record["target_stage_hash"]), str(record["edit_hash"]))
        previous = observed.get(key)
        if previous is None:
            observed[key] = supervision
        elif previous == supervision:
            duplicate_count += 1
        else:
            conflicts.append(
                {
                    "record_index": index,
                    "source_stage_hash": key[0],
                    "condition_hash": key[1],
                    "previous": previous,
                    "current": supervision,
                }
            )
    if conflicts:
        raise CanonicalizationError(
            "E_SUPERVISION_COLLISION",
            "同一 canonical key 对应冲突 supervision",
            conflicts=conflicts,
        )
    return CollisionReport(len(observed), duplicate_count, 0)


def audit_building_uid_collisions(records: Iterable[Mapping[str, str]]) -> BuildingUidReport:
    observed: dict[str, str] = {}
    duplicate_count = 0
    for index, record in enumerate(records):
        building_uid = str(record["building_uid"])
        building_key = str(record["building_key"])
        previous = observed.get(building_uid)
        if previous is None:
            observed[building_uid] = building_key
        elif previous == building_key:
            duplicate_count += 1
        else:
            raise CanonicalizationError(
                "E_BUILDING_UID_COLLISION",
                "截断 building UID 对应多个 building key",
                record_index=index,
                building_uid=building_uid,
                previous_building_key=previous,
                current_building_key=building_key,
            )
    return BuildingUidReport(len(observed), duplicate_count)

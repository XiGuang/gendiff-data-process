from __future__ import annotations

from pathlib import Path

import yaml

from gendiff_data_process.canonicalization.core import stage_from_canonical_layers
from gendiff_data_process.canonicalization.edit_v3 import build_canonical_edit
from gendiff_data_process.canonicalization.serialize import canonical_hash
from gendiff_data_process.canonicalization.types import (
    CanonicalBuildingSequence,
    CanonicalLayer,
    RawBuildingSequence,
    RawLayer,
    RawStage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/canonicalizer"
CONFIG_PATH = REPO_ROOT / "configs/canonicalizer_v1.yaml"


def load_raw_sequence(path: str | Path) -> RawBuildingSequence:
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stages = []
    for stage_mapping in mapping["stages"]:
        layers = tuple(RawLayer.from_mapping(layer) for layer in stage_mapping.get("layers") or [])
        stages.append(
            RawStage(
                int(stage_mapping["stage_index"]),
                str(stage_mapping.get("stage_key", f"stage_{stage_mapping['stage_index']}")),
                layers,
            )
        )
    expected = mapping.get("expected_stage_indices")
    return RawBuildingSequence(
        str(mapping["building_key"]),
        str(mapping["coordinate_frame"]),
        tuple(stages),
        tuple(int(index) for index in expected) if expected is not None else None,
    )


def make_all_actions_sequence(bundle) -> CanonicalBuildingSequence:
    cfg = bundle.canonicalizer
    source_layer = CanonicalLayer(
        0,
        0,
        1000,
        ((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
        "source_geometry",
        0,
        (0, 1, 2, 3),
    )
    target_layer = CanonicalLayer(
        0,
        0,
        2000,
        ((0, 0), (1100, 0), (1500, 500), (1000, 1000)),
        "target_geometry",
        0,
        (0, 1, 4, 2),
    )
    inserted_layer = CanonicalLayer(
        1,
        0,
        1000,
        ((3000, 0), (4000, 0), (4000, 1000), (3000, 1000)),
        "inserted_geometry",
        1,
        (0, 1, 2, 3),
    )
    source = stage_from_canonical_layers(0, "stage_0", (source_layer,), cfg)
    target = stage_from_canonical_layers(1, "stage_1", (target_layer, inserted_layer), cfg)
    edit = build_canonical_edit(source, target, cfg)
    return CanonicalBuildingSequence(
        "SYN_ADAPTER_001",
        "0" * 32,
        cfg.canonicalizer_version,
        cfg.geometry_version,
        cfg.geometry_config_hash,
        cfg.canonicalizer_config_hash,
        (source, target),
        (edit,),
        canonical_hash({"fixture": "SYN_ADAPTER_001"}),
    )

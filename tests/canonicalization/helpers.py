from __future__ import annotations

import copy
from pathlib import Path

import yaml

from gendiff_data_process.canonicalization.adapters.area_v2 import (
    AreaNormalizationStats,
    AreaV2Adapter,
)
from gendiff_data_process.canonicalization.core import stage_from_canonical_layers
from gendiff_data_process.canonicalization.edit_v3 import build_canonical_edit
from gendiff_data_process.canonicalization.packed_contract import (
    validate_packed_release_meta,
    validate_packed_sample,
)
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
BIDIRECTIONAL_CONFIG_PATH = REPO_ROOT / "configs/canonicalizer_bidirectional_v1.yaml"


def load_raw_sequence(path: str | Path) -> RawBuildingSequence:
    mapping = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stages = []
    for stage_mapping in mapping["stages"]:
        layers = tuple(
            RawLayer.from_mapping(layer) for layer in stage_mapping.get("layers") or []
        )
        stages.append(
            RawStage(
                int(stage_mapping["stage_index"]),
                str(
                    stage_mapping.get(
                        "stage_key", f"stage_{stage_mapping['stage_index']}"
                    )
                ),
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
    target = stage_from_canonical_layers(
        1, "stage_1", (target_layer, inserted_layer), cfg
    )
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


def make_layer_delete_sequence(bundle) -> CanonicalBuildingSequence:
    cfg = bundle.canonicalizer
    retained = CanonicalLayer(
        0,
        0,
        1000,
        ((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
        "retained_geometry",
        0,
        (0, 1, 2, 3),
    )
    deleted = CanonicalLayer(
        1,
        0,
        1000,
        ((2000, 0), (3000, 0), (3000, 1000), (2000, 1000)),
        "deleted_geometry",
        1,
        (0, 1, 2, 3),
    )
    source = stage_from_canonical_layers(0, "stage_0", (retained, deleted), cfg)
    target = stage_from_canonical_layers(1, "stage_1", (retained,), cfg)
    edit = build_canonical_edit(source, target, cfg)
    return CanonicalBuildingSequence(
        "SYN_LAYER_DELETE_001",
        "1" * 32,
        cfg.canonicalizer_version,
        cfg.geometry_version,
        cfg.geometry_config_hash,
        cfg.canonicalizer_config_hash,
        (source, target),
        (edit,),
        canonical_hash({"fixture": "SYN_LAYER_DELETE_001"}),
    )


def write_bidirectional_packed_fixture(
    root: Path,
    bundle,
    normalization: AreaNormalizationStats,
) -> dict:
    import torch

    adapter = AreaV2Adapter(bundle, normalization)
    base_pair = adapter.adapt_pair(
        make_all_actions_sequence(bundle),
        0,
        [[0.0, 0.0, 0.0]] * bundle.condition_sampling.point_count,
        condition_hash="fixture_condition_hash",
        split="train",
    )
    demolition_sequence = make_layer_delete_sequence(bundle)
    demolition_pair = adapter.adapt_pair(
        demolition_sequence,
        0,
        [[0.0, 0.0, 0.0]] * bundle.condition_sampling.point_count,
        condition_hash="fixture_demolition_condition_hash",
        split="train",
    )
    reverse_edit = build_canonical_edit(
        demolition_sequence.stages[1],
        demolition_sequence.stages[0],
        bundle.canonicalizer,
    )
    construction_pair = adapter.adapt_transition(
        demolition_sequence,
        1,
        0,
        reverse_edit,
        [[0.0, 0.0, 0.0]] * bundle.condition_sampling.point_count,
        condition_hash="fixture_construction_condition_hash",
        split="train",
    )
    states = []
    split_samples = {}
    pair_templates = (base_pair, demolition_pair, construction_pair)
    for split_index, (split, pair_template) in enumerate(
        zip(("train", "val", "test"), pair_templates)
    ):
        pair = copy.deepcopy(pair_template)
        building_key = f"fixture_building_{split_index}"
        pair["source_state"]["meta"]["building_key"] = building_key
        pair["target_state"]["meta"]["building_key"] = building_key
        source_index = len(states)
        states.extend((pair["source_state"], pair["target_state"]))
        sample = pair["sample"]
        sample["source_state_index"] = source_index
        sample["target_state_index"] = source_index + 1
        sample["pair_name"] = f"{building_key}_pair"
        sample["canonical_metadata"]["building_uid"] = f"fixture_uid_{split_index}"
        sample["canonical_metadata"]["split"] = split
        sample["validation"]["canonical_metadata"] = sample["canonical_metadata"]
        sample["condition"] = torch.tensor(sample["condition"], dtype=torch.float32)
        validate_packed_sample(sample)
        split_samples[split] = sample

    meta = {
        "schema_version": base_pair["pack_schema_version"],
        "edit_schema_version": base_pair["edit_schema_version"],
        "states_path": str(root / "states.pt"),
        "canonical_contract": adapter.canonical_contract(),
        "split_sample_counts": {split: 1 for split in split_samples},
    }
    validate_packed_release_meta(meta)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(meta, root / "dataset_meta.pt")
    with (root / "dataset_meta.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, allow_unicode=True, sort_keys=False)
    torch.save(
        {
            "schema_version": base_pair["pack_schema_version"],
            "states": states,
            "normalization_stats_tensor": base_pair["normalization_stats_tensor"],
        },
        root / "states.pt",
    )
    for split, sample in split_samples.items():
        shard_dir = root / "shards" / split
        shard_dir.mkdir(parents=True)
        shard_path = shard_dir / f"{split}_00000.pt"
        torch.save(
            {
                "schema_version": base_pair["pack_schema_version"],
                "sample_count": 1,
                "sample_offset": 0,
                "samples": [sample],
            },
            shard_path,
        )
        index = {
            "shards": [{"path": str(shard_path), "split": split, "sample_count": 1}]
        }
        torch.save(index, root / f"{split}_index.pt")
        with (root / f"{split}_index.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(index, handle, allow_unicode=True, sort_keys=False)
    return meta

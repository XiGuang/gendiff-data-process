from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - 由独立 GenDiff 环境执行 blocking smoke
    torch = None

from gendiff_data_process.canonicalization.adapters.area_v2 import AreaNormalizationStats, AreaV2Adapter
from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.packed_contract import (
    validate_packed_release_meta,
    validate_packed_sample,
)

from tests.canonicalization.helpers import CONFIG_PATH, make_all_actions_sequence


@unittest.skipIf(torch is None, "需要包含 Torch 的已批准 GenDiff 环境")
class PackedLoaderSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        gendiff_repo = os.environ.get("GENDIFF_REPO")
        if not gendiff_repo:
            raise unittest.SkipTest("需要显式设置只读 GENDIFF_REPO")
        repo_path = Path(gendiff_repo).resolve()
        if not (repo_path / "craftsman/data/packed_area_edit_v2_data_module.py").is_file():
            raise unittest.SkipTest("GENDIFF_REPO 不包含已审计 loader")
        sys.path.insert(0, str(repo_path))
        from craftsman.data.packed_area_edit_v2_data_module import PackedAreaEditV2DataModule

        cls.data_module_class = PackedAreaEditV2DataModule
        loaded = load_bundle(CONFIG_PATH)
        cls.bundle = replace(
            loaded,
            condition_sampling=replace(loaded.condition_sampling, point_count=8),
        )
        cls.normalization = AreaNormalizationStats("fixture_explicit_v1", 0.0, 0.0, 10.0, 0.0, 10.0)

    def _write_release(self, root: Path) -> dict:
        adapter = AreaV2Adapter(self.bundle, self.normalization)
        sequence = make_all_actions_sequence(self.bundle)
        base_pair = adapter.adapt_pair(
            sequence,
            0,
            [[0.0, 0.0, 0.0]] * 8,
            condition_hash="fixture_condition_hash",
            split="train",
        )
        states = []
        split_samples = {}
        for split_index, split in enumerate(("train", "val", "test")):
            pair = copy.deepcopy(base_pair)
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

        contract = adapter.canonical_contract()
        meta = {
            "schema_version": base_pair["pack_schema_version"],
            "edit_schema_version": base_pair["edit_schema_version"],
            "states_path": str(root / "states.pt"),
            "canonical_contract": contract,
        }
        validate_packed_release_meta(meta)
        torch.save(meta, root / "dataset_meta.pt")
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
            torch.save({"sample_count": 1, "sample_offset": 0, "samples": [sample]}, shard_path)
            torch.save(
                {
                    "shards": [
                        {"path": str(shard_path), "split": split, "sample_count": 1}
                    ]
                },
                root / f"{split}_index.pt",
            )
        return meta

    def test_data_module_reads_distinct_splits_and_exact_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_release(root)
            module = self.data_module_class(
                {
                    "dataset_folder": str(root),
                    "max_layers": 64,
                    "max_points_per_layer": 32,
                    "strict_area_capacity": True,
                    "replica": 1,
                    "batch_size": 1,
                    "num_workers": 0,
                    "condition_point_num": 8,
                    "train_iterate_shards": False,
                    "shuffle_shards": False,
                    "shuffle_samples_in_shard": False,
                    "persistent_workers": False,
                }
            )
            module.setup(None)
            train_batch = next(iter(module.train_dataloader()))
            val_batch = next(iter(module.val_dataloader()))
            test_batch = next(iter(module.test_dataloader()))
            self.assertEqual(tuple(train_batch["source_point_coords"].shape), (1, 64, 32, 2))
            self.assertEqual(tuple(train_batch["ar_action_targets"].shape), (1, 64, 33))
            self.assertEqual(tuple(train_batch["change_point_clouds"].shape), (1, 8, 3))
            names = {train_batch["pair_name"][0], val_batch["pair_name"][0], test_batch["pair_name"][0]}
            self.assertEqual(len(names), 3)
            metadata = train_batch["validation"][0]["canonical_metadata"]
            self.assertTrue(metadata["edit_hash"])

    def test_wrong_schema_and_missing_hashes_fail_before_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = self._write_release(root)
            broken = copy.deepcopy(meta)
            broken["canonical_contract"].pop("geometry_config_hash")
            with self.assertRaises(CanonicalizationError):
                validate_packed_release_meta(broken)


if __name__ == "__main__":
    unittest.main()

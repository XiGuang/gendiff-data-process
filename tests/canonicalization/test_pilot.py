from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.history_adapter import load_history_sequence
from gendiff_data_process.canonicalization.pilot import (
    PilotTask,
    building_keys_from_range,
    make_pilot_tasks,
    pair_accounting,
    pilot_fingerprint,
    process_pilot_tasks,
)

from tests.canonicalization.helpers import BIDIRECTIONAL_CONFIG_PATH, CONFIG_PATH


class PilotTests(unittest.TestCase):
    @staticmethod
    def _write_history(root: Path, building_key: str) -> None:
        building = root / building_key
        for stage_index, layers in (
            (0, []),
            (
                1,
                [
                    {
                        "proxy_id": 7,
                        "min_height": 0,
                        "max_height": 1,
                        "footprint": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
            ),
        ):
            stage = building / f"stage_{stage_index}"
            stage.mkdir(parents=True)
            (stage / f"stage_{stage_index}.yaml").write_text(
                yaml.safe_dump(layers, sort_keys=True),
                encoding="utf-8",
            )

    @staticmethod
    def _write_mixed_history(root: Path, building_key: str) -> None:
        building = root / building_key
        for stage_index, offset in ((0, 0.0), (1, 0.5)):
            stage = building / f"stage_{stage_index}"
            stage.mkdir(parents=True)
            layers = [
                {
                    "proxy_id": 7,
                    "min_height": 0,
                    "max_height": 1,
                    "footprint": [
                        [offset, 0],
                        [offset + 1, 0],
                        [offset + 1, 1],
                        [offset, 1],
                    ],
                }
            ]
            (stage / f"stage_{stage_index}.yaml").write_text(
                yaml.safe_dump(layers, sort_keys=True),
                encoding="utf-8",
            )

    def test_history_adapter_reads_only_explicit_stages_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_history(root, "building_0001")
            loaded = load_history_sequence(root, "building_0001", (0, 1))
            self.assertEqual(loaded.sequence.expected_stage_indices, (0, 1))
            self.assertEqual(len(loaded.sequence.stages[0].layers), 0)
            self.assertEqual(len(loaded.sequence.stages[1].layers), 1)
            self.assertEqual(len(loaded.source_files), 2)
            self.assertTrue(all(len(item.sha256) == 64 for item in loaded.source_files))

    def test_fingerprint_matches_worker_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_history(root, "building_0001")
            task = PilotTask(
                str(root),
                "building_0001",
                "test",
                (0, 1),
                str(CONFIG_PATH),
            )
            serial = process_pilot_tasks((task,), workers=1)
            parallel = process_pilot_tasks((task,), workers=2)
            self.assertEqual(pilot_fingerprint(serial), pilot_fingerprint(parallel))
            self.assertIsNotNone(serial[0].sequence)
            self.assertEqual(len(serial[0].pairs), 1)

            failed = replace(
                serial[0],
                sequence=None,
                pairs=(),
                error_code="E_TEST_FAILURE",
            )
            accounting = pair_accounting(
                (serial[0], failed),
                (0, 1),
                emitted_sample_count=1,
                duplicate_row_count=0,
                conflicting_row_count=0,
            )
            self.assertEqual(accounting["attempted"], 2)
            self.assertEqual(accounting["failed_building_pair_slots"], 1)
            self.assertEqual(accounting["explicit_failures"], 1)
            self.assertEqual(accounting["silent_drop_count"], 0)

    def test_bidirectional_profile_emits_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_history(root, "building_0001")
            task = PilotTask(
                str(root),
                "building_0001",
                "test",
                (0, 1),
                str(BIDIRECTIONAL_CONFIG_PATH),
            )
            result = process_pilot_tasks((task,), workers=1)[0]
            self.assertIsNotNone(result.sequence)
            self.assertEqual(
                [pair.change_kind for pair in result.pairs],
                ["construction", "demolition"],
            )
            self.assertEqual(result.pair_failures, ())
            accounting = pair_accounting(
                (result,),
                (0, 1),
                emitted_sample_count=2,
                duplicate_row_count=0,
                conflicting_row_count=0,
                directions_per_transition=2,
            )
            self.assertEqual(accounting["attempted"], 2)
            self.assertEqual(accounting["silent_drop_count"], 0)

    def test_bidirectional_profile_accounts_mixed_pair_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_mixed_history(root, "building_0001")
            task = PilotTask(
                str(root),
                "building_0001",
                "test",
                (0, 1),
                str(BIDIRECTIONAL_CONFIG_PATH),
            )
            result = process_pilot_tasks((task,), workers=1)[0]
            self.assertIsNotNone(result.sequence)
            self.assertEqual(result.pairs, ())
            self.assertEqual(len(result.pair_failures), 2)
            self.assertEqual(
                {failure.error_code for failure in result.pair_failures},
                {"E_MIXED_CHANGE_UNSUPPORTED"},
            )
            accounting = pair_accounting(
                (result,),
                (0, 1),
                emitted_sample_count=0,
                duplicate_row_count=0,
                conflicting_row_count=0,
                directions_per_transition=2,
            )
            self.assertEqual(accounting["explicit_failures"], 2)
            self.assertEqual(accounting["silent_drop_count"], 0)

    def test_building_range_is_bounded(self) -> None:
        self.assertEqual(len(building_keys_from_range(1, 100)), 100)
        with self.assertRaises(ValueError):
            building_keys_from_range(1, 101)
        with self.assertRaises(ValueError):
            make_pilot_tasks(
                "/tmp",
                CONFIG_PATH,
                ("building_0001",),
                (1, 0),
                load_bundle(CONFIG_PATH),
            )

    def test_package_version_matches_frozen_contract(self) -> None:
        bundle = load_bundle(BIDIRECTIONAL_CONFIG_PATH)
        from gendiff_data_process import __version__

        self.assertEqual(bundle.package.version, __version__)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import unittest

import yaml

from gendiff_data_process.canonicalization.adapters.area_v2 import (
    AreaNormalizationStats,
    AreaV2Adapter,
    AreaV2Capacity,
)
from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.packed_contract import (
    canonical_pair_hash,
    validate_packed_sample,
)

from tests.canonicalization.helpers import (
    BIDIRECTIONAL_CONFIG_PATH,
    CONFIG_PATH,
    FIXTURE_ROOT,
    make_all_actions_sequence,
    make_layer_delete_sequence,
)


class AreaV2AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)
        fixture = yaml.safe_load(
            (FIXTURE_ROOT / "adapter/all_actions.yaml").read_text(encoding="utf-8")
        )
        cls.normalization = AreaNormalizationStats(**fixture["normalization"])
        cls.sequence = make_all_actions_sequence(cls.bundle)

    def _adapt_pair(self, adapter: AreaV2Adapter) -> dict:
        condition = [[0.0, 0.0, 0.0]] * self.bundle.condition_sampling.point_count
        return adapter.adapt_pair(
            self.sequence,
            0,
            condition,
            condition_hash="fixture_condition_hash",
            split="train",
        )

    def test_all_actions_and_absolute_target_values(self) -> None:
        adapter = AreaV2Adapter(self.bundle, self.normalization)
        pair = self._adapt_pair(adapter)
        self.assertEqual(pair["pack_schema_version"], "area_v2_packed_v1")
        self.assertEqual(
            pair["edit_schema_version"], "area_v2_absolute_target_coord_no_anchor"
        )
        actions = {
            point["action"]
            for layer in pair["sample"]["edit_object"]
            for point in layer["point_edits"]
        }
        self.assertEqual(actions, {"KEEP", "MOVE", "DELETE", "INSERT"})
        for layer in pair["sample"]["edit_object"]:
            for point in layer["point_edits"]:
                if point["action"] in {"MOVE", "INSERT"}:
                    self.assertEqual(point["value"], point["target_coord"])
                else:
                    self.assertEqual(point["value"], [0.0, 0.0])

    def test_hash_metadata_survives_adapter_boundary(self) -> None:
        adapter = AreaV2Adapter(self.bundle, self.normalization)
        pair = self._adapt_pair(adapter)
        metadata = pair["sample"]["canonical_metadata"]
        self.assertEqual(
            metadata["edit_hash"], self.sequence.adjacent_edits[0].edit_hash
        )
        self.assertEqual(
            metadata["canonicalizer_config_hash"],
            self.bundle.canonicalizer.canonicalizer_config_hash,
        )
        self.assertEqual(metadata["normalization_profile_id"], "fixture_explicit_v1")
        self.assertEqual(metadata["pair_hash"], canonical_pair_hash(metadata))
        self.assertEqual(metadata["change_kind"], "mixed")

    def test_demolition_direction_is_explicit_in_packed_metadata(self) -> None:
        bundle = load_bundle(BIDIRECTIONAL_CONFIG_PATH)
        sequence = make_layer_delete_sequence(bundle)
        adapter = AreaV2Adapter(bundle, self.normalization)
        pair = adapter.adapt_pair(
            sequence,
            0,
            [[0.0, 0.0, 0.0]] * bundle.condition_sampling.point_count,
            condition_hash="fixture_demolition_condition_hash",
            split="train",
        )
        metadata = pair["sample"]["canonical_metadata"]
        self.assertEqual(metadata["change_kind"], "demolition")
        self.assertIn(
            "DELETE", {edit["action"] for edit in pair["sample"]["edit_object"]}
        )
        validate_packed_sample(pair["sample"])

    def test_pair_hash_tampering_fails_but_legacy_v1_metadata_remains_readable(
        self,
    ) -> None:
        pair = self._adapt_pair(AreaV2Adapter(self.bundle, self.normalization))
        tampered = copy.deepcopy(pair["sample"])
        tampered["canonical_metadata"]["pair_hash"] = "0" * 64
        with self.assertRaises(CanonicalizationError) as caught:
            validate_packed_sample(tampered)
        self.assertEqual(caught.exception.code, "E_PACKED_CANONICAL_METADATA")

        legacy = copy.deepcopy(pair["sample"])
        legacy["canonical_metadata"].pop("change_kind")
        legacy["canonical_metadata"].pop("pair_hash")
        validate_packed_sample(legacy)

    def test_capacity_failure_happens_before_tensorization(self) -> None:
        adapter = AreaV2Adapter(
            self.bundle,
            self.normalization,
            AreaV2Capacity(max_layers=1, max_points_per_layer=32, max_buildings=1),
        )
        with self.assertRaises(CanonicalizationError) as caught:
            self._adapt_pair(adapter)
        self.assertEqual(caught.exception.code, "E_LAYER_CAPACITY")

    def test_condition_count_and_split_contract_fail_closed(self) -> None:
        adapter = AreaV2Adapter(self.bundle, self.normalization)
        with self.assertRaises(CanonicalizationError) as caught:
            adapter.adapt_pair(
                self.sequence,
                0,
                [[0.0, 0.0, 0.0]],
                condition_hash="fixture_condition_hash",
                split="train",
            )
        self.assertEqual(caught.exception.code, "E_CONDITION_SAMPLING")

    def test_16_building_boundary_passes_and_17th_fails(self) -> None:
        adapter = AreaV2Adapter(self.bundle, self.normalization)
        adapter.adapt_state(self.sequence.stages[0], 15, self.sequence.building_key)
        with self.assertRaises(CanonicalizationError) as caught:
            adapter.adapt_state(self.sequence.stages[0], 16, self.sequence.building_key)
        self.assertEqual(caught.exception.code, "E_BUILDING_CAPACITY")

    def test_lineage_stride_overflow_fails(self) -> None:
        with self.assertRaises(CanonicalizationError) as caught:
            AreaV2Adapter._proxy_id(0, 10_000)
        self.assertEqual(caught.exception.code, "E_ID_OVERFLOW")


if __name__ == "__main__":
    unittest.main()

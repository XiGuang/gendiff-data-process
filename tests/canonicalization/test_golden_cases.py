from __future__ import annotations

import unittest
from dataclasses import replace

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.core import canonicalize_building_sequence, canonicalize_stage
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.point_matching import align_points

from tests.canonicalization.helpers import CONFIG_PATH, FIXTURE_ROOT, load_raw_sequence


class GoldenCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)

    def _point_actions(self, fixture_name: str) -> list[str]:
        raw = load_raw_sequence(FIXTURE_ROOT / f"golden/{fixture_name}.yaml")
        source = canonicalize_stage(raw.stages[0], self.bundle).layers[0]
        target = canonicalize_stage(raw.stages[1], self.bundle).layers[0]
        source = replace(source, layer_lineage_id=0, point_lineage_ids=tuple(range(len(source.footprint_q))))
        return [edit.action for edit in align_points(source, target.footprint_q, self.bundle.canonicalizer).edits]

    def test_building_0097_only_real_coordinate_change_moves(self) -> None:
        actions = self._point_actions("building_0097")
        self.assertEqual(actions.count("KEEP_POINT"), 4)
        self.assertEqual(actions.count("MOVE_POINT"), 1)
        self.assertNotIn("DELETE_POINT", actions)
        self.assertNotIn("INSERT_POINT", actions)

    def test_building_0099_winding_and_start_do_not_create_false_actions(self) -> None:
        actions = self._point_actions("building_0099")
        self.assertEqual(actions.count("KEEP_POINT"), 3)
        self.assertEqual(actions.count("MOVE_POINT"), 1)
        self.assertNotIn("DELETE_POINT", actions)
        self.assertNotIn("INSERT_POINT", actions)

    def test_building_0112_shared_point_lineage_is_preserved(self) -> None:
        actions = self._point_actions("building_0112")
        self.assertEqual(actions.count("KEEP_POINT"), 5)
        self.assertEqual(actions.count("MOVE_POINT"), 1)
        self.assertNotIn("DELETE_POINT", actions)
        self.assertNotIn("INSERT_POINT", actions)

    def test_building_0299_33_points_is_hard_error(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "golden/building_0299.yaml")
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_stage(raw.stages[0], self.bundle)
        self.assertEqual(caught.exception.code, "E_POINT_CAPACITY")

    def test_building_1500_zero_height_is_hard_error(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "golden/building_1500.yaml")
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_stage(raw.stages[0], self.bundle)
        self.assertEqual(caught.exception.code, "E_INVALID_HEIGHT")

    def test_building_0006_overlap_is_reported(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "golden/building_0006.yaml")
        stage = canonicalize_stage(raw.stages[0], self.bundle)
        self.assertIn("W_RAW_OVERLAP_CANONICALIZED", stage.warnings)

    def test_building_0007_selected_regression_fails_construction(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "golden/building_0007.yaml")
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_building_sequence(raw, self.bundle)
        self.assertEqual(caught.exception.code, "E_CONSTRUCTION_REMOVAL")


if __name__ == "__main__":
    unittest.main()

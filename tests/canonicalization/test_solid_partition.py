from __future__ import annotations

import math
import unittest

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.core import canonicalize_stage
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.types import RawLayer, RawStage

from tests.canonicalization.helpers import CONFIG_PATH, FIXTURE_ROOT, load_raw_sequence


class SolidPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)

    def test_overlap_decomposition_matches_single_union(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/overlap_equivalent.yaml")
        overlapping = canonicalize_stage(raw.stages[0], self.bundle)
        single = canonicalize_stage(raw.stages[1], self.bundle)
        self.assertEqual(overlapping.stage_hash, single.stage_hash)
        self.assertIn("W_RAW_OVERLAP_CANONICALIZED", overlapping.warnings)

    def test_multipolygon_keeps_all_components(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/multipolygon.yaml")
        stage = canonicalize_stage(raw.stages[0], self.bundle)
        self.assertEqual(len(stage.layers), 2)

    def test_union_hole_fails_closed(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/hole.yaml")
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_stage(raw.stages[0], self.bundle)
        self.assertEqual(caught.exception.code, "E_HOLE_UNSUPPORTED")

    def test_self_intersection_fails_closed(self) -> None:
        layer = RawLayer(0, 1, ((0, 0), (2, 2), (0, 2), (2, 0)))
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_stage(RawStage(0, "self_x", (layer,)), self.bundle)
        self.assertEqual(caught.exception.code, "E_SELF_INTERSECTION")

    def test_33_point_ring_fails_capacity_without_truncation(self) -> None:
        points = tuple(
            (10.0 * math.cos(2 * math.pi * index / 33), 10.0 * math.sin(2 * math.pi * index / 33))
            for index in range(33)
        )
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_stage(RawStage(0, "capacity", (RawLayer(0, 1, points),)), self.bundle)
        self.assertEqual(caught.exception.code, "E_POINT_CAPACITY")

    def test_32_point_ring_passes_exact_capacity(self) -> None:
        points = tuple(
            (10.0 * math.cos(2 * math.pi * index / 32), 10.0 * math.sin(2 * math.pi * index / 32))
            for index in range(32)
        )
        stage = canonicalize_stage(RawStage(0, "capacity", (RawLayer(0, 1, points),)), self.bundle)
        self.assertEqual(len(stage.layers[0].footprint_q), 32)

    def test_64_layers_pass_and_65_layers_fail_without_truncation(self) -> None:
        layers = tuple(
            RawLayer(0, 1, ((index * 2, 0), (index * 2 + 1, 0), (index * 2 + 1, 1), (index * 2, 1)))
            for index in range(65)
        )
        accepted = canonicalize_stage(RawStage(0, "capacity_64", layers[:64]), self.bundle)
        self.assertEqual(len(accepted.layers), 64)
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_stage(RawStage(0, "capacity_65", layers), self.bundle)
        self.assertEqual(caught.exception.code, "E_LAYER_CAPACITY")


if __name__ == "__main__":
    unittest.main()

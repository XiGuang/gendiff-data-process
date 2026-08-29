from __future__ import annotations

import unittest

from gendiff_data_process.canonicalization.condition import build_canonical_condition
from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.core import canonicalize_stage
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.types import RawLayer, RawStage

from tests.canonicalization.helpers import BIDIRECTIONAL_CONFIG_PATH, CONFIG_PATH


class ConditionSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)
        cls.bidirectional_bundle = load_bundle(BIDIRECTIONAL_CONFIG_PATH)
        cls.empty = canonicalize_stage(RawStage(0, "empty", ()), cls.bundle)
        layer = RawLayer(0, 1, ((0, 0), (1, 0), (1, 1), (0, 1)))
        cls.box = canonicalize_stage(RawStage(1, "box", (layer,)), cls.bundle)

    def test_box_surface_sampling_is_deterministic_sorted_and_exact(self) -> None:
        first = build_canonical_condition(
            self.empty, self.box, self.bundle.condition_sampling
        )
        second = build_canonical_condition(
            self.empty, self.box, self.bundle.condition_sampling
        )
        self.assertEqual(first, second)
        self.assertEqual(
            len(first.points_q), self.bundle.condition_sampling.point_count
        )
        self.assertEqual(first.points_q, tuple(sorted(first.points_q)))
        for x_q, y_q, z_q in first.points_q:
            self.assertTrue(x_q in {0, 1000} or y_q in {0, 1000} or z_q in {0, 1000})

    def test_noop_condition_fails_instead_of_emitting_zero_point(self) -> None:
        with self.assertRaises(CanonicalizationError) as caught:
            build_canonical_condition(
                self.box, self.box, self.bundle.condition_sampling
            )
        self.assertEqual(caught.exception.code, "E_CONDITION_EMPTY")

    def test_removal_condition_fails_closed(self) -> None:
        with self.assertRaises(CanonicalizationError) as caught:
            build_canonical_condition(
                self.box, self.empty, self.bundle.condition_sampling
            )
        self.assertEqual(caught.exception.code, "E_CONSTRUCTION_REMOVAL")

    def test_directional_condition_supports_construction_and_demolition(self) -> None:
        config = self.bidirectional_bundle.condition_sampling
        construction = build_canonical_condition(self.empty, self.box, config)
        demolition = build_canonical_condition(self.box, self.empty, config)
        self.assertEqual(construction.change_kind, "construction")
        self.assertEqual(demolition.change_kind, "demolition")
        self.assertEqual(len(construction.points_q), config.point_count)
        self.assertEqual(len(demolition.points_q), config.point_count)
        self.assertEqual(construction.points_q, demolition.points_q)
        self.assertEqual(construction.seed_hex, demolition.seed_hex)
        self.assertNotEqual(construction.condition_hash, demolition.condition_hash)

    def test_directional_condition_rejects_mixed_change(self) -> None:
        source_layer = RawLayer(0, 1, ((0, 0), (1, 0), (1, 1), (0, 1)))
        target_layer = RawLayer(0, 1, ((0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)))
        source = canonicalize_stage(
            RawStage(0, "source", (source_layer,)), self.bidirectional_bundle
        )
        target = canonicalize_stage(
            RawStage(1, "target", (target_layer,)), self.bidirectional_bundle
        )
        with self.assertRaises(CanonicalizationError) as caught:
            build_canonical_condition(
                source, target, self.bidirectional_bundle.condition_sampling
            )
        self.assertEqual(caught.exception.code, "E_MIXED_CHANGE_UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()

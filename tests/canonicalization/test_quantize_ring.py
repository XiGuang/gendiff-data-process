from __future__ import annotations

import unittest
from dataclasses import replace

from gendiff_data_process.canonicalization.config import load_bundle, validate_bundle
from gendiff_data_process.canonicalization.core import canonicalize_stage
from gendiff_data_process.canonicalization.polygon import cleanup_ring
from gendiff_data_process.canonicalization.quantize import quantize_scalar
from gendiff_data_process.canonicalization.serialize import canonical_json_bytes
from gendiff_data_process.canonicalization.types import RawLayer, RawStage

from tests.canonicalization.helpers import CONFIG_PATH


class QuantizeAndRingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)

    def test_half_away_from_zero_boundaries(self) -> None:
        self.assertEqual(quantize_scalar("0.0005", "0.001"), 1)
        self.assertEqual(quantize_scalar("-0.0005", "0.001"), -1)
        self.assertEqual(quantize_scalar("1.2344", "0.001"), 1234)
        self.assertEqual(quantize_scalar("1.2345", "0.001"), 1235)

    def test_rotation_reverse_closure_and_collinear_are_invariant(self) -> None:
        expected = cleanup_ring(((0, 0), (4000, 0), (4000, 2000), (0, 2000)))
        variants = [
            ((4000, 2000), (0, 2000), (0, 0), (4000, 0)),
            ((0, 0), (0, 2000), (4000, 2000), (4000, 0)),
            ((0, 0), (2000, 0), (4000, 0), (4000, 2000), (0, 2000), (0, 0)),
        ]
        for variant in variants:
            self.assertEqual(cleanup_ring(variant), expected)

    def test_raw_layer_order_and_ids_do_not_change_stage_hash(self) -> None:
        left = RawLayer(0, 1, ((0, 0), (1, 0), (1, 1), (0, 1)), raw_proxy_id=11)
        right = RawLayer(0, 1, ((3, 0), (4, 0), (4, 1), (3, 1)), raw_proxy_id=12)
        first = canonicalize_stage(RawStage(0, "stage", (left, right)), self.bundle)
        second = canonicalize_stage(
            RawStage(0, "stage", (replace(right, raw_proxy_id=-1), replace(left, raw_proxy_id=999))),
            self.bundle,
        )
        self.assertEqual(first.stage_hash, second.stage_hash)
        self.assertEqual(canonical_json_bytes(first.layers), canonical_json_bytes(second.layers))

    def test_quantization_point_collapse_is_reported(self) -> None:
        layer = RawLayer(
            0,
            1,
            ((0, 0), ("0.0004", 0), (1, 0), (1, 1), (0, 1)),
        )
        stage = canonicalize_stage(RawStage(0, "collapse", (layer,)), self.bundle)
        self.assertIn("W_QUANTIZATION_COLLAPSE", stage.warnings)

    def test_validation_capacity_does_not_change_geometry_hash(self) -> None:
        changed_profile = replace(self.bundle.validation_profile, max_layers=32)
        self.assertNotEqual(changed_profile.config_hash, self.bundle.validation_profile.config_hash)
        layer = RawLayer(0, 1, ((0, 0), (1, 0), (1, 1), (0, 1)))
        raw = RawStage(0, "stage", (layer,))
        baseline = canonicalize_stage(raw, self.bundle)
        changed = canonicalize_stage(raw, replace(self.bundle, validation_profile=changed_profile))
        self.assertEqual(baseline.stage_hash, changed.stage_hash)

    def test_geometry_config_change_changes_stage_hash(self) -> None:
        layer = RawLayer(0, 1, ((0, 0), (1, 0), (1, 1), (0, 1)))
        raw = RawStage(0, "stage", (layer,))
        baseline = canonicalize_stage(raw, self.bundle)
        changed_config = replace(self.bundle.canonicalizer, geometry_version="canonical_geometry_test")
        changed = canonicalize_stage(raw, replace(self.bundle, canonicalizer=changed_config))
        self.assertNotEqual(baseline.stage_hash, changed.stage_hash)

    def test_unsupported_declared_policy_is_rejected(self) -> None:
        changed_config = replace(self.bundle.canonicalizer, rounding="bankers")
        with self.assertRaises(ValueError):
            validate_bundle(replace(self.bundle, canonicalizer=changed_config))


if __name__ == "__main__":
    unittest.main()

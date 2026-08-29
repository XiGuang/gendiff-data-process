from __future__ import annotations

import unittest
from dataclasses import replace

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.core import canonicalize_building_sequence
from gendiff_data_process.canonicalization.edit_v3 import apply_canonical_edit
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.layer_matching import match_layers
from gendiff_data_process.canonicalization.types import CanonicalLayer

from tests.canonicalization.helpers import (
    BIDIRECTIONAL_CONFIG_PATH,
    CONFIG_PATH,
    FIXTURE_ROOT,
    load_raw_sequence,
)


class LineageAndEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)
        cls.bidirectional_bundle = load_bundle(BIDIRECTIONAL_CONFIG_PATH)

    def test_ring_changes_do_not_create_false_point_actions(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/ring_rect.yaml")
        sequence = canonicalize_building_sequence(raw, self.bundle)
        point_actions = [
            point_edit.action
            for layer_edit in sequence.adjacent_edits[0].layer_edits
            for point_edit in layer_edit.point_edits
            if point_edit.action != "EOS"
        ]
        self.assertEqual(point_actions, ["KEEP_POINT"] * 4)
        self.assertEqual(
            sequence.stages[0].layers[0].point_lineage_ids,
            sequence.stages[1].layers[0].point_lineage_ids,
        )

    def test_roundtrip_and_repeated_run_are_identical(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/ring_rect.yaml")
        first = canonicalize_building_sequence(raw, self.bundle)
        second = canonicalize_building_sequence(raw, self.bundle)
        self.assertEqual(first, second)
        self.assertEqual(first.sequence_hash, second.sequence_hash)

    def test_tampered_edit_hash_fails_closed(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/ring_rect.yaml")
        sequence = canonicalize_building_sequence(raw, self.bundle)
        tampered = replace(sequence.adjacent_edits[0], edit_hash="0" * 64)
        with self.assertRaises(CanonicalizationError) as caught:
            apply_canonical_edit(
                sequence.stages[0], tampered, self.bundle.canonicalizer
            )
        self.assertEqual(caught.exception.code, "E_ROUNDTRIP_MISMATCH")

    def test_noop_is_marked(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/overlap_equivalent.yaml")
        sequence = canonicalize_building_sequence(raw, self.bundle)
        self.assertIn("W_NOOP_STAGE", sequence.stages[1].warnings)

    def test_construction_removal_is_hard_error(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/construction_removal.yaml")
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_building_sequence(raw, self.bundle)
        self.assertEqual(caught.exception.code, "E_CONSTRUCTION_REMOVAL")

    def test_bidirectional_profile_keeps_demolition_edit_executable(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/construction_removal.yaml")
        sequence = canonicalize_building_sequence(raw, self.bidirectional_bundle)
        edit = sequence.adjacent_edits[0]
        applied = apply_canonical_edit(
            sequence.stages[0],
            edit,
            self.bidirectional_bundle.canonicalizer,
        )
        self.assertEqual(applied.stage_hash, sequence.stages[1].stage_hash)

    def test_explicit_expected_stage_list_rejects_missing_stage(self) -> None:
        raw = load_raw_sequence(FIXTURE_ROOT / "synthetic/ring_rect.yaml")
        raw = replace(raw, expected_stage_indices=(0, 1, 2))
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_building_sequence(raw, self.bundle)
        self.assertEqual(caught.exception.code, "E_MISSING_STAGE")
        self.assertEqual(caught.exception.context["actual"], (0, 1))
        self.assertEqual(caught.exception.context["expected"], (0, 1, 2))

    def test_symmetric_layer_split_uses_stable_lowest_target(self) -> None:
        source = CanonicalLayer(
            0, 0, 1000, ((0, 0), (4000, 0), (4000, 2000), (0, 2000)), "s"
        )
        left = CanonicalLayer(
            0, 0, 1000, ((0, 0), (2000, 0), (2000, 2000), (0, 2000)), "l"
        )
        right = CanonicalLayer(
            1, 0, 1000, ((2000, 0), (4000, 0), (4000, 2000), (2000, 2000)), "r"
        )
        results = [
            match_layers((source,), (left, right), self.bundle.canonicalizer)
            for _ in range(100)
        ]
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(results[0][0][0].target_index, 0)
        self.assertIn("W_LAYER_SPLIT", results[0][1])


if __name__ == "__main__":
    unittest.main()

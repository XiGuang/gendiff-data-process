from __future__ import annotations

import random
import unittest

from gendiff_data_process.canonicalization.collision import (
    audit_building_uid_collisions,
    audit_supervision_collisions,
)
from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.core import canonicalize_stage
from gendiff_data_process.canonicalization.errors import CanonicalizationError
from gendiff_data_process.canonicalization.types import RawLayer, RawStage

from tests.canonicalization.helpers import CONFIG_PATH


class DeterminismAndCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)

    @staticmethod
    def _rotate(points, offset: int, reverse: bool):
        sequence = list(reversed(points)) if reverse else list(points)
        return tuple(sequence[offset:] + sequence[:offset])

    def test_random_ring_layer_and_id_permutations_are_invariant(self) -> None:
        base_rings = (
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
            ((4.0, 0.0), (5.0, 0.0), (5.0, 2.0), (4.0, 2.0)),
        )
        baseline = canonicalize_stage(
            RawStage(0, "stage", tuple(RawLayer(0, 1, ring, index) for index, ring in enumerate(base_rings))),
            self.bundle,
        )
        rng = random.Random(0)
        for _ in range(100):
            layers = [
                RawLayer(
                    0,
                    1,
                    self._rotate(ring, rng.randrange(len(ring)), bool(rng.randrange(2))),
                    rng.randrange(-100000, 100000),
                )
                for ring in base_rings
            ]
            rng.shuffle(layers)
            candidate = canonicalize_stage(RawStage(0, "stage", tuple(layers)), self.bundle)
            self.assertEqual(candidate.stage_hash, baseline.stage_hash)

    def test_same_grid_jitter_is_invariant_and_cross_grid_change_is_not(self) -> None:
        base = RawLayer(0, 1, ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
        jitter = RawLayer(0, 1, ((0.0001, 0.0001), (2.0001, 0.0001), (2.0001, 1.0001), (0.0001, 1.0001)))
        changed = RawLayer(0, 1, ((0.0006, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
        base_hash = canonicalize_stage(RawStage(0, "base", (base,)), self.bundle).stage_hash
        self.assertEqual(base_hash, canonicalize_stage(RawStage(0, "jitter", (jitter,)), self.bundle).stage_hash)
        self.assertNotEqual(base_hash, canonicalize_stage(RawStage(0, "changed", (changed,)), self.bundle).stage_hash)

    def test_duplicate_rows_are_reported_without_conflict(self) -> None:
        record = {
            "source_stage_hash": "source",
            "condition_hash": "condition",
            "target_stage_hash": "target",
            "edit_hash": "edit",
        }
        report = audit_supervision_collisions((record, record))
        self.assertEqual(report.unique_key_count, 1)
        self.assertEqual(report.duplicate_row_count, 1)

    def test_conflicting_supervision_fails_closed(self) -> None:
        records = (
            {"source_stage_hash": "s", "condition_hash": "c", "target_stage_hash": "t1", "edit_hash": "e1"},
            {"source_stage_hash": "s", "condition_hash": "c", "target_stage_hash": "t2", "edit_hash": "e2"},
        )
        with self.assertRaises(CanonicalizationError) as caught:
            audit_supervision_collisions(records)
        self.assertEqual(caught.exception.code, "E_SUPERVISION_COLLISION")

    def test_truncated_building_uid_collision_fails_closed(self) -> None:
        records = (
            {"building_key": "building_a", "building_uid": "same_uid"},
            {"building_key": "building_b", "building_uid": "same_uid"},
        )
        with self.assertRaises(CanonicalizationError) as caught:
            audit_building_uid_collisions(records)
        self.assertEqual(caught.exception.code, "E_BUILDING_UID_COLLISION")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.release_contracts import (
    assign_building_splits,
    building_uid_from_key,
    compute_train_normalization_profile,
    split_for_building_uid,
)

from tests.canonicalization.helpers import CONFIG_PATH, make_all_actions_sequence


class ReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)

    def test_building_split_is_order_invariant_and_disjoint(self) -> None:
        keys = [f"building_{index:04d}" for index in range(1, 101)]
        forward = assign_building_splits(keys, self.bundle.split)
        reverse = assign_building_splits(reversed(keys), self.bundle.split)
        self.assertEqual(forward, reverse)
        self.assertEqual(set(forward), set(keys))
        self.assertEqual(set(forward.values()), {"train", "val", "test"})

    def test_split_known_value_is_frozen(self) -> None:
        uid = building_uid_from_key("building_0001")
        self.assertEqual(split_for_building_uid(uid, self.bundle.split), "test")

    def test_train_only_bbox_profile_is_permutation_invariant(self) -> None:
        first = make_all_actions_sequence(self.bundle)
        second = replace(
            first,
            building_key="SYN_ADAPTER_002",
            building_uid=building_uid_from_key("SYN_ADAPTER_002"),
        )
        profile_a = compute_train_normalization_profile(
            (first, second),
            self.bundle.normalization,
            grid_xz=self.bundle.canonicalizer.grid_xz,
            grid_y=self.bundle.canonicalizer.grid_y,
        )
        profile_b = compute_train_normalization_profile(
            (second, first),
            self.bundle.normalization,
            grid_xz=self.bundle.canonicalizer.grid_xz,
            grid_y=self.bundle.canonicalizer.grid_y,
        )
        self.assertEqual(profile_a, profile_b)
        self.assertEqual(profile_a.center_x, 2.0)
        self.assertEqual(profile_a.center_z, 0.5)
        self.assertEqual(profile_a.center_y, 1.0)
        self.assertEqual(profile_a.scale_xz, 4.0)
        self.assertEqual(profile_a.scale_y, 4.0)

    def test_non_train_outlier_cannot_change_profile(self) -> None:
        train = make_all_actions_sequence(self.bundle)
        baseline = compute_train_normalization_profile(
            (train,),
            self.bundle.normalization,
            grid_xz=self.bundle.canonicalizer.grid_xz,
            grid_y=self.bundle.canonicalizer.grid_y,
        )
        outlier = replace(
            train,
            building_key="VALIDATION_OUTLIER",
            building_uid=building_uid_from_key("VALIDATION_OUTLIER"),
        )
        observed_after_loading_outlier = compute_train_normalization_profile(
            (train,),
            self.bundle.normalization,
            grid_xz=self.bundle.canonicalizer.grid_xz,
            grid_y=self.bundle.canonicalizer.grid_y,
        )
        self.assertNotEqual(outlier.building_uid, train.building_uid)
        self.assertEqual(baseline, observed_after_loading_outlier)


if __name__ == "__main__":
    unittest.main()

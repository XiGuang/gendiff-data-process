from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

try:
    import torch
except ImportError:  # pragma: no cover - viewer packed 检查需要 Torch
    torch = None

from gendiff_data_process.canonicalization.adapters.area_v2 import (
    AreaNormalizationStats,
)
from gendiff_data_process.canonicalization.config import load_bundle
from tests.canonicalization.helpers import (
    BIDIRECTIONAL_CONFIG_PATH,
    write_bidirectional_packed_fixture,
)

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import dataset_browser_api as browser_api  # noqa: E402
import export_edit_animation_viewer_data as viewer_exporter  # noqa: E402


@unittest.skipIf(torch is None, "需要包含 Torch 的既有 GenDiff Python 环境")
class PackedViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        loaded = load_bundle(BIDIRECTIONAL_CONFIG_PATH)
        self.bundle = replace(
            loaded,
            condition_sampling=replace(loaded.condition_sampling, point_count=8),
        )
        normalization = AreaNormalizationStats(
            "viewer_fixture_v1", 0.0, 0.0, 10.0, 0.0, 10.0
        )
        write_bidirectional_packed_fixture(self.root, self.bundle, normalization)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_summary_and_split_page_use_packed_indexes(self) -> None:
        summary = browser_api.dataset_summary(self.root)
        self.assertEqual(summary["datasetFormat"], "packed")
        self.assertEqual(summary["pairTotal"], 3)
        self.assertEqual(summary["stateTotal"], 6)
        self.assertEqual(summary["splits"], {"train": 1, "val": 1, "test": 1})

        page = browser_api.pair_list(self.root, "val", "", 0, 100)
        self.assertEqual(page["total"], 1)
        item = page["pairs"][0]
        self.assertEqual(item["changeKind"], "demolition")
        self.assertTrue(item["pairLocator"].startswith("packed:val:"))
        self.assertTrue(item["pairHash"])
        self.assertEqual(item["conditionPointCount"], 8)

    def test_demolition_and_reverse_construction_export_for_playback(self) -> None:
        val_item = browser_api.pair_list(self.root, "val", "", 0, 1)["pairs"][0]
        demolition = viewer_exporter.export_viewer_data(
            self.root,
            val_item["pairId"],
            pair_locator=val_item["pairLocator"],
        )
        pair = demolition["pairs"][0]
        self.assertEqual(demolition["dataset_format"], "packed")
        self.assertEqual(pair["change_kind"], "demolition")
        self.assertEqual(pair["dataset_locator"], val_item["pairLocator"])
        deleted_layers = [
            layer for layer in pair["layers"] if layer["layer_action"] == "DELETE"
        ]
        self.assertEqual(len(deleted_layers), 1)
        self.assertGreaterEqual(len(deleted_layers[0]["source_points"]), 3)
        self.assertEqual(deleted_layers[0]["target_points"], [])
        self.assertTrue(
            all(op["type"] == "DELETE_POINT" for op in deleted_layers[0]["ops"])
        )

        test_item = browser_api.pair_list(self.root, "test", "", 0, 1)["pairs"][0]
        construction = viewer_exporter.export_viewer_data(
            self.root,
            test_item["pairId"],
            pair_locator=test_item["pairLocator"],
        )
        reverse_pair = construction["pairs"][0]
        self.assertEqual(reverse_pair["change_kind"], "construction")
        self.assertTrue(
            any(layer["layer_action"] == "INSERT" for layer in reverse_pair["layers"])
        )
        self.assertNotEqual(reverse_pair["pair_hash"], pair["pair_hash"])

    def test_condition_is_loaded_from_selected_packed_sample(self) -> None:
        item = browser_api.pair_list(self.root, "val", "", 0, 1)["pairs"][0]
        condition = browser_api.condition(
            self.root, item["pairId"], 4, item["pairLocator"]
        )
        self.assertTrue(condition["available"])
        self.assertEqual(condition["totalPoints"], 8)
        self.assertEqual(condition["sampledPoints"], 4)
        self.assertEqual(len(condition["points"]), 12)
        self.assertIn(".condition", condition["path"])


class RawViewerCompatibilityTests(unittest.TestCase):
    def test_raw_area_v2_pair_still_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_name = "area_state_000000"
            target_name = "area_state_000001"
            pair_name = f"pair_000000_{source_name}_to_{target_name}"
            source_layer = {
                "proxy_id": 7,
                "building_id": 0,
                "building_layer_index": 0,
                "min_height": 0.0,
                "max_height": 0.5,
                "footprint": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            }
            target_layer = {
                **source_layer,
                "max_height": 1.0,
            }
            for state_name, layer in (
                (source_name, source_layer),
                (target_name, target_layer),
            ):
                state_dir = root / "states" / state_name
                state_dir.mkdir(parents=True)
                (state_dir / f"bs_{state_name}_r0.yaml").write_text(
                    yaml.safe_dump([layer], sort_keys=False), encoding="utf-8"
                )
            tokens = [
                {
                    "type": "MODIFY_LAYER",
                    "value": {
                        "source_proxy_id": 7,
                        "target_proxy_id": 7,
                        "source_building_id": 0,
                        "target_building_id": 0,
                    },
                },
                *[
                    {
                        "type": "KEEP_POINT",
                        "value": {
                            "source_point_index": index,
                            "target_point_index": index,
                        },
                    }
                    for index in range(4)
                ],
            ]
            for folder, payload in (
                ("edit_sequences_v2", tokens),
                (
                    "pair_meta",
                    {
                        "source_state": source_name,
                        "target_state": target_name,
                        "is_demolition_pair": False,
                    },
                ),
                (
                    "edit_objects",
                    [
                        {
                            "action": "MODIFY",
                            "source_proxy_id": 7,
                            "target_proxy_id": 7,
                            "source_building_id": 0,
                            "target_building_id": 0,
                        }
                    ],
                ),
            ):
                path = root / folder / f"{pair_name}.yaml"
                path.parent.mkdir(parents=True)
                path.write_text(
                    yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
                )
            (root / "dataset_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "edit_schema_version": "area_v2_absolute_target_coord_no_anchor",
                        "coordinate_normalized": True,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            summary = browser_api.dataset_summary(root)
            exported = viewer_exporter.export_viewer_data(root, pair_name)
            self.assertEqual(summary["datasetFormat"], "raw")
            self.assertEqual(summary["pairTotal"], 1)
            self.assertEqual(exported["dataset_format"], "raw")
            self.assertEqual(
                exported["pairs"][0]["layers"][0]["layer_action"], "MODIFY"
            )
            self.assertEqual(len(exported["pairs"][0]["layers"][0]["ops"]), 4)


if __name__ == "__main__":
    unittest.main()

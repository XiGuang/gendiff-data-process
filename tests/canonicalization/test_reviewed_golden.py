from __future__ import annotations

import hashlib
import json
import unicodedata
import unittest
from dataclasses import asdict, is_dataclass
from decimal import Decimal, ROUND_HALF_UP

import yaml
from shapely import set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.core import canonicalize_building_sequence
from gendiff_data_process.canonicalization.errors import CanonicalizationError

from tests.canonicalization.helpers import CONFIG_PATH, FIXTURE_ROOT, load_raw_sequence


def _independent_quantize(value, grid: str = "0.001") -> int:
    return int((Decimal(str(value)) / Decimal(grid)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _independent_json_value(value):
    if is_dataclass(value):
        return _independent_json_value(asdict(value))
    if isinstance(value, dict):
        return {
            unicodedata.normalize("NFC", str(key)): _independent_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_independent_json_value(item) for item in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def _independent_hash(value) -> str:
    payload = json.dumps(
        _independent_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_geometry(stage, lower_q: int, upper_q: int):
    polygons = []
    for layer in stage.layers:
        minimum = _independent_quantize(layer.min_height)
        maximum = _independent_quantize(layer.max_height)
        if minimum <= lower_q and maximum >= upper_q:
            ring = [
                (_independent_quantize(point[0]), _independent_quantize(point[1]))
                for point in layer.footprint
            ]
            polygons.append(Polygon(ring))
    return set_precision(unary_union(polygons), grid_size=1.0) if polygons else None


def _canonical_geometry(stage, lower_q: int, upper_q: int):
    polygons = [
        Polygon(layer.footprint_q)
        for layer in stage.layers
        if layer.min_height_q <= lower_q and layer.max_height_q >= upper_q
    ]
    return set_precision(unary_union(polygons), grid_size=1.0) if polygons else None


def _assert_stage_solid_equal(test: unittest.TestCase, raw_stage, canonical_stage) -> None:
    heights = sorted(
        {
            *(
                value
                for layer in raw_stage.layers
                for value in (
                    _independent_quantize(layer.min_height),
                    _independent_quantize(layer.max_height),
                )
            ),
            *(
                value
                for layer in canonical_stage.layers
                for value in (layer.min_height_q, layer.max_height_q)
            ),
        }
    )
    for lower_q, upper_q in zip(heights, heights[1:]):
        raw = _raw_geometry(raw_stage, lower_q, upper_q)
        canonical = _canonical_geometry(canonical_stage, lower_q, upper_q)
        if raw is None or raw.is_empty:
            test.assertTrue(canonical is None or canonical.is_empty)
        elif canonical is None or canonical.is_empty:
            test.fail(f"canonical solid 缺少高度区间 {lower_q}:{upper_q}")
        else:
            test.assertEqual(raw.symmetric_difference(canonical).area, 0.0)


def _independent_removed_volume2(source, target) -> int:
    heights = sorted(
        {
            value
            for stage in (source, target)
            for layer in stage.layers
            for value in (
                _independent_quantize(layer.min_height),
                _independent_quantize(layer.max_height),
            )
        }
    )
    removed = 0
    for lower_q, upper_q in zip(heights, heights[1:]):
        source_geometry = _raw_geometry(source, lower_q, upper_q)
        target_geometry = _raw_geometry(target, lower_q, upper_q)
        if source_geometry is None or source_geometry.is_empty:
            continue
        difference = source_geometry if target_geometry is None else source_geometry.difference(target_geometry)
        removed += int(round(difference.area * 2)) * (upper_q - lower_q)
    return removed


class ReviewedGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(CONFIG_PATH)
        cls.reviewed = yaml.safe_load(
            (FIXTURE_ROOT / "golden/reviewed_hashes.yaml").read_text(encoding="utf-8")
        )

    def test_reviewed_config_hashes_match(self) -> None:
        self.assertEqual(
            self.reviewed["geometry_config_hash"],
            self.bundle.canonicalizer.geometry_config_hash,
        )
        self.assertEqual(
            self.reviewed["canonicalizer_config_hash"],
            self.bundle.canonicalizer.canonicalizer_config_hash,
        )

    def test_reviewed_pass_hashes_with_independent_oracle(self) -> None:
        for fixture_name in ("building_0006", "building_0097", "building_0112"):
            with self.subTest(fixture=fixture_name):
                expected = self.reviewed["fixtures"][fixture_name]
                raw = load_raw_sequence(FIXTURE_ROOT / f"golden/{fixture_name}.yaml")
                sequence = canonicalize_building_sequence(raw, self.bundle)
                self.assertEqual([stage.stage_hash for stage in sequence.stages], expected["stage_hashes"])
                self.assertEqual([edit.edit_hash for edit in sequence.adjacent_edits], expected["edit_hashes"])
                self.assertEqual(sequence.sequence_hash, expected["sequence_hash"])
                self.assertEqual(list(sequence.warnings), expected["warnings"])

                for raw_stage, stage in zip(raw.stages, sequence.stages):
                    _assert_stage_solid_equal(self, raw_stage, stage)
                    independent_stage_hash = _independent_hash(
                        {
                            "geometry_version": self.bundle.canonicalizer.geometry_version,
                            "geometry_config_hash": self.bundle.canonicalizer.geometry_config_hash,
                            "layers": [
                                {
                                    "min_height_q": layer.min_height_q,
                                    "max_height_q": layer.max_height_q,
                                    "footprint_q": layer.footprint_q,
                                }
                                for layer in stage.layers
                            ],
                        }
                    )
                    self.assertEqual(independent_stage_hash, stage.stage_hash)

                for pair_index, edit in enumerate(sequence.adjacent_edits):
                    source = sequence.stages[pair_index]
                    target = sequence.stages[pair_index + 1]
                    source_by_index = {layer.canonical_layer_index: layer for layer in source.layers}
                    target_by_index = {layer.canonical_layer_index: layer for layer in target.layers}
                    for layer_edit in edit.layer_edits:
                        if layer_edit.action == "DELETE_LAYER":
                            continue
                        points = {}
                        for point_edit in layer_edit.point_edits:
                            if point_edit.action in {"DELETE_POINT", "EOS"}:
                                continue
                            if point_edit.action == "KEEP_POINT":
                                source_layer = source_by_index[layer_edit.source_layer_index]
                                coordinate = source_layer.footprint_q[point_edit.source_index]
                            else:
                                coordinate = point_edit.target_coord_q
                            points[point_edit.target_index] = coordinate
                        target_layer = target_by_index[layer_edit.target_layer_index]
                        self.assertEqual(tuple(points[index] for index in sorted(points)), target_layer.footprint_q)
                        self.assertEqual(
                            layer_edit.target_height_q,
                            (target_layer.min_height_q, target_layer.max_height_q),
                        )
                    independent_edit_hash = _independent_hash(
                        {
                            "schema_version": "canonical_edit_v3",
                            "canonicalizer_version": self.bundle.canonicalizer.canonicalizer_version,
                            "geometry_version": self.bundle.canonicalizer.geometry_version,
                            "geometry_config_hash": self.bundle.canonicalizer.geometry_config_hash,
                            "canonicalizer_config_hash": self.bundle.canonicalizer.canonicalizer_config_hash,
                            "source_stage_hash": source.stage_hash,
                            "target_stage_hash": target.stage_hash,
                            "target_stage_index": target.stage_index,
                            "target_stage_key": target.stage_key,
                            "layer_edits": edit.layer_edits,
                        }
                    )
                    self.assertEqual(independent_edit_hash, edit.edit_hash)

                independent_uid = hashlib.sha256(
                    unicodedata.normalize("NFC", sequence.building_key).encode("utf-8")
                ).hexdigest()[:32]
                self.assertEqual(independent_uid, sequence.building_uid)
                independent_sequence_hash = _independent_hash(
                    {
                        "canonicalizer_version": self.bundle.canonicalizer.canonicalizer_version,
                        "canonicalizer_config_hash": self.bundle.canonicalizer.canonicalizer_config_hash,
                        "building_key": sequence.building_key,
                        "stage_hashes": [stage.stage_hash for stage in sequence.stages],
                        "lineage": [
                            [
                                {
                                    "layer_lineage_id": layer.layer_lineage_id,
                                    "point_lineage_ids": layer.point_lineage_ids,
                                }
                                for layer in stage.layers
                            ]
                            for stage in sequence.stages
                        ],
                        "edit_hashes": [edit.edit_hash for edit in sequence.adjacent_edits],
                    }
                )
                self.assertEqual(independent_sequence_hash, sequence.sequence_hash)

    def test_reviewed_error_cases_have_independent_trigger(self) -> None:
        for fixture_name in ("building_0007", "building_0099"):
            raw = load_raw_sequence(FIXTURE_ROOT / f"golden/{fixture_name}.yaml")
            self.assertGreater(_independent_removed_volume2(raw.stages[0], raw.stages[1]), 0)
            with self.assertRaises(CanonicalizationError) as caught:
                canonicalize_building_sequence(raw, self.bundle)
            self.assertEqual(caught.exception.code, "E_CONSTRUCTION_REMOVAL")

        capacity = load_raw_sequence(FIXTURE_ROOT / "golden/building_0299.yaml")
        self.assertEqual(len(capacity.stages[0].layers[0].footprint), 33)
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_building_sequence(capacity, self.bundle)
        self.assertEqual(caught.exception.code, "E_POINT_CAPACITY")

        height = load_raw_sequence(FIXTURE_ROOT / "golden/building_1500.yaml")
        layer = height.stages[0].layers[0]
        self.assertEqual(_independent_quantize(layer.min_height), _independent_quantize(layer.max_height))
        with self.assertRaises(CanonicalizationError) as caught:
            canonicalize_building_sequence(height, self.bundle)
        self.assertEqual(caught.exception.code, "E_INVALID_HEIGHT")


if __name__ == "__main__":
    unittest.main()

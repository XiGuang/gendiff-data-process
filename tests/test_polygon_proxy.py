from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from polygon_proxy.core import (
    ProxyConfig,
    build_face_adjacency,
    build_proxy_artifact,
    cluster_planar_patches,
    extract_patch_polygon,
    load_obj_mesh,
    make_mesh,
    weld_vertices,
)


def write_obj(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> None:
    lines = []
    for vertex in vertices:
        lines.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}")
    for face in faces:
        indices = " ".join(str(index) for index in face)
        lines.append(f"f {indices}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def box_geometry(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 6, 7),
        (5, 7, 8),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    return vertices, faces


class PolygonProxySyntheticTests(unittest.TestCase):
    def test_obj_parser_triangulates_ngons(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "ngon.obj"
            write_obj(
                obj_path,
                [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)],
                [(1, 2, 3, 4)],
            )
            mesh = load_obj_mesh(obj_path)
            self.assertEqual(mesh.face_count, 2)

    def test_weld_vertices_collapses_duplicates(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0005, 0.0002, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int64)
        welded_vertices, welded_faces = weld_vertices(vertices, faces, epsilon=0.01)
        self.assertLess(len(welded_vertices), len(vertices))
        self.assertTrue(np.all(welded_faces >= 0))

    def test_cluster_planar_patches_finds_box_faces(self) -> None:
        vertices, faces = box_geometry(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
        mesh = make_mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64) - 1)
        adjacency = build_face_adjacency(mesh.faces)
        config = ProxyConfig(min_patch_faces=2, plane_distance_tolerance=0.01, patch_merge_distance=0.01)
        patches = cluster_planar_patches(mesh, np.arange(mesh.face_count), adjacency, config)
        self.assertEqual(len(patches), 6)

    def test_extract_patch_polygon_closes_loop(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = make_mesh(vertices, faces)
        patch = extract_patch_polygon(
            mesh,
            np.array([0, 1], dtype=np.int64),
            up_axis=2,
            config=ProxyConfig(min_patch_faces=2, min_polygon_area=0.1),
        )
        self.assertIsNotNone(patch)
        vertices_3d = np.asarray(patch["vertices"])
        self.assertTrue(np.allclose(vertices_3d[0], vertices_3d[-1]))
        self.assertEqual(len(vertices_3d), 5)

    def test_ground_only_mesh_yields_zero_buildings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_dir = Path(tmpdir) / "ground_only"
            tile_dir.mkdir(parents=True)
            obj_path = tile_dir / "bs_ground_only.obj"
            write_obj(
                obj_path,
                [(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)],
                [(1, 2, 3), (1, 3, 4)],
            )
            artifact = build_proxy_artifact(
                obj_path,
                ProxyConfig(min_component_faces=1, min_component_area=1.0, ground_patch_min_area=1.0),
            )
            self.assertEqual(artifact.metrics["buildings_detected"], 0)


class PolygonProxyRealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.input_root = cls.repo_root / "data/block/yingrenshi_building_simple"

    def test_cli_generates_json_and_obj_for_real_tiles(self) -> None:
        sample_tiles = ["0_1_1", "3_2_16"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "polygon_proxy"
            command = [
                sys.executable,
                str(self.repo_root / "tools/build_polygon_proxy.py"),
                "--input-root",
                str(self.input_root),
                "--output-root",
                str(output_root),
                "--tiles",
                ",".join(sample_tiles),
                "--workers",
                "1",
                "--overwrite",
            ]
            subprocess.run(command, cwd=self.repo_root, check=True)

            for tile in sample_tiles:
                json_path = output_root / tile / f"{tile}.proxy.json"
                obj_path = output_root / tile / f"{tile}.proxy.obj"
                metrics_path = output_root / tile / f"{tile}.metrics.json"
                self.assertTrue(json_path.exists(), json_path)
                self.assertTrue(obj_path.exists(), obj_path)
                self.assertTrue(metrics_path.exists(), metrics_path)

                data = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(data["tile_id"], tile)
                self.assertGreaterEqual(data["metrics"]["buildings_detected"], 1)
                self.assertLess(data["metrics"]["proxy_faces"], data["metrics"]["input_triangles"])
                for building in data["buildings"]:
                    self.assertTrue(building["faces"])
                    for face in building["faces"]:
                        vertices = np.asarray(face["vertices"], dtype=np.float64)
                        self.assertTrue(np.allclose(vertices[0], vertices[-1]))
                        unique_vertices = np.unique(np.round(vertices[:-1], 6), axis=0)
                        self.assertGreaterEqual(len(unique_vertices), 3)
                        normal = np.asarray(face["plane_normal"], dtype=np.float64)
                        offset = float(face["plane_offset"])
                        distances = np.abs(vertices[:-1] @ normal - offset)
                        self.assertLess(float(distances.max()), 0.25)

    def test_cli_skips_requested_tile_without_source_obj(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "polygon_proxy"
            command = [
                sys.executable,
                str(self.repo_root / "tools/build_polygon_proxy.py"),
                "--input-root",
                str(self.input_root),
                "--output-root",
                str(output_root),
                "--tiles",
                "0_1_0,0_1_1",
                "--workers",
                "1",
                "--overwrite",
            ]
            completed = subprocess.run(command, cwd=self.repo_root, check=True, capture_output=True, text=True)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["tiles_processed"], 1)
            self.assertEqual(summary["results"][0]["tile_id"], "0_1_1")


if __name__ == "__main__":
    unittest.main()

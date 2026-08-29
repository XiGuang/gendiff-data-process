import * as THREE from "three";
import type { Vec2 } from "./types";
import { lerpNumber, smoothstep } from "./vec2";

function signedArea(footprint: Vec2[]): number {
  let area = 0;
  for (let i = 0; i < footprint.length; i += 1) {
    const a = footprint[i];
    const b = footprint[(i + 1) % footprint.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return area / 2;
}

export function createExtrudedPolygonGeometry(footprint: Vec2[], minY: number, maxY: number): THREE.BufferGeometry {
  if (footprint.length < 3) {
    throw new Error("createExtrudedPolygonGeometry requires at least 3 footprint points");
  }
  const points = signedArea(footprint) < 0 ? [...footprint].reverse() : [...footprint];
  const positions: number[] = [];
  const indices: number[] = [];

  for (const [x, z] of points) {
    positions.push(x, minY, z);
  }
  for (const [x, z] of points) {
    positions.push(x, maxY, z);
  }

  const n = points.length;
  for (let i = 0; i < n; i += 1) {
    const next = (i + 1) % n;
    indices.push(i, next, n + next);
    indices.push(i, n + next, n + i);
  }

  const shapePoints = points.map(([x, z]) => new THREE.Vector2(x, z));
  const triangles = THREE.ShapeUtils.triangulateShape(shapePoints, []);
  for (const tri of triangles) {
    indices.push(n + tri[0], n + tri[1], n + tri[2]);
    indices.push(tri[2], tri[1], tri[0]);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export function sampleLayerHeight(
  sourceHeight: [number, number] | undefined,
  targetHeight: [number, number] | undefined,
  alpha: number,
): [number, number] {
  const target = targetHeight ?? sourceHeight ?? [0, 1];
  const source = sourceHeight ?? [target[0], target[0]];
  const t = smoothstep(alpha);
  return [lerpNumber(source[0], target[0], t), lerpNumber(source[1], target[1], t)];
}

import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { ConditionPointCloud } from "../../types";

export type PointCloudColorMode = "uniform" | "height";

type ConditionPointCloudProps = {
  pointCloud: ConditionPointCloud | null;
  colorMode: PointCloudColorMode;
  isDemolition?: boolean;
};

function colorForHeight(y: number, minY: number, maxY: number) {
  const t = maxY > minY ? (y - minY) / (maxY - minY) : 0.5;
  const color = new THREE.Color();
  color.setHSL(0.58 - t * 0.42, 0.82, 0.58);
  return color;
}

export function ConditionPointCloud({ pointCloud, colorMode, isDemolition = false }: ConditionPointCloudProps) {
  const geometry = useMemo(() => {
    if (!pointCloud?.available || pointCloud.points.length < 3) return null;
    const positions = new Float32Array(pointCloud.points);
    const colors = new Float32Array((positions.length / 3) * 3);
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let index = 1; index < positions.length; index += 3) {
      minY = Math.min(minY, positions[index]);
      maxY = Math.max(maxY, positions[index]);
    }
    const uniform = new THREE.Color(isDemolition ? "#f97316" : "#e5edf7");
    for (let index = 0; index < positions.length; index += 3) {
      const color = colorMode === "height" ? colorForHeight(positions[index + 1], minY, maxY) : uniform;
      colors[index] = color.r;
      colors[index + 1] = color.g;
      colors[index + 2] = color.b;
    }
    const next = new THREE.BufferGeometry();
    next.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    next.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    next.computeBoundingSphere();
    return next;
  }, [pointCloud, colorMode, isDemolition]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!geometry) return null;
  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.012} vertexColors transparent opacity={0.9} sizeAttenuation />
    </points>
  );
}

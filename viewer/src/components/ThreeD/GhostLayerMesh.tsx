import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { Vec2 } from "../../types";
import { createExtrudedPolygonGeometry } from "../../meshGeometry";

type GhostLayerMeshProps = {
  footprint: Vec2[];
  height: [number, number] | undefined;
  className: "source" | "target" | "delete";
  visible: boolean;
};

const MATERIALS = {
  source: new THREE.MeshStandardMaterial({ color: "#60a5fa", transparent: true, opacity: 0.16, depthWrite: false }),
  target: new THREE.MeshStandardMaterial({ color: "#f59e0b", transparent: true, opacity: 0.14, depthWrite: false }),
  delete: new THREE.MeshStandardMaterial({ color: "#ef4444", transparent: true, opacity: 0.2, depthWrite: false }),
};

export function GhostLayerMesh({ footprint, height, className, visible }: GhostLayerMeshProps) {
  const geometry = useMemo(() => {
    if (!visible || footprint.length < 3) return null;
    const [minY, maxY] = height ?? [0, 1];
    return createExtrudedPolygonGeometry(footprint, minY, maxY);
  }, [footprint, height, visible]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!visible || !geometry) return null;
  return <mesh geometry={geometry} material={MATERIALS[className]} />;
}

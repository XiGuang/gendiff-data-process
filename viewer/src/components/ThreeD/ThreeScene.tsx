import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { useThree } from "@react-three/fiber";
import { Grid, OrbitControls, Stats } from "@react-three/drei";
import * as THREE from "three";
import type { ConditionPointCloud as ConditionPointCloudData, OverlayProxyMode, ViewerLayer, ViewerPair } from "../../types";
import { filterPairLayers } from "../../sequenceUtils";
import { MultiLayerBuilding } from "./MultiLayerBuilding";
import { ConditionPointCloud, type PointCloudColorMode } from "./ConditionPointCloud";

export type ThreeSceneProps = {
  pair: ViewerPair;
  selectedLayerIds: Array<string | number>;
  selectedBuildingIds: Array<string | number | null>;
  alpha: number;
  showSource: boolean;
  showTarget: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showDeleted: boolean;
  showWireframe: boolean;
  showAllLayers: boolean;
  pointCloud?: ConditionPointCloudData | null;
  pointCloudColorMode?: PointCloudColorMode;
  overlayProxyMode?: OverlayProxyMode;
  showProxy?: boolean;
  showPointCloud?: boolean;
};

type SceneBounds = {
  center: [number, number, number];
  radius: number;
  gridSize: number;
};

function includePoint(min: THREE.Vector3, max: THREE.Vector3, x: number, y: number, z: number) {
  min.set(Math.min(min.x, x), Math.min(min.y, y), Math.min(min.z, z));
  max.set(Math.max(max.x, x), Math.max(max.y, y), Math.max(max.z, z));
}

function includeLayerBounds(min: THREE.Vector3, max: THREE.Vector3, layer: ViewerLayer, mode: OverlayProxyMode) {
  const candidates =
    mode === "source"
      ? [[layer.source_points, layer.source_height] as const]
      : mode === "target"
        ? [[layer.target_points, layer.target_height] as const]
        : [
            [layer.source_points, layer.source_height] as const,
            [layer.target_points, layer.target_height] as const,
          ];
  for (const [points, height] of candidates) {
    if (points.length === 0) continue;
    const [minY, maxY] = height ?? [0, 1];
    for (const [x, z] of points) {
      includePoint(min, max, x, minY, z);
      includePoint(min, max, x, maxY, z);
    }
  }
}

function computeSceneBounds(layers: ViewerLayer[], pointCloud: ConditionPointCloudData | null | undefined, includePointCloud: boolean, proxyMode: OverlayProxyMode): SceneBounds {
  const min = new THREE.Vector3(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY);
  const max = new THREE.Vector3(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY);
  for (const layer of layers) {
    includeLayerBounds(min, max, layer, proxyMode);
  }
  if (includePointCloud && pointCloud?.available) {
    for (let index = 0; index + 2 < pointCloud.points.length; index += 3) {
      includePoint(min, max, pointCloud.points[index], pointCloud.points[index + 1], pointCloud.points[index + 2]);
    }
  }
  if (!Number.isFinite(min.x)) {
    return { center: [0, 0, 0], radius: 1, gridSize: 2 };
  }
  const center = min.clone().add(max).multiplyScalar(0.5);
  const size = max.clone().sub(min);
  const radius = Math.max(0.25, size.length() * 0.58);
  return {
    center: [center.x, center.y, center.z],
    radius,
    gridSize: Math.max(0.5, Math.ceil(Math.max(size.x, size.z, 0.25) * 2) / 2),
  };
}

function CameraFit({ bounds }: { bounds: SceneBounds }) {
  const { camera } = useThree();
  const key = `${bounds.center.join(",")}:${bounds.radius}`;
  useEffect(() => {
    const [x, y, z] = bounds.center;
    const distance = Math.max(0.9, bounds.radius * 2.3);
    camera.position.set(x + distance, y + distance * 0.72, z + distance);
    camera.near = Math.max(0.001, distance / 100);
    camera.far = Math.max(50, distance * 100);
    camera.lookAt(x, y, z);
    camera.updateProjectionMatrix();
  }, [camera, key]);
  return null;
}

export function ThreeScene(props: ThreeSceneProps) {
  const hasBuildingSelector = (props.pair.buildings?.length ?? 0) > 1;
  const visibleLayers = useMemo(
    () => filterPairLayers(props.pair, props.selectedLayerIds, props.selectedBuildingIds, props.showAllLayers, !hasBuildingSelector),
    [props.pair, props.selectedLayerIds, props.selectedBuildingIds, props.showAllLayers, hasBuildingSelector],
  );
  const showProxy = props.showProxy ?? true;
  const showPointCloud = props.showPointCloud ?? false;
  const overlayProxyMode = props.overlayProxyMode ?? "edit";
  const bounds = useMemo(() => computeSceneBounds(showProxy ? visibleLayers : [], props.pointCloud, showPointCloud, overlayProxyMode), [showProxy, visibleLayers, props.pointCloud, showPointCloud, overlayProxyMode]);
  if (showProxy && visibleLayers.length === 0) {
    return <div className="empty-viewer-state">Select at least one building and layer.</div>;
  }
  if (showPointCloud && !props.pointCloud?.available && !showProxy) {
    return <div className="empty-viewer-state">Condition point cloud is missing for this pair.</div>;
  }
  return (
    <Canvas className="three-canvas" camera={{ position: [1.6, 1.2, 1.8], fov: 48 }} dpr={[1, 2]}>
      <color attach="background" args={["#08111f"]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[3, 4, 5]} intensity={1.2} />
      <CameraFit bounds={bounds} />
      <Grid args={[bounds.gridSize, bounds.gridSize]} cellSize={0.1} sectionSize={0.5} fadeDistance={8} fadeStrength={1.4} infiniteGrid />
      {showProxy ? <MultiLayerBuilding {...props} overlayProxyMode={overlayProxyMode} /> : null}
      {showPointCloud ? <ConditionPointCloud pointCloud={props.pointCloud ?? null} colorMode={props.pointCloudColorMode ?? "height"} isDemolition={props.pair.is_demolition_pair === true} /> : null}
      <OrbitControls makeDefault enableDamping target={bounds.center} />
      <Stats />
    </Canvas>
  );
}

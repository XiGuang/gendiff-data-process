import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import type { OverlayProxyMode, ViewerLayer } from "../../types";
import { buildPointTracks } from "../../buildTracks";
import { sampleFrame } from "../../sampleFrame";
import { createExtrudedPolygonGeometry, sampleLayerHeight } from "../../meshGeometry";
import { getLayerTrackInputs } from "../../layerTrackInputs";
import { smoothstep } from "../../vec2";
import { GhostLayerMesh } from "./GhostLayerMesh";

export type BuildingLayerMeshProps = {
  layer: ViewerLayer;
  alpha: number;
  showSource: boolean;
  showTarget: boolean;
  showDeleted: boolean;
  showWireframe: boolean;
  showLabels: boolean;
  renderMode?: OverlayProxyMode;
  layerStyleIndex?: number;
};

const ACTION_COLORS: Record<string, string> = {
  KEEP: "#94a3b8",
  MODIFY: "#38bdf8",
  INSERT: "#34d399",
  DELETE: "#fb7185",
  REMOVE: "#fb7185",
};

function actionOf(layer: ViewerLayer): string {
  return String(layer.layer_action ?? "MODIFY").toUpperCase();
}

function actionColor(layer: ViewerLayer): string {
  return ACTION_COLORS[actionOf(layer)] ?? "#facc15";
}

function renderModeColor(layer: ViewerLayer, renderMode: OverlayProxyMode): string {
  if (renderMode === "source") return "#60a5fa";
  if (renderMode === "target") return layer.layer_action === "DELETE" ? "#fb7185" : "#f59e0b";
  return actionColor(layer);
}

function fallbackFootprint(layer: ViewerLayer, alpha: number) {
  if (layer.target_points.length >= 3 && alpha >= 0.5) return layer.target_points;
  if (layer.source_points.length >= 3) return layer.source_points;
  return layer.target_points;
}

function layerLabel(layer: ViewerLayer): string {
  const building = layer.building_name ?? (layer.building_id === null || layer.building_id === undefined ? "building n/a" : `building ${String(layer.building_id)}`);
  const localLayer = layer.building_layer_index ?? layer.level_index ?? layer.layer_order ?? "n/a";
  const proxy = layer.proxy_id ?? layer.target_proxy_id ?? layer.source_proxy_id ?? layer.layer_id;
  const localProxy = layer.local_proxy_id === null || layer.local_proxy_id === undefined ? "" : ` local=${String(layer.local_proxy_id)}`;
  return `${building} L${String(localLayer)} proxy=${String(proxy)}${localProxy}`;
}

export function BuildingLayerMesh({ layer, alpha, showSource, showTarget, showDeleted, showWireframe, showLabels, renderMode = "edit" }: BuildingLayerMeshProps) {
  const result = useMemo(() => {
    try {
      const action = actionOf(layer);
      const t = smoothstep(alpha);
      let footprint = fallbackFootprint(layer, alpha);
      let minY = 0;
      let maxY = 1;
      let opacity = 0.62;

      if (renderMode === "source") {
        footprint = layer.source_points;
        [minY, maxY] = layer.source_height ?? layer.target_height ?? [0, 1];
        opacity = 0.5;
      } else if (renderMode === "target") {
        footprint = layer.target_points;
        [minY, maxY] = layer.target_height ?? layer.source_height ?? [0, 1];
        opacity = 0.5;
      } else if (action === "INSERT") {
        footprint = layer.target_points;
        [minY, maxY] = sampleLayerHeight(layer.source_height, layer.target_height, alpha);
        opacity = 0.2 + 0.48 * t;
      } else if (action === "DELETE" || action === "REMOVE") {
        footprint = layer.source_points;
        const source = layer.source_height ?? layer.target_height ?? [0, 1];
        minY = source[0];
        maxY = source[0] + (source[1] - source[0]) * (1 - t);
        opacity = Math.max(0.08, 0.68 * (1 - t));
      } else if (layer.ops.some((op) => op.type === "DELETE_POINT")) {
        footprint = fallbackFootprint(layer, alpha);
        [minY, maxY] = sampleLayerHeight(layer.source_height, layer.target_height, alpha);
      } else if (layer.ops.length > 0 && layer.source_points.length > 0 && layer.target_points.length > 0) {
        const inputs = getLayerTrackInputs(layer);
        const tracks = buildPointTracks(inputs.sourcePoints, inputs.targetPoints, inputs.ops);
        const frame = sampleFrame(tracks, alpha);
        footprint = frame.activeMainPoints.map((point) => point.coord);
        [minY, maxY] = sampleLayerHeight(layer.source_height, layer.target_height, alpha);
      } else {
        [minY, maxY] = sampleLayerHeight(layer.source_height, layer.target_height, alpha);
      }

      if (footprint.length < 3) {
        return { geometry: null, labelPosition: null, opacity: 0, error: null };
      }
      const center = footprint.reduce<[number, number]>((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
      const labelPosition: [number, number, number] = [center[0] / footprint.length, maxY + 0.01, center[1] / footprint.length];
      return { geometry: createExtrudedPolygonGeometry(footprint, minY, maxY), labelPosition, opacity, error: null };
    } catch (error) {
      return { geometry: null, labelPosition: null, opacity: 0, error: error instanceof Error ? error.message : "failed to build mesh" };
    }
  }, [layer, alpha, renderMode]);

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: renderModeColor(layer, renderMode),
        transparent: true,
        opacity: result.opacity,
        roughness: 0.65,
        metalness: 0.05,
        side: THREE.DoubleSide,
        depthWrite: result.opacity > 0.4,
      }),
    [layer, renderMode, result.opacity],
  );

  useEffect(() => {
    return () => {
      result.geometry?.dispose();
    };
  }, [result.geometry]);

  useEffect(() => {
    return () => {
      material.dispose();
    };
  }, [material]);

  return (
    <group name={`layer-${String(layer.layer_id)}`}>
      {renderMode === "edit" ? (
        <>
          <GhostLayerMesh
            footprint={layer.source_points}
            height={layer.source_height}
            className="source"
            visible={showSource}
          />
          <GhostLayerMesh footprint={layer.source_points} height={layer.source_height} className="delete" visible={showDeleted && (actionOf(layer) === "DELETE" || actionOf(layer) === "REMOVE")} />
          <GhostLayerMesh footprint={layer.target_points} height={layer.target_height} className="target" visible={showTarget} />
        </>
      ) : null}
      {result.geometry ? <mesh geometry={result.geometry} material={material} /> : null}
      {result.geometry && showWireframe ? (
        <lineSegments>
          <edgesGeometry args={[result.geometry]} />
          <lineBasicMaterial color="#f8fafc" transparent opacity={0.42} />
        </lineSegments>
      ) : null}
      {!result.geometry ? (
        <points>
          <bufferGeometry />
          <pointsMaterial size={0.02} color="#f8fafc" />
        </points>
      ) : null}
      {showLabels && result.labelPosition ? <Html position={result.labelPosition} className="label3d">{layerLabel(layer)}</Html> : null}
    </group>
  );
}

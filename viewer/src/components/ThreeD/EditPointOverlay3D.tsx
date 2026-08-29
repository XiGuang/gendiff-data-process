import { Html, Line } from "@react-three/drei";
import type { PointTrack, Vec2, ViewerLayer } from "../../types";
import { buildPointTracks } from "../../buildTracks";
import { sampleFrame } from "../../sampleFrame";
import { sampleLayerHeight } from "../../meshGeometry";
import { getLayerTrackInputs } from "../../layerTrackInputs";

export type EditPointOverlay3DProps = {
  layer: ViewerLayer;
  alpha: number;
  showLabels: boolean;
  showTrails: boolean;
  showDeleted: boolean;
  layerStyleIndex?: number;
};

const POINT_COLORS = {
  keep: "#dbeafe",
  move: "#38bdf8",
  insert: "#34d399",
  delete: "#fb7185",
};

function toVec3(point: Vec2, y: number): [number, number, number] {
  return [point[0], y, point[1]];
}

function overlayEpsilon(minY: number, maxY: number): number {
  const thickness = Math.abs(maxY - minY);
  return Math.max(0.0005, Math.min(0.002, thickness * 0.02));
}

function pointRadius(minY: number, maxY: number): number {
  const thickness = Math.abs(maxY - minY);
  return Math.max(0.0035, Math.min(0.007, thickness * 0.16));
}

function trailFor(track: PointTrack): [Vec2, Vec2] | null {
  if (track.action === "move" && track.sourceCoord && track.targetCoord) return [track.sourceCoord, track.targetCoord];
  if (track.action === "insert" && track.birthCoord && track.targetCoord) return [track.birthCoord, track.targetCoord];
  if (track.action === "delete" && track.sourceCoord && track.collapseCoord) return [track.sourceCoord, track.collapseCoord];
  return null;
}

export function EditPointOverlay3D({ layer, alpha, showLabels, showTrails, showDeleted }: EditPointOverlay3DProps) {
  try {
    const inputs = getLayerTrackInputs(layer);
    const tracks = buildPointTracks(inputs.sourcePoints, inputs.targetPoints, inputs.ops);
    const frame = sampleFrame(tracks, alpha);
    const [minY, maxY] = sampleLayerHeight(layer.source_height, layer.target_height, alpha);
    const epsilon = overlayEpsilon(minY, maxY);
    const y = maxY + epsilon;
    const radius = pointRadius(minY, maxY);
    return (
      <group>
        {showTrails
          ? tracks.map((track) => {
              const trail = trailFor(track);
              if (!trail) return null;
              return (
                <Line
                  key={`trail-${String(track.id)}`}
                  points={[toVec3(trail[0], y), toVec3(trail[1], y)]}
                  color={POINT_COLORS[track.action]}
                  transparent
                  opacity={0.6}
                  lineWidth={1}
                />
              );
            })
          : null}
        {frame.mainPoints.map((point) => (
          <group key={`main-${String(point.id)}`} position={toVec3(point.coord, y)}>
            <mesh>
              <sphereGeometry args={[radius * point.radiusScale, 12, 12]} />
              <meshBasicMaterial color={POINT_COLORS[point.action]} transparent opacity={point.opacity} />
            </mesh>
            {showLabels ? <Html className="label3d">{point.action}:{point.renderOrder}</Html> : null}
          </group>
        ))}
        {showDeleted
          ? frame.deletedPoints.map((point) => (
              <group key={`delete-${String(point.id)}`} position={toVec3(point.coord, y)}>
                <mesh>
                  <sphereGeometry args={[radius * 1.15 * point.radiusScale, 12, 12]} />
                  <meshBasicMaterial color={POINT_COLORS.delete} transparent opacity={point.opacity} />
                </mesh>
                {showLabels ? <Html className="label3d">delete:{point.renderOrder}</Html> : null}
              </group>
            ))
          : null}
      </group>
    );
  } catch {
    return null;
  }
}

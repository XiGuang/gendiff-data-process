import type { PointTrack, SampledFrame, Vec2, ViewerLayer } from "../../types";
import { buildPointTracks } from "../../buildTracks";
import { sampleFrame } from "../../sampleFrame";
import { getLayerTrackInputs } from "../../layerTrackInputs";

export type Bounds2D = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

type Transform2D = {
  x: (value: number) => number;
  y: (value: number) => number;
  scale: number;
};

export function computeBounds(layers: ViewerLayer[]): Bounds2D {
  const points = layers.flatMap((layer) => [...layer.source_points, ...layer.target_points]);
  if (points.length === 0) {
    return { minX: -1, maxX: 1, minZ: -1, maxZ: 1 };
  }
  const xs = points.map((p) => p[0]);
  const zs = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const padX = Math.max((maxX - minX) * 0.08, 0.02);
  const padZ = Math.max((maxZ - minZ) * 0.08, 0.02);
  return { minX: minX - padX, maxX: maxX + padX, minZ: minZ - padZ, maxZ: maxZ + padZ };
}

function makeTransform(bounds: Bounds2D, width: number, height: number): Transform2D {
  const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const spanZ = Math.max(bounds.maxZ - bounds.minZ, 1e-6);
  const scale = Math.min(width / spanX, height / spanZ);
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanZ * scale) / 2;
  return {
    x: (value: number) => offsetX + (value - bounds.minX) * scale,
    y: (value: number) => height - (offsetY + (value - bounds.minZ) * scale),
    scale,
  };
}

function pointsToSvg(points: Vec2[], transform: Transform2D): string {
  return points.map(([x, z]) => `${transform.x(x)},${transform.y(z)}`).join(" ");
}

function sampledToSvg(frame: SampledFrame, transform: Transform2D): string {
  return frame.activeMainPoints.map((point) => `${transform.x(point.coord[0])},${transform.y(point.coord[1])}`).join(" ");
}

function trackTrail(track: PointTrack): [Vec2, Vec2] | null {
  if (track.action === "move" && track.sourceCoord && track.targetCoord) {
    return [track.sourceCoord, track.targetCoord];
  }
  if (track.action === "insert" && track.birthCoord && track.targetCoord) {
    return [track.birthCoord, track.targetCoord];
  }
  if (track.action === "delete" && track.sourceCoord && track.collapseCoord) {
    return [track.sourceCoord, track.collapseCoord];
  }
  return null;
}

export type SvgLayerOverlayProps = {
  layer: ViewerLayer;
  alpha: number;
  bounds: Bounds2D;
  width: number;
  height: number;
  showSource: boolean;
  showTarget: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showDeleted: boolean;
  layerStyleIndex?: number;
  label?: string;
};

export function SvgLayerOverlay({
  layer,
  alpha,
  bounds,
  width,
  height,
  showSource,
  showTarget,
  showLabels,
  showTrails,
  showDeleted,
  layerStyleIndex = 0,
  label,
}: SvgLayerOverlayProps) {
  const transform = makeTransform(bounds, width, height);
  try {
    const inputs = getLayerTrackInputs(layer);
    const tracks = buildPointTracks(inputs.sourcePoints, inputs.targetPoints, inputs.ops);
    const frame = sampleFrame(tracks, alpha);
    const layerClass = `layer-${layerStyleIndex % 5}`;
    return (
      <g className={`svg-layer ${layerClass}`}>
        {showSource && layer.source_points.length >= 3 ? <polygon className="ghost-source" points={pointsToSvg(layer.source_points, transform)} /> : null}
        {showTarget && layer.target_points.length >= 3 ? <polygon className="ghost-target" points={pointsToSvg(layer.target_points, transform)} /> : null}
        {showTrails
          ? tracks.map((track) => {
              const trail = trackTrail(track);
              if (!trail) return null;
              const [a, b] = trail;
              return (
                <line
                  key={`trail-${track.action}-${String(track.id)}`}
                  className={`trail-${track.action}`}
                  x1={transform.x(a[0])}
                  y1={transform.y(a[1])}
                  x2={transform.x(b[0])}
                  y2={transform.y(b[1])}
                />
              );
            })
          : null}
        {frame.activeMainPoints.length >= 3 ? <polygon className="animated-polygon" points={sampledToSvg(frame, transform)} /> : null}
        {frame.mainPoints.map((point) => (
          <g key={`main-${String(point.id)}`} className={`action-${point.action}`}>
            <circle cx={transform.x(point.coord[0])} cy={transform.y(point.coord[1])} r={Math.max(2.5, 4 * point.radiusScale)} opacity={point.opacity} />
            {showLabels ? (
              <text className="point-label" x={transform.x(point.coord[0]) + 5} y={transform.y(point.coord[1]) - 5}>
                {point.action}:{point.renderOrder}
              </text>
            ) : null}
          </g>
        ))}
        {showDeleted
          ? frame.deletedPoints.map((point) => (
              <g key={`delete-${String(point.id)}`} className="action-delete">
                <circle cx={transform.x(point.coord[0])} cy={transform.y(point.coord[1])} r={Math.max(2.5, 4 * point.radiusScale)} opacity={point.opacity} />
                {showLabels ? (
                  <text className="point-label" x={transform.x(point.coord[0]) + 5} y={transform.y(point.coord[1]) - 5} opacity={point.opacity}>
                    delete:{point.renderOrder}
                  </text>
                ) : null}
              </g>
            ))
          : null}
        {label ? (
          <text className="layer-label" x={12} y={22 + layerStyleIndex * 18}>
            {label}
          </text>
        ) : null}
      </g>
    );
  } catch (error) {
    return (
      <g>
        <text className="svg-error" x={16} y={32}>
          {error instanceof Error ? error.message : "failed to build tracks"}
        </text>
      </g>
    );
  }
}

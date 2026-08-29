import type { EditOp, Vec2, ViewerLayer } from "./types";

export function getLayerTrackInputs(layer: ViewerLayer): { sourcePoints: Vec2[]; targetPoints: Vec2[]; ops: EditOp[] } {
  const needsSource = layer.ops.some((op) => op.type === "KEEP_POINT" || op.type === "MOVE_POINT" || op.type === "DELETE_POINT");
  const needsTarget = layer.ops.some((op) => op.type === "KEEP_POINT" || op.type === "MOVE_POINT" || op.type === "INSERT_POINT");
  const sourcePoints = layer.source_points.length > 0 ? layer.source_points : needsSource ? layer.source_points : layer.target_points;
  const targetPoints = layer.target_points.length > 0 ? layer.target_points : needsTarget ? layer.target_points : layer.source_points;
  return { sourcePoints, targetPoints, ops: layer.ops };
}

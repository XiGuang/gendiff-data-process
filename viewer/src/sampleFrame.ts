import type { PointTrack, SampledFrame, SampledPoint } from "./types";
import { clamp01, lerpVec2, smoothstep } from "./vec2";

function requireCoord(track: PointTrack, key: "sourceCoord" | "targetCoord" | "birthCoord" | "collapseCoord") {
  const coord = track[key];
  if (!coord) {
    throw new Error(`${track.action} track ${String(track.id)} is missing ${key}`);
  }
  return coord;
}

export function sampleFrame(tracks: PointTrack[], alpha: number): SampledFrame {
  const clamped = clamp01(alpha);
  const t = smoothstep(clamped);
  const mainPoints: SampledPoint[] = [];
  const activeMainPoints: SampledPoint[] = [];
  const deletedPoints: SampledPoint[] = [];

  for (const track of tracks) {
    if (track.action === "keep") {
      const point = {
        id: track.id,
        action: "keep",
        renderOrder: track.renderOrder,
        coord: lerpVec2(requireCoord(track, "sourceCoord"), requireCoord(track, "targetCoord"), t),
        opacity: 1,
        radiusScale: 1,
      } satisfies SampledPoint;
      mainPoints.push(point);
      activeMainPoints.push(point);
    } else if (track.action === "move") {
      const point = {
        id: track.id,
        action: "move",
        renderOrder: track.renderOrder,
        coord: lerpVec2(requireCoord(track, "sourceCoord"), requireCoord(track, "targetCoord"), t),
        opacity: 1,
        radiusScale: 1,
      } satisfies SampledPoint;
      mainPoints.push(point);
      activeMainPoints.push(point);
    } else if (track.action === "insert") {
      const startAlpha = track.startAlpha ?? 0;
      const endAlpha = track.endAlpha ?? 1;
      const localAlpha = endAlpha <= startAlpha ? (clamped >= endAlpha ? 1 : 0) : clamp01((clamped - startAlpha) / (endAlpha - startAlpha));
      const localT = smoothstep(localAlpha);
      const point = {
        id: track.id,
        action: "insert",
        renderOrder: track.renderOrder,
        coord: lerpVec2(requireCoord(track, "birthCoord"), requireCoord(track, "targetCoord"), localT),
        opacity: localT,
        radiusScale: Math.max(0.15, localT),
      } satisfies SampledPoint;
      mainPoints.push(point);
      if (localAlpha > 0 || (track.endAlpha ?? 1) <= (track.startAlpha ?? 0)) {
        activeMainPoints.push(point);
      }
    } else if (track.action === "delete") {
      deletedPoints.push({
        id: track.id,
        action: "delete",
        renderOrder: track.renderOrder,
        coord: lerpVec2(requireCoord(track, "sourceCoord"), requireCoord(track, "collapseCoord"), t),
        opacity: 1 - t,
        radiusScale: Math.max(0.15, 1 - t),
      });
    }
  }

  mainPoints.sort((a, b) => a.renderOrder - b.renderOrder);
  activeMainPoints.sort((a, b) => a.renderOrder - b.renderOrder);
  deletedPoints.sort((a, b) => a.renderOrder - b.renderOrder);
  return { alpha: clamped, easedAlpha: t, mainPoints, activeMainPoints, deletedPoints };
}

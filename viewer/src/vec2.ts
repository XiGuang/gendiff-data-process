import type { Vec2 } from "./types";

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function smoothstep(value: number): number {
  const t = clamp01(value);
  return t * t * (3 - 2 * t);
}

export function lerpNumber(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function lerpVec2(a: Vec2, b: Vec2, t: number): Vec2 {
  return [lerpNumber(a[0], b[0], t), lerpNumber(a[1], b[1], t)];
}

export function addVec2(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]];
}

export function subVec2(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]];
}

export function mulVec2(a: Vec2, scalar: number): Vec2 {
  return [a[0] * scalar, a[1] * scalar];
}

export function dotVec2(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

export function lengthSqVec2(a: Vec2): number {
  return dotVec2(a, a);
}

export function midpoint(a: Vec2, b: Vec2): Vec2 {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

export function projectionToSegment(point: Vec2, a: Vec2, b: Vec2): Vec2 {
  const ab = subVec2(b, a);
  const lenSq = lengthSqVec2(ab);
  if (lenSq === 0) {
    return [a[0], a[1]];
  }
  const t = clamp01(dotVec2(subVec2(point, a), ab) / lenSq);
  return addVec2(a, mulVec2(ab, t));
}

export function getClosedPrev<T>(items: T[], index: number): T {
  if (items.length === 0) {
    throw new Error("getClosedPrev requires a non-empty array");
  }
  return items[(index - 1 + items.length) % items.length];
}

export function getClosedNext<T>(items: T[], index: number): T {
  if (items.length === 0) {
    throw new Error("getClosedNext requires a non-empty array");
  }
  return items[(index + 1) % items.length];
}

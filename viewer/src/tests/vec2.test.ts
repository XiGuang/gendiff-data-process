import { describe, expect, it } from "vitest";
import { getClosedNext, getClosedPrev, lerpVec2, projectionToSegment, smoothstep } from "../vec2";

describe("vec2 utilities", () => {
  it("computes smoothstep endpoints", () => {
    expect(smoothstep(0)).toBe(0);
    expect(smoothstep(1)).toBe(1);
    expect(smoothstep(-2)).toBe(0);
    expect(smoothstep(2)).toBe(1);
  });

  it("lerps Vec2 values", () => {
    expect(lerpVec2([0, 2], [10, 12], 0.25)).toEqual([2.5, 4.5]);
  });

  it("projects inside and outside a segment", () => {
    expect(projectionToSegment([5, 3], [0, 0], [10, 0])).toEqual([5, 0]);
    expect(projectionToSegment([-5, 3], [0, 0], [10, 0])).toEqual([0, 0]);
    expect(projectionToSegment([15, 3], [0, 0], [10, 0])).toEqual([10, 0]);
  });

  it("wraps closed prev and next", () => {
    expect(getClosedPrev(["a", "b", "c"], 0)).toBe("c");
    expect(getClosedNext(["a", "b", "c"], 2)).toBe("a");
  });
});

import { describe, expect, it } from "vitest";
import { createExtrudedPolygonGeometry, sampleLayerHeight } from "../meshGeometry";
import { validateViewerData } from "../loadViewerData";
import { filterLayers, layerKey } from "../sequenceUtils";

describe("mesh geometry", () => {
  it("creates non-empty triangle and quad geometries", () => {
    const triangle = createExtrudedPolygonGeometry(
      [
        [0, 0],
        [1, 0],
        [0, 1],
      ],
      0,
      1,
    );
    const quad = createExtrudedPolygonGeometry(
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ],
      0,
      1,
    );
    expect(triangle.getAttribute("position").count).toBeGreaterThan(0);
    expect(quad.getAttribute("position").count).toBeGreaterThan(0);
  });

  it("interpolates layer height with smoothstep", () => {
    expect(sampleLayerHeight([0, 1], [2, 5], 0)).toEqual([0, 1]);
    expect(sampleLayerHeight([0, 1], [2, 5], 1)).toEqual([2, 5]);
    expect(sampleLayerHeight([0, 1], [2, 5], 0.5)).toEqual([1, 3]);
  });

  it("starts inserted layers as zero-thickness slabs at the target base", () => {
    expect(sampleLayerHeight(undefined, [2, 5], 0)).toEqual([2, 2]);
    expect(sampleLayerHeight(undefined, [2, 5], 1)).toEqual([2, 5]);
  });

  it("keeps multi-layer data during validation and filtering", () => {
    const data = validateViewerData({
      schema: "edit_sequence_multiview_animation_v1",
      sequence_id: "test",
      pairs: [
        {
          pair_id: "p",
          source_stage_id: "a",
          target_stage_id: "b",
          layers: [
            { layer_id: "0", source_points: [[0, 0]], target_points: [[0, 0]], ops: [] },
            { layer_id: "1", source_points: [[1, 1]], target_points: [[1, 1]], ops: [] },
          ],
        },
      ],
    });
    expect(data.pairs[0].layers).toHaveLength(2);
    expect(filterLayers(data.pairs[0], [layerKey(data.pairs[0].layers[1])]).map((layer) => layer.layer_id)).toEqual(["1"]);
  });
});

import { describe, expect, it } from "vitest";
import { buildPointTracks } from "../buildTracks";
import { getLayerTrackInputs } from "../layerTrackInputs";
import { validateViewerData } from "../loadViewerData";
import { sampleFrame } from "../sampleFrame";

describe("packed viewer data", () => {
  it("preserves direction, pair hash, and packed locator", () => {
    const data = validateViewerData({
      schema: "edit_sequence_multiview_animation_v1",
      sequence_id: "fixture",
      dataset_format: "packed",
      pairs: [
        {
          pair_id: "reverse_pair",
          dataset_locator: "packed:val:0:0",
          source_stage_id: "stage_1",
          target_stage_id: "stage_0",
          change_kind: "demolition",
          pair_hash: "abc123",
          layers: [],
        },
      ],
    });
    expect(data.dataset_format).toBe("packed");
    expect(data.pairs[0].change_kind).toBe("demolition");
    expect(data.pairs[0].pair_hash).toBe("abc123");
    expect(data.pairs[0].dataset_locator).toBe("packed:val:0:0");
  });

  it("plays a deleted layer when the packed target footprint is empty", () => {
    const layer = {
      layer_id: "deleted",
      layer_action: "DELETE",
      source_points: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ] as [number, number][],
      target_points: [] as [number, number][],
      source_height: [0, 1] as [number, number],
      target_height: [0, 0] as [number, number],
      ops: [0, 1, 2, 3].map((source_point_index) => ({
        type: "DELETE_POINT" as const,
        value: { source_point_index },
      })),
    };
    const inputs = getLayerTrackInputs(layer);
    const tracks = buildPointTracks(inputs.sourcePoints, inputs.targetPoints, inputs.ops);
    expect(tracks).toHaveLength(4);
    expect(sampleFrame(tracks, 0).deletedPoints).toHaveLength(4);
    expect(sampleFrame(tracks, 1).deletedPoints.every((point) => point.opacity === 0)).toBe(true);
  });
});

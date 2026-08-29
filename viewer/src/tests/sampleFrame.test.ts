import { describe, expect, it } from "vitest";
import { buildPointTracks } from "../buildTracks";
import { sampleFrame } from "../sampleFrame";
import type { EditOp, Vec2 } from "../types";

describe("sampleFrame", () => {
  const source: Vec2[] = [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2],
  ];
  const target: Vec2[] = [
    [0, 0],
    [1, 0],
    [2, 0],
    [2, 2],
    [0, 2],
  ];

  it("samples MOVE at source and target endpoints", () => {
    const tracks = buildPointTracks(source, target, [
      { type: "MOVE_POINT", value: { source_point_index: 0, target_point_index: 0, target_coord: [1, 1], target_point_id: "m" } },
    ]);
    expect(sampleFrame(tracks, 0).mainPoints[0].coord).toEqual([0, 0]);
    expect(sampleFrame(tracks, 1).mainPoints[0].coord).toEqual([1, 1]);
  });

  it("fades INSERT opacity from 0 to 1", () => {
    const tracks = buildPointTracks(source, target, [
      { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 0, target_point_id: "k" } },
      { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i" } },
    ]);
    const start = sampleFrame(tracks, 0).mainPoints.find((point) => point.renderOrder === 1);
    const end = sampleFrame(tracks, 1).mainPoints.find((point) => point.renderOrder === 1);
    expect(start?.opacity).toBe(0);
    expect(end?.opacity).toBe(1);
  });

  it("fades DELETE opacity from 1 to 0", () => {
    const tracks = buildPointTracks(source, target, [
      { type: "DELETE_POINT", value: { source_point_index: 1, source_point_id: "d" } },
    ]);
    expect(sampleFrame(tracks, 0).deletedPoints[0].opacity).toBe(1);
    expect(sampleFrame(tracks, 1).deletedPoints[0].opacity).toBe(0);
  });

  it("does not put deleted points into mainPoints and sorts outputs", () => {
    const ops: EditOp[] = [
      { type: "DELETE_POINT", value: { source_point_index: 3, source_point_id: "d3" } },
      { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 2, target_point_id: "k2" } },
      { type: "DELETE_POINT", value: { source_point_index: 1, source_point_id: "d1" } },
      { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i1" } },
    ];
    const frame = sampleFrame(buildPointTracks(source, target, ops), 0.5);
    expect(frame.mainPoints.map((point) => point.renderOrder)).toEqual([1, 2]);
    expect(frame.deletedPoints.map((point) => point.renderOrder)).toEqual([1, 3]);
  });

  it("waits for adjacent inserted vertices to complete before starting inner inserts", () => {
    const tracks = buildPointTracks(
      [
        [0, 0],
        [4, 0],
      ],
      [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
      ],
      [
        { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 0, target_point_id: "k0" } },
        { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i1" } },
        { type: "INSERT_POINT", value: { target_point_index: 2, target_coord: [2, 0], target_point_id: "i2" } },
        { type: "INSERT_POINT", value: { target_point_index: 3, target_coord: [3, 0], target_point_id: "i3" } },
        { type: "KEEP_POINT", value: { source_point_index: 1, target_point_index: 4, target_point_id: "k4" } },
      ],
    );
    const beforeMiddleStarts = sampleFrame(tracks, 0.49).mainPoints.find((point) => point.renderOrder === 2);
    const afterMiddleCompletes = sampleFrame(tracks, 1).mainPoints.find((point) => point.renderOrder === 2);
    expect(beforeMiddleStarts?.opacity).toBe(0);
    expect(beforeMiddleStarts?.coord).toEqual([1, 0]);
    expect(afterMiddleCompletes?.opacity).toBe(1);
    expect(afterMiddleCompletes?.coord).toEqual([2, 0]);
  });

  it("keeps waiting insert points out of topology points", () => {
    const tracks = buildPointTracks(
      [
        [0, 0],
        [4, 0],
      ],
      [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
      ],
      [
        { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 0, target_point_id: "k0" } },
        { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i1" } },
        { type: "INSERT_POINT", value: { target_point_index: 2, target_coord: [2, 0], target_point_id: "i2" } },
        { type: "INSERT_POINT", value: { target_point_index: 3, target_coord: [3, 0], target_point_id: "i3" } },
        { type: "KEEP_POINT", value: { source_point_index: 1, target_point_index: 4, target_point_id: "k4" } },
      ],
    );
    const frame = sampleFrame(tracks, 0.49);
    expect(frame.mainPoints.map((point) => point.renderOrder)).toEqual([0, 1, 2, 3, 4]);
    expect(frame.activeMainPoints.map((point) => point.renderOrder)).toEqual([0, 1, 3, 4]);
  });
});

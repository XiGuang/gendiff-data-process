import { describe, expect, it } from "vitest";
import { buildPointTracks } from "../buildTracks";
import type { EditOp, Vec2 } from "../types";

describe("buildPointTracks", () => {
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

  it("builds MOVE tracks from source to absolute target coord", () => {
    const tracks = buildPointTracks(source, target, [
      { type: "MOVE_POINT", value: { source_point_index: 1, target_point_index: 2, target_coord: [2, 0], target_point_id: "m" } },
    ]);
    expect(tracks[0].sourceCoord).toEqual([2, 0]);
    expect(tracks[0].targetCoord).toEqual([2, 0]);
  });

  it("moves INSERT points out from the nearest existing target-topology point", () => {
    const tracks = buildPointTracks(source, target, [
      { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 0, target_point_id: "k0" } },
      { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i" } },
      { type: "KEEP_POINT", value: { source_point_index: 1, target_point_index: 2, target_point_id: "k2" } },
    ]);
    const insert = tracks.find((track) => track.action === "insert");
    expect(insert?.birthCoord).toEqual([0, 0]);
    expect(insert?.targetCoord).toEqual([1, 0]);
  });

  it("uses two immediate seeds for pure inserted layers so inserts grow from two sides", () => {
    const tracks = buildPointTracks(target, target, [
      { type: "INSERT_POINT", value: { target_point_index: 0, target_coord: [0, 0], target_point_id: "seed" } },
      { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i" } },
      { type: "INSERT_POINT", value: { target_point_index: 2, target_coord: [2, 0], target_point_id: "seed2" } },
    ]);
    const seed = tracks.find((track) => track.renderOrder === 0);
    const seed2 = tracks.find((track) => track.renderOrder === 2);
    const insert = tracks.find((track) => track.renderOrder === 1);
    expect(seed?.birthCoord).toEqual([0, 0]);
    expect(seed?.startAlpha).toBe(0);
    expect(seed?.endAlpha).toBe(0);
    expect(seed2?.birthCoord).toEqual([2, 0]);
    expect(seed2?.startAlpha).toBe(0);
    expect(seed2?.endAlpha).toBe(0);
    expect(insert?.birthCoord).toEqual([0, 0]);
  });

  it("chains INSERT timing inward from adjacent existing vertices", () => {
    const chainTarget: Vec2[] = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
    ];
    const tracks = buildPointTracks(
      [
        [0, 0],
        [4, 0],
      ],
      chainTarget,
      [
        { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 0, target_point_id: "k0" } },
        { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i1" } },
        { type: "INSERT_POINT", value: { target_point_index: 2, target_coord: [2, 0], target_point_id: "i2" } },
        { type: "INSERT_POINT", value: { target_point_index: 3, target_coord: [3, 0], target_point_id: "i3" } },
        { type: "KEEP_POINT", value: { source_point_index: 1, target_point_index: 4, target_point_id: "k4" } },
      ],
    );
    const inserts = tracks.filter((track) => track.action === "insert").sort((a, b) => a.renderOrder - b.renderOrder);
    expect(inserts.map((track) => [track.renderOrder, track.birthCoord, track.startAlpha, track.endAlpha])).toEqual([
      [1, [0, 0], 0, 0.5],
      [2, [1, 0], 0.5, 1],
      [3, [4, 0], 0, 0.5],
    ]);
  });

  it("propagates long INSERT runs from both existing ends at the same time", () => {
    const chainTarget: Vec2[] = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
      [5, 0],
      [6, 0],
    ];
    const tracks = buildPointTracks(
      [
        [0, 0],
        [6, 0],
      ],
      chainTarget,
      [
        { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 0, target_point_id: "k0" } },
        { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "i1" } },
        { type: "INSERT_POINT", value: { target_point_index: 2, target_coord: [2, 0], target_point_id: "i2" } },
        { type: "INSERT_POINT", value: { target_point_index: 3, target_coord: [3, 0], target_point_id: "i3" } },
        { type: "INSERT_POINT", value: { target_point_index: 4, target_coord: [4, 0], target_point_id: "i4" } },
        { type: "INSERT_POINT", value: { target_point_index: 5, target_coord: [5, 0], target_point_id: "i5" } },
        { type: "KEEP_POINT", value: { source_point_index: 1, target_point_index: 6, target_point_id: "k6" } },
      ],
    );
    const inserts = tracks.filter((track) => track.action === "insert").sort((a, b) => a.renderOrder - b.renderOrder);
    expect(inserts.map((track) => [track.renderOrder, track.birthCoord, track.startAlpha, track.endAlpha])).toEqual([
      [1, [0, 0], 0, 1 / 3],
      [2, [1, 0], 1 / 3, 2 / 3],
      [3, [2, 0], 2 / 3, 1],
      [4, [5, 0], 1 / 3, 2 / 3],
      [5, [6, 0], 0, 1 / 3],
    ]);
  });

  it("places DELETE collapseCoord on the source neighbor edge", () => {
    const tracks = buildPointTracks(source, target, [
      { type: "DELETE_POINT", value: { source_point_index: 1, source_point_id: "d" } },
    ]);
    expect(tracks[0].collapseCoord).toEqual([1, 1]);
  });

  it("keeps target render ordering", () => {
    const ops: EditOp[] = [
      { type: "KEEP_POINT", value: { source_point_index: 0, target_point_index: 2, target_point_id: "a" } },
      { type: "INSERT_POINT", value: { target_point_index: 1, target_coord: [1, 0], target_point_id: "b" } },
    ];
    const tracks = buildPointTracks(source, target, ops);
    expect(tracks.map((track) => track.renderOrder)).toEqual([2, 1]);
  });
});

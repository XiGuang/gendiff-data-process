import type { EditOp, PointTrack, TrackBuildOptions, Vec2 } from "./types";
import { getClosedNext, getClosedPrev, midpoint, projectionToSegment } from "./vec2";

function assertNonEmpty(points: Vec2[], name: string): void {
  if (points.length === 0) {
    throw new Error(`${name} must contain at least one point`);
  }
}

function pointAt(points: Vec2[], index: number, name: string): Vec2 {
  if (!Number.isInteger(index) || index < 0 || index >= points.length) {
    throw new Error(`${name} index ${index} is out of range for ${points.length} points`);
  }
  return points[index];
}

function stableId(value: unknown, fallback: string): string | number {
  return typeof value === "string" || typeof value === "number" ? value : fallback;
}

function centroid(points: Vec2[]): Vec2 {
  const sum = points.reduce<Vec2>((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
  return [sum[0] / points.length, sum[1] / points.length];
}

function closedIndexDistance(a: number, b: number, count: number): number {
  const distance = Math.abs(a - b);
  return Math.min(distance, count - distance);
}

type InsertSeed = {
  op: Extract<EditOp, { type: "INSERT_POINT" }>;
  opIndex: number;
};

type ResolvedInsert = InsertSeed & {
  generation: number;
  birthCoord: Vec2;
  immediate?: boolean;
};

function existingCoord(track: PointTrack): Vec2 {
  return track.sourceCoord ?? track.targetCoord ?? track.birthCoord ?? [0, 0];
}

function resolveDependentInserts(
  deferredInserts: InsertSeed[],
  targetPoints: Vec2[],
  existingByTargetIndex: Map<number, PointTrack>,
): ResolvedInsert[] {
  const pending = new Map<number, InsertSeed>();
  const available = new Map<number, { coord: Vec2; generation: number }>();
  for (const [targetIndex, track] of existingByTargetIndex) {
    available.set(targetIndex, { coord: existingCoord(track), generation: -1 });
  }
  for (const item of deferredInserts) {
    pending.set(item.op.value.target_point_index, item);
  }

  const resolved: ResolvedInsert[] = [];
  if (available.size === 0 && pending.size > 0) {
    const sortedPending = [...pending.values()].sort((a, b) => a.op.value.target_point_index - b.op.value.target_point_index);
    const first = sortedPending[0];
    const second = sortedPending
      .slice(1)
      .sort(
        (a, b) =>
          closedIndexDistance(b.op.value.target_point_index, first.op.value.target_point_index, targetPoints.length) -
            closedIndexDistance(a.op.value.target_point_index, first.op.value.target_point_index, targetPoints.length) ||
          a.op.value.target_point_index - b.op.value.target_point_index,
      )[0];
    for (const seed of [first, second].filter((item): item is InsertSeed => Boolean(item))) {
      const targetIndex = seed.op.value.target_point_index;
      resolved.push({ ...seed, generation: 0, birthCoord: seed.op.value.target_coord, immediate: true });
      available.set(targetIndex, { coord: seed.op.value.target_coord, generation: -1 });
      pending.delete(targetIndex);
    }
  }

  while (pending.size > 0) {
    let progressed = false;
    const availableAtGenerationStart = new Map(available);
    const newlyAvailable: Array<{ targetIndex: number; coord: Vec2; generation: number }> = [];
    const batch = [...pending.values()].sort((a, b) => a.op.value.target_point_index - b.op.value.target_point_index);
    for (const item of batch) {
      const targetIndex = item.op.value.target_point_index;
      const prevIndex = (targetIndex - 1 + targetPoints.length) % targetPoints.length;
      const nextIndex = (targetIndex + 1) % targetPoints.length;
      const prev = availableAtGenerationStart.get(prevIndex);
      const next = availableAtGenerationStart.get(nextIndex);
      const parent =
        prev && next
          ? prev.generation <= next.generation
            ? prev
            : next
          : prev ?? next;
      if (!parent) {
        continue;
      }
      const generation = parent.generation + 1;
      resolved.push({ ...item, generation, birthCoord: parent.coord });
      newlyAvailable.push({ targetIndex, coord: item.op.value.target_coord, generation });
      pending.delete(targetIndex);
      progressed = true;
    }
    for (const item of newlyAvailable) {
      available.set(item.targetIndex, { coord: item.coord, generation: item.generation });
    }
    if (!progressed) {
      const fallback = [...pending.values()].sort((a, b) => {
        const da = Math.min(...[...available.keys()].map((index) => closedIndexDistance(a.op.value.target_point_index, index, targetPoints.length)));
        const db = Math.min(...[...available.keys()].map((index) => closedIndexDistance(b.op.value.target_point_index, index, targetPoints.length)));
        return da - db || a.op.value.target_point_index - b.op.value.target_point_index;
      })[0];
      resolved.push({ ...fallback, generation: 0, birthCoord: fallback.op.value.target_coord, immediate: true });
      available.set(fallback.op.value.target_point_index, { coord: fallback.op.value.target_coord, generation: -1 });
      pending.delete(fallback.op.value.target_point_index);
    }
  }
  return resolved;
}

export function buildPointTracks(
  sourcePoints: Vec2[],
  targetPoints: Vec2[],
  editOps: EditOp[],
  options: TrackBuildOptions = {},
): PointTrack[] {
  assertNonEmpty(sourcePoints, "sourcePoints");
  assertNonEmpty(targetPoints, "targetPoints");

  const insertBirthMode = options.insertBirthMode ?? "nearest_existing_point";
  const deleteCollapseMode = options.deleteCollapseMode ?? "source_edge_projection";
  const tracks: PointTrack[] = [];
  const existingByTargetIndex = new Map<number, PointTrack>();
  const deferredInserts: Array<{ op: Extract<EditOp, { type: "INSERT_POINT" }>; opIndex: number }> = [];

  for (const [opIndex, op] of editOps.entries()) {
    switch (op.type) {
      case "KEEP_POINT": {
        const value = op.value;
        const sourceCoord = pointAt(sourcePoints, value.source_point_index, "source_point_index");
        const targetCoord = pointAt(targetPoints, value.target_point_index, "target_point_index");
        const track: PointTrack = {
          id: stableId(value.target_point_id, `keep-${opIndex}`),
          action: "keep",
          renderOrder: value.target_point_index,
          sourceCoord,
          targetCoord,
        };
        existingByTargetIndex.set(value.target_point_index, track);
        tracks.push(track);
        break;
      }
      case "MOVE_POINT": {
        const value = op.value;
        const sourceCoord = pointAt(sourcePoints, value.source_point_index, "source_point_index");
        pointAt(targetPoints, value.target_point_index, "target_point_index");
        const track: PointTrack = {
          id: stableId(value.target_point_id, `move-${opIndex}`),
          action: "move",
          renderOrder: value.target_point_index,
          sourceCoord,
          targetCoord: value.target_coord,
        };
        existingByTargetIndex.set(value.target_point_index, track);
        tracks.push(track);
        break;
      }
      case "INSERT_POINT": {
        deferredInserts.push({ op, opIndex });
        break;
      }
      case "DELETE_POINT": {
        const value = op.value;
        const sourceCoord = pointAt(sourcePoints, value.source_point_index, "source_point_index");
        const prevSource = getClosedPrev(sourcePoints, value.source_point_index);
        const nextSource = getClosedNext(sourcePoints, value.source_point_index);
        let collapseCoord: Vec2;
        if (deleteCollapseMode === "source_edge_midpoint") {
          collapseCoord = midpoint(prevSource, nextSource);
        } else if (deleteCollapseMode === "previous_point") {
          collapseCoord = prevSource;
        } else if (deleteCollapseMode === "next_point") {
          collapseCoord = nextSource;
        } else {
          collapseCoord = projectionToSegment(sourceCoord, prevSource, nextSource);
        }
        tracks.push({
          id: stableId(value.source_point_id, `delete-${opIndex}`),
          action: "delete",
          renderOrder: value.source_point_index,
          sourceCoord,
          collapseCoord,
        });
        break;
      }
      default:
        throw new Error(`unsupported edit op type ${(op as { type?: string }).type}`);
    }
  }

  const resolvedInserts =
    insertBirthMode === "nearest_existing_point"
      ? resolveDependentInserts(deferredInserts, targetPoints, existingByTargetIndex)
      : deferredInserts.map<ResolvedInsert>(({ op, opIndex }) => {
          const value = op.value;
          pointAt(targetPoints, value.target_point_index, "target_point_index");
          const prevTarget = getClosedPrev(targetPoints, value.target_point_index);
          const nextTarget = getClosedNext(targetPoints, value.target_point_index);
          const birthCoord =
            insertBirthMode === "target_edge_midpoint"
              ? midpoint(prevTarget, nextTarget)
              : insertBirthMode === "target_centroid"
                ? centroid(targetPoints)
                : projectionToSegment(value.target_coord, prevTarget, nextTarget);
          return { op, opIndex, generation: 0, birthCoord };
        });
  const maxGeneration = Math.max(0, ...resolvedInserts.map((insert) => insert.generation));
  const generationCount = maxGeneration + 1;

  for (const { op, opIndex, generation, birthCoord, immediate } of resolvedInserts) {
    const value = op.value;
    pointAt(targetPoints, value.target_point_index, "target_point_index");
    tracks.push({
      id: stableId(value.target_point_id, `insert-${opIndex}`),
      action: "insert",
      renderOrder: value.target_point_index,
      targetCoord: value.target_coord,
      birthCoord,
      startAlpha: immediate ? 0 : generation / generationCount,
      endAlpha: immediate ? 0 : (generation + 1) / generationCount,
    });
  }

  return tracks;
}

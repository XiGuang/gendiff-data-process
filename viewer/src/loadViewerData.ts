import type { EditOp, Vec2, ViewerBuildingSummary, ViewerData, ViewerLayer, ViewerPair } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asVec2Array(value: unknown, path: string): Vec2[] {
  if (!Array.isArray(value)) {
    throw new Error(`${path} must be an array`);
  }
  return value.map((point, index) => {
    if (!Array.isArray(point) || point.length < 2 || typeof point[0] !== "number" || typeof point[1] !== "number") {
      throw new Error(`${path}[${index}] must be [number, number]`);
    }
    return [point[0], point[1]];
  });
}

function asHeight(value: unknown): [number, number] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value) || value.length < 2 || typeof value[0] !== "number" || typeof value[1] !== "number") {
    throw new Error("height must be [number, number]");
  }
  return [value[0], value[1]];
}

function asStringNumberNull(value: unknown): string | number | null | undefined {
  if (value === undefined) return undefined;
  if (value === null || typeof value === "string" || typeof value === "number") return value;
  return undefined;
}

function asNumberNull(value: unknown): number | null | undefined {
  if (value === undefined) return undefined;
  if (value === null || typeof value === "number") return value;
  return undefined;
}

function validateOps(value: unknown, path: string): EditOp[] {
  if (!Array.isArray(value)) {
    throw new Error(`${path} must be an array`);
  }
  return value.map((op, index) => {
    if (!isRecord(op) || typeof op.type !== "string" || !isRecord(op.value)) {
      throw new Error(`${path}[${index}] must be an edit op with a value object`);
    }
    if (!["KEEP_POINT", "MOVE_POINT", "INSERT_POINT", "DELETE_POINT"].includes(op.type)) {
      throw new Error(`${path}[${index}] has unsupported op type ${op.type}`);
    }
    return op as EditOp;
  });
}

function validateLayer(value: unknown, path: string): ViewerLayer {
  if (!isRecord(value)) {
    throw new Error(`${path} must be an object`);
  }
  if (typeof value.layer_id !== "string" && typeof value.layer_id !== "number") {
    throw new Error(`${path}.layer_id must be string or number`);
  }
  return {
    layer_id: value.layer_id,
    layer_order: typeof value.layer_order === "number" ? value.layer_order : undefined,
    level_index: typeof value.level_index === "number" || value.level_index === null ? value.level_index : undefined,
    building_id: asStringNumberNull(value.building_id),
    building_name: typeof value.building_name === "string" || value.building_name === null ? value.building_name : undefined,
    building_stage_name: typeof value.building_stage_name === "string" || value.building_stage_name === null ? value.building_stage_name : undefined,
    building_layer_index: asNumberNull(value.building_layer_index),
    local_proxy_id: asStringNumberNull(value.local_proxy_id),
    proxy_id: asStringNumberNull(value.proxy_id),
    source_building_id: asStringNumberNull(value.source_building_id),
    target_building_id: asStringNumberNull(value.target_building_id),
    source_building_layer_index: asNumberNull(value.source_building_layer_index),
    target_building_layer_index: asNumberNull(value.target_building_layer_index),
    source_proxy_id: asStringNumberNull(value.source_proxy_id),
    target_proxy_id: asStringNumberNull(value.target_proxy_id),
    source_layer_index: asNumberNull(value.source_layer_index),
    target_layer_index: asNumberNull(value.target_layer_index),
    layer_action: typeof value.layer_action === "string" ? value.layer_action : undefined,
    source_points: asVec2Array(value.source_points, `${path}.source_points`),
    target_points: asVec2Array(value.target_points, `${path}.target_points`),
    source_height: asHeight(value.source_height),
    target_height: asHeight(value.target_height),
    ops: validateOps(value.ops, `${path}.ops`),
    debug_edit_objects: Array.isArray(value.debug_edit_objects) ? value.debug_edit_objects : undefined,
  };
}

function validateBuildings(value: unknown): ViewerBuildingSummary[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value
    .filter(isRecord)
    .map((item) => ({
      building_id: asStringNumberNull(item.building_id) ?? null,
      building_name: typeof item.building_name === "string" || item.building_name === null ? item.building_name : undefined,
      source_stage_name: typeof item.source_stage_name === "string" || item.source_stage_name === null ? item.source_stage_name : undefined,
      target_stage_name: typeof item.target_stage_name === "string" || item.target_stage_name === null ? item.target_stage_name : undefined,
      source_stage_index: asNumberNull(item.source_stage_index),
      target_stage_index: asNumberNull(item.target_stage_index),
      source_stage_position: asNumberNull(item.source_stage_position),
      target_stage_position: asNumberNull(item.target_stage_position),
    }));
}

function validatePair(value: unknown, index: number): ViewerPair {
  if (!isRecord(value)) {
    throw new Error(`pairs[${index}] must be an object`);
  }
  if (typeof value.pair_id !== "string" || typeof value.source_stage_id !== "string" || typeof value.target_stage_id !== "string") {
    throw new Error(`pairs[${index}] is missing pair/stage ids`);
  }
  if (!Array.isArray(value.layers)) {
    throw new Error(`pairs[${index}].layers must be an array`);
  }
  return {
    pair_id: value.pair_id,
    dataset_locator: typeof value.dataset_locator === "string" ? value.dataset_locator : undefined,
    source_stage_id: value.source_stage_id,
    target_stage_id: value.target_stage_id,
    source_state_id: typeof value.source_state_id === "string" || value.source_state_id === null ? value.source_state_id : undefined,
    target_state_id: typeof value.target_state_id === "string" || value.target_state_id === null ? value.target_state_id : undefined,
    edit_sequence_path: typeof value.edit_sequence_path === "string" ? value.edit_sequence_path : undefined,
    edit_object_path: typeof value.edit_object_path === "string" ? value.edit_object_path : null,
    layers: value.layers.map((layer, layerIndex) => validateLayer(layer, `pairs[${index}].layers[${layerIndex}]`)),
    buildings: validateBuildings(value.buildings),
    include_demolition: typeof value.include_demolition === "boolean" || value.include_demolition === null ? value.include_demolition : undefined,
    is_demolition_pair: typeof value.is_demolition_pair === "boolean" || value.is_demolition_pair === null ? value.is_demolition_pair : undefined,
    change_kind: typeof value.change_kind === "string" || value.change_kind === null ? value.change_kind : undefined,
    pair_hash: typeof value.pair_hash === "string" || value.pair_hash === null ? value.pair_hash : undefined,
    source_state_meta: value.source_state_meta,
    target_state_meta: value.target_state_meta,
    metadata: isRecord(value.metadata) ? value.metadata : undefined,
    validation: isRecord(value.validation) ? value.validation : undefined,
  };
}

export function validateViewerData(data: unknown): ViewerData {
  if (!isRecord(data)) {
    throw new Error("viewer data must be an object");
  }
  if (data.schema !== "edit_sequence_multiview_animation_v1") {
    throw new Error("viewer data schema must be edit_sequence_multiview_animation_v1");
  }
  if (!Array.isArray(data.pairs)) {
    throw new Error("viewer data pairs must be an array");
  }
  return {
    schema: "edit_sequence_multiview_animation_v1",
    sequence_id: typeof data.sequence_id === "string" ? data.sequence_id : "unknown_sequence",
    dataset_dir: typeof data.dataset_dir === "string" ? data.dataset_dir : undefined,
    dataset_kind: typeof data.dataset_kind === "string" ? data.dataset_kind : undefined,
    dataset_format: typeof data.dataset_format === "string" ? data.dataset_format : undefined,
    normalization: isRecord(data.normalization) ? data.normalization : undefined,
    dataset_meta: data.dataset_meta,
    pairs: data.pairs.map(validatePair),
  };
}

export async function loadViewerData(url: string): Promise<ViewerData> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load viewer data from ${url}: ${response.status} ${response.statusText}`);
  }
  return validateViewerData(await response.json());
}

export function loadDefaultViewerData(): Promise<ViewerData> {
  return loadViewerData("/data/default.viewer.json");
}

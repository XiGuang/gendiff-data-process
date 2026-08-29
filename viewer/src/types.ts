export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export type PointId = string | number | null | undefined;

export type KeepPointOp = {
  type: "KEEP_POINT";
  value: {
    source_point_id?: PointId;
    target_point_id?: PointId;
    source_point_index: number;
    target_point_index: number;
  };
};

export type MovePointOp = {
  type: "MOVE_POINT";
  value: {
    source_point_id?: PointId;
    target_point_id?: PointId;
    source_point_index: number;
    target_point_index: number;
    target_coord: Vec2;
  };
};

export type InsertPointOp = {
  type: "INSERT_POINT";
  value: {
    target_point_id?: PointId;
    target_point_index: number;
    target_coord: Vec2;
  };
};

export type DeletePointOp = {
  type: "DELETE_POINT";
  value: {
    source_point_id?: PointId;
    source_point_index: number;
  };
};

export type EditOp = KeepPointOp | MovePointOp | InsertPointOp | DeletePointOp;

export type LayerAction = "KEEP" | "MODIFY" | "INSERT" | "DELETE" | string;

export type ViewerLayer = {
  layer_id: string | number;
  layer_order?: number;
  level_index?: number | null;
  building_id?: string | number | null;
  building_name?: string | null;
  building_stage_name?: string | null;
  building_layer_index?: number | null;
  local_proxy_id?: string | number | null;
  proxy_id?: string | number | null;
  source_building_id?: string | number | null;
  target_building_id?: string | number | null;
  source_building_layer_index?: number | null;
  target_building_layer_index?: number | null;
  source_proxy_id?: string | number | null;
  target_proxy_id?: string | number | null;
  source_layer_index?: number | null;
  target_layer_index?: number | null;
  layer_action?: LayerAction;
  source_points: Vec2[];
  target_points: Vec2[];
  source_height?: [number, number];
  target_height?: [number, number];
  ops: EditOp[];
  debug_edit_objects?: unknown[];
};

export type ViewerBuildingSummary = {
  building_id: string | number | null;
  building_name?: string | null;
  source_stage_name?: string | null;
  target_stage_name?: string | null;
  source_stage_index?: number | null;
  target_stage_index?: number | null;
  source_stage_position?: number | null;
  target_stage_position?: number | null;
};

export type ViewerPair = {
  pair_id: string;
  dataset_locator?: string;
  source_stage_id: string;
  target_stage_id: string;
  source_state_id?: string | null;
  target_state_id?: string | null;
  edit_sequence_path?: string;
  edit_object_path?: string | null;
  layers: ViewerLayer[];
  buildings?: ViewerBuildingSummary[];
  include_demolition?: boolean | null;
  is_demolition_pair?: boolean | null;
  change_kind?: "construction" | "demolition" | "mixed" | string | null;
  pair_hash?: string | null;
  source_state_meta?: unknown;
  target_state_meta?: unknown;
  metadata?: Record<string, unknown>;
  validation?: Record<string, unknown>;
};

export type ViewerData = {
  schema: "edit_sequence_multiview_animation_v1";
  sequence_id: string;
  dataset_dir?: string;
  dataset_kind?: "building" | "area" | string;
  dataset_format?: "raw" | "packed" | string;
  normalization?: Record<string, unknown>;
  dataset_meta?: unknown;
  pairs: ViewerPair[];
};

export type DatasetSummary = {
  datasetDir: string;
  datasetName: string;
  datasetKind?: "building" | "area" | string;
  datasetFormat?: "raw" | "packed" | string;
  splits: Record<string, number>;
  pairTotal: number;
  stateTotal?: number;
  stageTotal?: number;
  hasConditions?: boolean;
  datasetMeta?: Record<string, unknown>;
  normalization?: Record<string, unknown>;
};

export type PairListItem = {
  pairId: string;
  pairLocator?: string | null;
  sourceState?: string | null;
  targetState?: string | null;
  isDemolitionPair?: boolean | null;
  includeDemolition?: boolean | null;
  validationOk?: boolean | null;
  conditionPointCount?: number | null;
  changeKind?: "construction" | "demolition" | "mixed" | string | null;
  pairHash?: string | null;
};

export type PairListPage = {
  datasetDir: string;
  datasetFormat?: "raw" | "packed" | string;
  split: string;
  query: string;
  offset: number;
  limit: number;
  total: number;
  searchScanned?: number;
  searchTruncated?: boolean;
  pairs: PairListItem[];
};

export type ConditionPointCloud = {
  pairId: string;
  pairLocator?: string | null;
  available: boolean;
  path?: string | null;
  totalPoints: number;
  sampledPoints: number;
  stride: 3;
  points: number[];
};

export type PointTrack = {
  id: string | number;
  action: "keep" | "move" | "insert" | "delete";
  renderOrder: number;
  sourceCoord?: Vec2;
  targetCoord?: Vec2;
  birthCoord?: Vec2;
  collapseCoord?: Vec2;
  startAlpha?: number;
  endAlpha?: number;
};

export type SampledPoint = {
  id: string | number;
  action: PointTrack["action"];
  renderOrder: number;
  coord: Vec2;
  opacity: number;
  radiusScale: number;
};

export type SampledFrame = {
  alpha: number;
  easedAlpha: number;
  mainPoints: SampledPoint[];
  activeMainPoints: SampledPoint[];
  deletedPoints: SampledPoint[];
};

export type SampledLayerFrame = {
  layer: ViewerLayer;
  frame: SampledFrame;
};

export type TrackBuildOptions = {
  insertBirthMode?: "nearest_existing_point" | "target_edge_projection" | "target_edge_midpoint" | "target_centroid";
  deleteCollapseMode?: "source_edge_projection" | "source_edge_midpoint" | "previous_point" | "next_point";
};

export type ViewMode = "2d" | "3d";
export type MainViewTab = "proxy3d" | "point_cloud" | "overlay" | "playback";
export type OverlayProxyMode = "source" | "target" | "edit";
export type TwoDMode = "single_layer" | "multi_layer_overlay" | "small_multiples";

import type { ViewerPair } from "../types";
import { countOps, filterPairLayers, layerKey } from "../sequenceUtils";

type DebugPanelProps = {
  pair: ViewerPair;
  selectedLayerIds: Array<string | number>;
  selectedBuildingIds: Array<string | number | null>;
};

function val(value: unknown): string {
  if (value === undefined || value === null) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toExponential(3);
  return String(value);
}

export function DebugPanel({ pair, selectedLayerIds, selectedBuildingIds }: DebugPanelProps) {
  const hasBuildingSelector = (pair.buildings?.length ?? 0) > 1;
  const selectedLayers = filterPairLayers(pair, selectedLayerIds, selectedBuildingIds, false, !hasBuildingSelector);
  const validation = pair.validation ?? {};
  const metadata = pair.metadata ?? {};
  const normalization = metadata.normalization_stats;
  return (
    <section className="panel-section debug-panel">
      <div className="section-title">Debug</div>
      <dl>
        <dt>pair</dt><dd>{pair.pair_id}</dd>
        <dt>source</dt><dd>{pair.source_stage_id}</dd>
        <dt>target</dt><dd>{pair.target_stage_id}</dd>
        <dt>direction</dt><dd>{val(pair.change_kind ?? metadata.change_kind)}</dd>
        <dt>pair hash</dt><dd>{val(pair.pair_hash ?? metadata.pair_hash)}</dd>
        <dt>locator</dt><dd>{val(pair.dataset_locator)}</dd>
        <dt>buildings</dt><dd>{selectedBuildingIds.map(String).join(", ") || (hasBuildingSelector ? "none" : "all")}</dd>
        <dt>selected</dt><dd>{selectedLayerIds.map(String).join(", ") || "none"}</dd>
        <dt>layer count</dt><dd>{pair.layers.length}</dd>
        <dt>demolition</dt><dd>{val(pair.is_demolition_pair)}</dd>
        <dt>schema</dt><dd>{val(metadata.edit_schema_version)}</dd>
        <dt>condition pts</dt><dd>{val(metadata.condition_point_count)}</dd>
        <dt>max tokens</dt><dd>{val(metadata.max_ar_tokens_required)}</dd>
      </dl>
      <div className="validation-grid">
        <span>layer match</span><strong>{val(validation.reconstructed_layer_count_match)}</strong>
        <span>point match</span><strong>{val(validation.reconstructed_point_count_match)}</strong>
        <span>max coord err</span><strong>{val(validation.max_coord_error)}</strong>
        <span>max height err</span><strong>{val(validation.max_height_error)}</strong>
      </div>
      {typeof normalization === "object" && normalization !== null ? (
        <div className="layer-debug">
          <strong>Normalization</strong>
          {Object.entries(normalization).map(([key, value]) => (
            <span key={key}>{key} {val(value)}</span>
          ))}
        </div>
      ) : null}
      {selectedLayers.map((layer) => {
        const counts = countOps(layer);
        return (
          <div className="layer-debug" key={layerKey(layer)}>
            <strong>Layer {String(layer.layer_id)}</strong>
            <span>building {layer.building_name ?? layer.building_id ?? "n/a"} | action {layer.layer_action ?? "n/a"}</span>
            <span>building layer {layer.building_layer_index ?? "n/a"} | proxy {layer.proxy_id ?? "n/a"} | local proxy {layer.local_proxy_id ?? "n/a"}</span>
            <span>source points {layer.source_points.length}</span>
            <span>target points {layer.target_points.length}</span>
            <span>ops {Object.entries(counts).map(([key, count]) => `${key}:${count}`).join(" ") || "none"}</span>
            <span>edit objects {layer.debug_edit_objects?.length ?? 0}</span>
            <span>source height {JSON.stringify(layer.source_height ?? [0, 1])}</span>
            <span>target height {JSON.stringify(layer.target_height ?? [0, 1])}</span>
          </div>
        );
      })}
    </section>
  );
}

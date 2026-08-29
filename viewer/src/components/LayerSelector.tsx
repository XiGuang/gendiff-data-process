import type { ViewerLayer } from "../types";
import { layerKey, sortLayers } from "../sequenceUtils";

type LayerSelectorProps = {
  layers: ViewerLayer[];
  selectedLayerIds: Array<string | number>;
  onChange: (ids: Array<string | number>) => void;
};

export function LayerSelector({ layers, selectedLayerIds, onChange }: LayerSelectorProps) {
  const sorted = sortLayers(layers);
  const selected = new Set(selectedLayerIds.map(String));
  const firstSelected = sorted.find((layer) => selected.has(layerKey(layer))) ?? sorted[0];

  function toggle(layer: ViewerLayer) {
    const lid = layerKey(layer);
    const next = new Set(selected);
    if (next.has(lid)) next.delete(lid);
    else next.add(lid);
    onChange(sorted.filter((candidate) => next.has(layerKey(candidate))).map(layerKey));
  }

  return (
    <section className="panel-section">
      <div className="section-title">Layers</div>
      <div className="button-row">
        <button type="button" onClick={() => onChange(sorted.map(layerKey))}>Select all</button>
        <button type="button" onClick={() => onChange([])}>Clear</button>
        <button type="button" onClick={() => firstSelected && onChange([layerKey(firstSelected)])}>Only current</button>
      </div>
      <div className="layer-list">
        {sorted.map((layer) => (
          <label key={layerKey(layer)} className="layer-option">
            <input type="checkbox" checked={selected.has(layerKey(layer))} onChange={() => toggle(layer)} />
            <span>
              <strong>{String(layer.layer_id)}</strong>
              <small>
                {layer.building_name ?? `bld ${layer.building_id ?? "n/a"}`} | L{layer.building_layer_index ?? layer.level_index ?? "n/a"} | proxy {layer.proxy_id ?? layer.layer_id} | local {layer.local_proxy_id ?? "n/a"} | {layer.layer_action ?? "edit"} | src {layer.source_points.length} | tgt {layer.target_points.length}
              </small>
            </span>
          </label>
        ))}
      </div>
    </section>
  );
}

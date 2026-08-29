import type { ViewerLayer } from "../../types";
import { layerKey, sortLayers } from "../../sequenceUtils";
import { computeBounds, SvgLayerOverlay } from "./SvgLayerOverlay";

export type SvgMultiLayerAnimatorProps = {
  layers: ViewerLayer[];
  selectedLayerIds: Array<string | number>;
  alpha: number;
  showSource: boolean;
  showTarget: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showDeleted: boolean;
};

const WIDTH = 1200;
const HEIGHT = 820;

export function SvgMultiLayerAnimator({
  layers,
  selectedLayerIds,
  alpha,
  showSource,
  showTarget,
  showLabels,
  showTrails,
  showDeleted,
}: SvgMultiLayerAnimatorProps) {
  const selected = new Set(selectedLayerIds.map(String));
  const visibleLayers = sortLayers(layers.filter((layer) => selected.has(layerKey(layer))));
  const bounds = computeBounds(visibleLayers);
  if (visibleLayers.length === 0) {
    return <div className="empty-viewer-state">Select at least one layer.</div>;
  }
  return (
    <svg className="viewer-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img">
      {visibleLayers.map((layer, index) => (
        <SvgLayerOverlay
          key={layerKey(layer)}
          layer={layer}
          alpha={alpha}
          bounds={bounds}
          width={WIDTH}
          height={HEIGHT}
          showSource={showSource}
          showTarget={showTarget}
          showLabels={showLabels}
          showTrails={showTrails}
          showDeleted={showDeleted}
          layerStyleIndex={index}
          label={`layer ${String(layer.layer_id)} level ${layer.level_index ?? "n/a"}`}
        />
      ))}
    </svg>
  );
}

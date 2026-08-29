import type { TwoDMode, ViewerPair } from "../../types";
import { filterPairLayers, layerKey } from "../../sequenceUtils";
import { SvgLayerAnimator } from "./SvgLayerAnimator";
import { SvgMultiLayerAnimator } from "./SvgMultiLayerAnimator";

export type TwoDSceneProps = {
  pair: ViewerPair;
  selectedLayerIds: Array<string | number>;
  selectedBuildingIds: Array<string | number | null>;
  alpha: number;
  showSource: boolean;
  showTarget: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showDeleted: boolean;
  twoDMode: TwoDMode;
};

export function TwoDScene(props: TwoDSceneProps) {
  const hasBuildingSelector = (props.pair.buildings?.length ?? 0) > 1;
  const selectedLayers = filterPairLayers(props.pair, props.selectedLayerIds, props.selectedBuildingIds, false, !hasBuildingSelector);
  if (selectedLayers.length === 0) {
    return <div className="empty-viewer-state">Select at least one layer.</div>;
  }
  if (props.twoDMode === "single_layer" && selectedLayers.length === 1) {
    return (
      <SvgLayerAnimator
        layer={selectedLayers[0]}
        alpha={props.alpha}
        showSource={props.showSource}
        showTarget={props.showTarget}
        showLabels={props.showLabels}
        showTrails={props.showTrails}
        showDeleted={props.showDeleted}
        width={1200}
        height={820}
      />
    );
  }
  if (props.twoDMode === "single_layer" && selectedLayers.length > 1) {
    return (
      <SvgMultiLayerAnimator
        layers={selectedLayers}
        selectedLayerIds={props.selectedLayerIds}
        alpha={props.alpha}
        showSource={props.showSource}
        showTarget={props.showTarget}
        showLabels={props.showLabels}
        showTrails={props.showTrails}
        showDeleted={props.showDeleted}
      />
    );
  }
  if (props.twoDMode === "small_multiples") {
    return (
      <div className="small-multiples-grid">
        {selectedLayers.map((layer, index) => (
          <div className="small-multiple-cell" key={layerKey(layer)}>
            <SvgLayerAnimator
              layer={layer}
              alpha={props.alpha}
              showSource={props.showSource}
              showTarget={props.showTarget}
              showLabels={props.showLabels}
              showTrails={props.showTrails}
              showDeleted={props.showDeleted}
              width={520}
              height={360}
              layerStyleIndex={index}
            />
          </div>
        ))}
      </div>
    );
  }
  return (
    <SvgMultiLayerAnimator
      layers={selectedLayers}
      selectedLayerIds={props.selectedLayerIds}
      alpha={props.alpha}
      showSource={props.showSource}
      showTarget={props.showTarget}
      showLabels={props.showLabels}
      showTrails={props.showTrails}
      showDeleted={props.showDeleted}
    />
  );
}

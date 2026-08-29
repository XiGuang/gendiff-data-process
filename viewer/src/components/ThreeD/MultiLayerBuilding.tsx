import type { OverlayProxyMode, ViewerPair } from "../../types";
import { filterPairLayers, layerKey } from "../../sequenceUtils";
import { BuildingLayerMesh } from "./BuildingLayerMesh";
import { EditPointOverlay3D } from "./EditPointOverlay3D";

export type MultiLayerBuildingProps = {
  pair: ViewerPair;
  selectedLayerIds: Array<string | number>;
  selectedBuildingIds: Array<string | number | null>;
  alpha: number;
  showSource: boolean;
  showTarget: boolean;
  showDeleted: boolean;
  showWireframe: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showAllLayers: boolean;
  overlayProxyMode?: OverlayProxyMode;
};

export function MultiLayerBuilding(props: MultiLayerBuildingProps) {
  const hasBuildingSelector = (props.pair.buildings?.length ?? 0) > 1;
  const layers = filterPairLayers(props.pair, props.selectedLayerIds, props.selectedBuildingIds, props.showAllLayers, !hasBuildingSelector);
  return (
    <group>
      {layers.map((layer, index) => (
        <group key={layerKey(layer)} name={`building-layer-${layerKey(layer)}`}>
          <BuildingLayerMesh
            layer={layer}
            alpha={props.alpha}
            showSource={props.showSource}
            showTarget={props.showTarget}
            showDeleted={props.showDeleted}
            showWireframe={props.showWireframe}
            showLabels={props.showLabels}
            renderMode={props.overlayProxyMode ?? "edit"}
            layerStyleIndex={index}
          />
          {(props.overlayProxyMode ?? "edit") === "edit" ? (
            <EditPointOverlay3D
              layer={layer}
              alpha={props.alpha}
              showLabels={props.showLabels}
              showTrails={props.showTrails}
              showDeleted={props.showDeleted}
              layerStyleIndex={index}
            />
          ) : null}
        </group>
      ))}
    </group>
  );
}

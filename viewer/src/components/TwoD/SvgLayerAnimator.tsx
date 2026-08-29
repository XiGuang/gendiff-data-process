import type { ViewerLayer } from "../../types";
import { computeBounds, SvgLayerOverlay } from "./SvgLayerOverlay";

export type SvgLayerAnimatorProps = {
  layer: ViewerLayer;
  alpha: number;
  showSource: boolean;
  showTarget: boolean;
  showLabels: boolean;
  showTrails: boolean;
  showDeleted: boolean;
  width: number;
  height: number;
  layerStyleIndex?: number;
};

export function SvgLayerAnimator(props: SvgLayerAnimatorProps) {
  const bounds = computeBounds([props.layer]);
  return (
    <svg className="viewer-svg" viewBox={`0 0 ${props.width} ${props.height}`} role="img">
      <SvgLayerOverlay {...props} bounds={bounds} label={`layer ${String(props.layer.layer_id)}`} />
    </svg>
  );
}

import type { ViewerBuildingSummary, ViewerLayer, ViewerPair } from "./types";

export function layerKey(layer: ViewerLayer): string {
  const order = layer.layer_order ?? layer.target_layer_index ?? layer.source_layer_index ?? "order";
  const proxy = layer.proxy_id ?? layer.target_proxy_id ?? layer.source_proxy_id ?? layer.layer_id;
  return [
    order,
    buildingKey(layer.building_id),
    proxy,
    layer.layer_id,
  ].map(String).join("::");
}

export function layerSortKey(layer: ViewerLayer): [number, string] {
  const building = Number(layer.building_id ?? -1);
  const order = layer.layer_order ?? layer.level_index ?? 0;
  return [Number.isFinite(building) ? building * 100000 + order : order, layerKey(layer)];
}

export function sortLayers(layers: ViewerLayer[]): ViewerLayer[] {
  return [...layers].sort((a, b) => {
    const ka = layerSortKey(a);
    const kb = layerSortKey(b);
    return ka[0] - kb[0] || ka[1].localeCompare(kb[1]);
  });
}

export function filterLayers(pair: ViewerPair, selectedLayerIds: Array<string | number>, includeAll = false): ViewerLayer[] {
  if (includeAll) {
    return sortLayers(pair.layers);
  }
  const selected = new Set(selectedLayerIds.map(String));
  return sortLayers(pair.layers.filter((layer) => selected.has(layerKey(layer))));
}

export function buildingKey(value: string | number | null | undefined): string {
  return value === null || value === undefined ? "__unknown__" : String(value);
}

export function getPairBuildings(pair: ViewerPair): ViewerBuildingSummary[] {
  if (pair.buildings && pair.buildings.length > 0) {
    return [...pair.buildings].sort((a, b) => buildingKey(a.building_id).localeCompare(buildingKey(b.building_id), undefined, { numeric: true }));
  }
  const byId = new Map<string, ViewerBuildingSummary>();
  for (const layer of pair.layers) {
    const key = buildingKey(layer.building_id);
    if (!byId.has(key)) {
      byId.set(key, {
        building_id: layer.building_id ?? null,
        building_name: layer.building_name ?? null,
        source_stage_name: undefined,
        target_stage_name: layer.building_stage_name ?? undefined,
      });
    }
  }
  return [...byId.values()].sort((a, b) => buildingKey(a.building_id).localeCompare(buildingKey(b.building_id), undefined, { numeric: true }));
}

export function filterLayersByBuildings(layers: ViewerLayer[], selectedBuildingIds: Array<string | number | null>, includeAll = false): ViewerLayer[] {
  if (includeAll) {
    return sortLayers(layers);
  }
  if (selectedBuildingIds.length === 0) return [];
  const selected = new Set(selectedBuildingIds.map(buildingKey));
  return sortLayers(layers.filter((layer) => selected.has(buildingKey(layer.building_id))));
}

export function filterPairLayers(
  pair: ViewerPair,
  selectedLayerIds: Array<string | number>,
  selectedBuildingIds: Array<string | number | null> = [],
  includeAllLayers = false,
  includeAllBuildings = false,
): ViewerLayer[] {
  const byBuilding = filterLayersByBuildings(pair.layers, selectedBuildingIds, includeAllBuildings);
  if (includeAllLayers) return sortLayers(byBuilding);
  const selected = new Set(selectedLayerIds.map(String));
  return sortLayers(byBuilding.filter((layer) => selected.has(layerKey(layer))));
}

export function countOps(layer: ViewerLayer): Record<string, number> {
  return layer.ops.reduce<Record<string, number>>((acc, op) => {
    acc[op.type] = (acc[op.type] ?? 0) + 1;
    return acc;
  }, {});
}

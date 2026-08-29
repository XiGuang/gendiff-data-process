import type { ViewerBuildingSummary } from "../types";
import { buildingKey } from "../sequenceUtils";

type BuildingSelectorProps = {
  buildings: ViewerBuildingSummary[];
  selectedBuildingIds: Array<string | number | null>;
  onChange: (ids: Array<string | number | null>) => void;
};

export function BuildingSelector({ buildings, selectedBuildingIds, onChange }: BuildingSelectorProps) {
  const selected = new Set(selectedBuildingIds.map(buildingKey));

  function toggle(building: ViewerBuildingSummary) {
    const key = buildingKey(building.building_id);
    const next = new Set(selected);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    onChange(buildings.filter((candidate) => next.has(buildingKey(candidate.building_id))).map((candidate) => candidate.building_id));
  }

  return (
    <section className="panel-section">
      <div className="section-title">Buildings</div>
      <div className="button-row">
        <button type="button" onClick={() => onChange(buildings.map((building) => building.building_id))}>Select all</button>
        <button type="button" onClick={() => onChange([])}>Clear</button>
      </div>
      <div className="layer-list">
        {buildings.map((building) => {
          const label = building.building_name ?? `building ${buildingKey(building.building_id)}`;
          const source = building.source_stage_name ?? building.source_stage_position ?? "n/a";
          const target = building.target_stage_name ?? building.target_stage_position ?? "n/a";
          return (
            <label key={buildingKey(building.building_id)} className="layer-option">
              <input type="checkbox" checked={selected.has(buildingKey(building.building_id))} onChange={() => toggle(building)} />
              <span>
                <strong>{label}</strong>
                <small>id {buildingKey(building.building_id)} | {String(source)} to {String(target)}</small>
              </span>
            </label>
          );
        })}
      </div>
    </section>
  );
}

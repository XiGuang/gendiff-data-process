import type { MainViewTab } from "../types";

type ViewModeToggleProps = {
  viewMode: MainViewTab;
  onChange: (mode: MainViewTab) => void;
};

const TABS: Array<{ id: MainViewTab; label: string }> = [
  { id: "proxy3d", label: "3D" },
  { id: "point_cloud", label: "Point Cloud" },
  { id: "overlay", label: "Overlay" },
  { id: "playback", label: "Edit Playback" },
];

export function ViewModeToggle({ viewMode, onChange }: ViewModeToggleProps) {
  return (
    <div className="segmented">
      {TABS.map((tab) => (
        <button key={tab.id} className={viewMode === tab.id ? "active" : ""} onClick={() => onChange(tab.id)} type="button">
          {tab.label}
        </button>
      ))}
    </div>
  );
}

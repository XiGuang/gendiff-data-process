import { useEffect, useState } from "react";
import type { ServerFsEntry, ServerFsListing } from "../serverApi";
import { listServerPath } from "../serverApi";

type ServerPathPickerProps = {
  title: string;
  mode: "directory" | "file";
  initialPath: string;
  onCancel: () => void;
  onSelect: (path: string) => void;
};

export function ServerPathPicker({ title, mode, initialPath, onCancel, onSelect }: ServerPathPickerProps) {
  const [path, setPath] = useState(initialPath);
  const [listing, setListing] = useState<ServerFsListing | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    listServerPath(path)
      .then((next) => {
        if (!cancelled) {
          setListing(next);
          setPath(next.path);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  function openEntry(entry: ServerFsEntry) {
    if (entry.type === "directory") {
      setPath(entry.path);
    } else if (mode === "file") {
      onSelect(entry.path);
    }
  }

  return (
    <div className="path-picker-backdrop">
      <div className="path-picker">
        <div className="path-picker-header">
          <strong>{title}</strong>
          <button type="button" onClick={onCancel}>Close</button>
        </div>
        <label className="control-block">
          <span>Server path</span>
          <input className="text-input" value={path} onChange={(event) => setPath(event.target.value)} />
        </label>
        <div className="button-row">
          <button type="button" onClick={() => listing && setPath(listing.parent)}>Up</button>
          {mode === "directory" ? <button type="button" onClick={() => onSelect(listing?.path ?? path)}>Use this directory</button> : null}
        </div>
        {error ? <div className="path-picker-error">{error}</div> : null}
        <div className="path-entry-list">
          {listing?.entries.map((entry) => (
            <button
              type="button"
              key={entry.path}
              className={`path-entry ${entry.type}`}
              onClick={() => openEntry(entry)}
              title={entry.path}
            >
              <span>{entry.type === "directory" ? "dir" : "yaml"}</span>
              <strong>{entry.name}</strong>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

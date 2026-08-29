import { useEffect, useMemo, useRef, useState } from "react";
import type { ConditionPointCloud, DatasetSummary, MainViewTab, OverlayProxyMode, TwoDMode, ViewerData } from "../types";
import { filterLayersByBuildings, getPairBuildings, layerKey, sortLayers } from "../sequenceUtils";
import { loadDatasetCondition } from "../serverApi";
import { ViewModeToggle } from "./ViewModeToggle";
import { PairSelector } from "./PairSelector";
import { LayerSelector } from "./LayerSelector";
import { TimelineControls } from "./TimelineControls";
import { DebugPanel } from "./DebugPanel";
import { TwoDScene } from "./TwoD/TwoDScene";
import { ThreeScene } from "./ThreeD/ThreeScene";
import { ServerDataControls } from "./ServerDataControls";
import { BuildingSelector } from "./BuildingSelector";

type SequenceViewerProps = {
  data: ViewerData | null;
  onDataLoaded: (data: ViewerData) => void;
};

const EMPTY_PAIRS: ViewerData["pairs"] = [];

export function SequenceViewer({ data, onDataLoaded }: SequenceViewerProps) {
  const [viewMode, setViewMode] = useState<MainViewTab>("proxy3d");
  const [twoDMode, setTwoDMode] = useState<TwoDMode>("single_layer");
  const [selectedPairIndex, setSelectedPairIndex] = useState(0);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [selectedBuildingIds, setSelectedBuildingIds] = useState<Array<string | number | null>>([]);
  const [selectedLayerIds, setSelectedLayerIds] = useState<Array<string | number>>([]);
  const [pointQuality, setPointQuality] = useState(8192);
  const [pointColorMode, setPointColorMode] = useState<"uniform" | "height">("height");
  const [overlayProxyMode, setOverlayProxyMode] = useState<OverlayProxyMode>("target");
  const [pointCloud, setPointCloud] = useState<ConditionPointCloud | null>(null);
  const [pointCloudStatus, setPointCloudStatus] = useState<string | null>(null);
  const [alpha, setAlpha] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showSource, setShowSource] = useState(true);
  const [showTarget, setShowTarget] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [showTrails, setShowTrails] = useState(true);
  const [showDeleted, setShowDeleted] = useState(true);
  const [showWireframe, setShowWireframe] = useState(false);
  const [showAllLayers3D, setShowAllLayers3D] = useState(false);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const pointCloudCacheRef = useRef<Map<string, ConditionPointCloud>>(new Map());

  const pairs = data?.pairs ?? EMPTY_PAIRS;
  const safePairIndex = Math.min(selectedPairIndex, Math.max(0, pairs.length - 1));
  const pair = pairs[safePairIndex];
  const datasetDir = data?.dataset_dir ?? datasetSummary?.datasetDir ?? "";
  const buildings = useMemo(() => (pair ? getPairBuildings(pair) : []), [pair]);
  const hasBuildingSelector = data?.dataset_kind === "area" || datasetSummary?.datasetKind === "area" || buildings.length > 1;
  const buildingFilteredLayers = useMemo(
    () => filterLayersByBuildings(pair?.layers ?? [], selectedBuildingIds, !hasBuildingSelector),
    [hasBuildingSelector, pair, selectedBuildingIds],
  );
  const sortedLayers = useMemo(() => sortLayers(buildingFilteredLayers), [buildingFilteredLayers]);

  useEffect(() => {
    if (selectedPairIndex !== safePairIndex) {
      setSelectedPairIndex(safePairIndex);
    }
  }, [safePairIndex, selectedPairIndex]);

  useEffect(() => {
    const nextPair = pairs[safePairIndex];
    const nextBuildings = nextPair ? getPairBuildings(nextPair) : [];
    const nextLayers = sortLayers(nextPair?.layers ?? []);
    const isArea = data?.dataset_kind === "area" || datasetSummary?.datasetKind === "area" || nextBuildings.length > 1;
    setSelectedBuildingIds(isArea ? nextBuildings.map((building) => building.building_id) : []);
    if (isArea) {
      setSelectedLayerIds(nextLayers.map(layerKey));
    } else {
      const first = nextLayers[0];
      setSelectedLayerIds(first ? [layerKey(first)] : []);
    }
    setAlpha(0);
    setIsPlaying(false);
    setPointCloud(null);
    setPointCloudStatus(null);
  }, [data?.dataset_kind, datasetSummary?.datasetKind, pairs, safePairIndex]);

  useEffect(() => {
    if (!pair || !datasetDir || (viewMode !== "point_cloud" && viewMode !== "overlay")) return;
    const key = `${datasetDir}:${pair.dataset_locator ?? pair.pair_id}:${pointQuality}`;
    const cached = pointCloudCacheRef.current.get(key);
    if (cached) {
      setPointCloud(cached);
      setPointCloudStatus(cached.available ? `Showing ${cached.sampledPoints} / ${cached.totalPoints} points` : "Condition point cloud is missing");
      return;
    }
    let cancelled = false;
    setPointCloud(null);
    setPointCloudStatus("Loading condition point cloud...");
    loadDatasetCondition(datasetDir, pair.pair_id, pointQuality, pair.dataset_locator)
      .then((next) => {
        if (cancelled) return;
        pointCloudCacheRef.current.set(key, next);
        while (pointCloudCacheRef.current.size > 5) {
          const oldest = pointCloudCacheRef.current.keys().next().value;
          if (oldest) pointCloudCacheRef.current.delete(oldest);
          else break;
        }
        setPointCloud(next);
        setPointCloudStatus(next.available ? `Showing ${next.sampledPoints} / ${next.totalPoints} points` : "Condition point cloud is missing");
      })
      .catch((error) => {
        if (!cancelled) setPointCloudStatus(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetDir, pair, pointQuality, viewMode]);

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTimeRef.current = null;
      return;
    }
    const step = (now: number) => {
      const last = lastTimeRef.current ?? now;
      lastTimeRef.current = now;
      const delta = ((now - last) / 1400) * playbackSpeed;
      setAlpha((current) => {
        const next = Math.min(1, current + delta);
        if (next >= 1) {
          setIsPlaying(false);
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying, playbackSpeed]);

  function setPairIndex(index: number) {
    setSelectedPairIndex(Math.max(0, Math.min(pairs.length - 1, index)));
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <h1>Area Edit V2 Browser</h1>
          <p>{data?.sequence_id ?? datasetSummary?.datasetName ?? "Select a dataset directory"}</p>
        </div>
        <ViewModeToggle viewMode={viewMode} onChange={setViewMode} />
        {pairs.length > 1 ? <PairSelector pairs={pairs} selectedPairIndex={selectedPairIndex} onChange={setPairIndex} /> : null}
        <ServerDataControls data={data} currentPairId={pair?.pair_id} onDataLoaded={onDataLoaded} onDatasetSummaryLoaded={setDatasetSummary} />
        {pair ? (
          <TimelineControls
            alpha={alpha}
            playbackSpeed={playbackSpeed}
            isPlaying={isPlaying}
            canPrev={selectedPairIndex > 0}
            canNext={selectedPairIndex < pairs.length - 1}
            onAlphaChange={(next) => {
              setAlpha(next);
              setIsPlaying(false);
            }}
            onPlaybackSpeedChange={setPlaybackSpeed}
            onPlayPause={() => {
              if (alpha >= 1) setAlpha(0);
              setIsPlaying((value) => !value);
            }}
            onPrevPair={() => setPairIndex(selectedPairIndex - 1)}
            onNextPair={() => setPairIndex(selectedPairIndex + 1)}
          />
        ) : null}
        {hasBuildingSelector ? (
          <BuildingSelector buildings={buildings} selectedBuildingIds={selectedBuildingIds} onChange={setSelectedBuildingIds} />
        ) : null}
        {pair ? <LayerSelector layers={sortedLayers} selectedLayerIds={selectedLayerIds} onChange={setSelectedLayerIds} /> : null}
        <section className="panel-section">
          <div className="section-title">Display</div>
          {viewMode === "playback" ? (
            <label className="control-block">
              <span>2D layout</span>
              <select value={twoDMode} onChange={(event) => setTwoDMode(event.target.value as TwoDMode)}>
                <option value="single_layer">single layer</option>
                <option value="multi_layer_overlay">multi-layer overlay</option>
                <option value="small_multiples">small multiples</option>
              </select>
            </label>
          ) : (
            <label className="check-row"><input type="checkbox" checked={showAllLayers3D} onChange={(event) => setShowAllLayers3D(event.target.checked)} />Ignore layer selection</label>
          )}
          {viewMode === "point_cloud" || viewMode === "overlay" ? (
            <>
              {viewMode === "overlay" ? (
                <label className="control-block">
                  <span>Overlay proxy</span>
                  <select value={overlayProxyMode} onChange={(event) => setOverlayProxyMode(event.target.value as OverlayProxyMode)}>
                    <option value="source">previous state</option>
                    <option value="target">next state</option>
                    <option value="edit">edit playback</option>
                  </select>
                </label>
              ) : null}
              <label className="control-block">
                <span>Point quality</span>
                <select value={pointQuality} onChange={(event) => setPointQuality(Number(event.target.value))}>
                  <option value={2048}>2k</option>
                  <option value={8192}>8k</option>
                  <option value={32768}>32k</option>
                  <option value={0}>all</option>
                </select>
              </label>
              <label className="control-block">
                <span>Point color</span>
                <select value={pointColorMode} onChange={(event) => setPointColorMode(event.target.value as "uniform" | "height")}>
                  <option value="height">height</option>
                  <option value="uniform">uniform</option>
                </select>
              </label>
              {pointCloudStatus ? <div className="server-data-status">{pointCloudStatus}</div> : null}
            </>
          ) : null}
          <label className="check-row"><input type="checkbox" checked={showSource} onChange={(event) => setShowSource(event.target.checked)} />Source</label>
          <label className="check-row"><input type="checkbox" checked={showTarget} onChange={(event) => setShowTarget(event.target.checked)} />Target</label>
          <label className="check-row"><input type="checkbox" checked={showTrails} onChange={(event) => setShowTrails(event.target.checked)} />Trails</label>
          <label className="check-row"><input type="checkbox" checked={showDeleted} onChange={(event) => setShowDeleted(event.target.checked)} />Deleted ghost</label>
          <label className="check-row"><input type="checkbox" checked={showLabels} onChange={(event) => setShowLabels(event.target.checked)} />Labels</label>
          {viewMode !== "playback" ? <label className="check-row"><input type="checkbox" checked={showWireframe} onChange={(event) => setShowWireframe(event.target.checked)} />Wireframe</label> : null}
        </section>
        {pair ? <DebugPanel pair={pair} selectedLayerIds={selectedLayerIds} selectedBuildingIds={selectedBuildingIds} /> : null}
      </aside>
      <main className="viewer-area">
        {!pair ? (
          <div className="empty-viewer-state">Load a dataset summary, search a split, then load one pair.</div>
        ) : viewMode === "playback" ? (
          <TwoDScene
            pair={pair}
            selectedLayerIds={selectedLayerIds}
            selectedBuildingIds={selectedBuildingIds}
            alpha={alpha}
            showSource={showSource}
            showTarget={showTarget}
            showLabels={showLabels}
            showTrails={showTrails}
            showDeleted={showDeleted}
            twoDMode={twoDMode}
          />
        ) : (
          <ThreeScene
            pair={pair}
            selectedLayerIds={selectedLayerIds}
            selectedBuildingIds={selectedBuildingIds}
            alpha={alpha}
            showSource={showSource}
            showTarget={showTarget}
            showLabels={showLabels}
            showTrails={showTrails}
            showDeleted={showDeleted}
            showWireframe={showWireframe}
            showAllLayers={showAllLayers3D}
            pointCloud={pointCloud}
            pointCloudColorMode={pointColorMode}
            overlayProxyMode={viewMode === "overlay" ? overlayProxyMode : "edit"}
            showProxy={viewMode !== "point_cloud"}
            showPointCloud={viewMode === "point_cloud" || viewMode === "overlay"}
          />
        )}
      </main>
    </div>
  );
}

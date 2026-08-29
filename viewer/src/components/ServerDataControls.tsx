import { useEffect, useState } from "react";
import type { DatasetSummary, PairListPage, ViewerData } from "../types";
import { exportViewerDataFromServer, loadDatasetPair, loadDatasetPairs, loadDatasetSummary } from "../serverApi";
import { ServerPathPicker } from "./ServerPathPicker";

type ServerDataControlsProps = {
  data: ViewerData | null;
  currentPairId?: string;
  onDataLoaded: (data: ViewerData) => void;
  onDatasetSummaryLoaded?: (summary: DatasetSummary | null) => void;
};

export function ServerDataControls({ data, currentPairId, onDataLoaded, onDatasetSummaryLoaded }: ServerDataControlsProps) {
  const [datasetDir, setDatasetDir] = useState(data?.dataset_dir ?? "");
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [split, setSplit] = useState("train");
  const [pairQuery, setPairQuery] = useState("");
  const [pairPage, setPairPage] = useState<PairListPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [selectedPairId, setSelectedPairId] = useState("");
  const [overrideYaml, setOverrideYaml] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [picker, setPicker] = useState<"dataset" | "yaml" | null>(null);
  const limit = 100;

  useEffect(() => {
    setDatasetDir(data?.dataset_dir ?? datasetDir);
    setOverrideYaml("");
    setStatus(null);
  }, [data?.dataset_dir, data?.sequence_id]);

  async function loadSummary(nextDatasetDir = datasetDir) {
    setStatus("Loading summary...");
    try {
      const next = await loadDatasetSummary(nextDatasetDir);
      setSummary(next);
      onDatasetSummaryLoaded?.(next);
      setPairPage(null);
      setSelectedPairId("");
      setOffset(0);
      setStatus(`Summary loaded: ${next.pairTotal} pairs`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function loadPairList(nextOffset = offset) {
    setStatus("Loading pair page...");
    try {
      const page = await loadDatasetPairs({ datasetDir, split, query: pairQuery, offset: nextOffset, limit });
      setPairPage(page);
      setOffset(page.offset);
      setSelectedPairId((current) => (current && page.pairs.some((pair) => pair.pairId === current) ? current : page.pairs[0]?.pairId ?? ""));
      setStatus(
        page.searchTruncated
          ? `Search stopped after ${page.searchScanned ?? 0} pairs; narrow the query`
          : `Loaded ${page.pairs.length} of ${page.total} matching pairs`,
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function loadSelectedPair(pairId = selectedPairId) {
    if (!pairId) return;
    setStatus(`Loading ${pairId}...`);
    try {
      const pairLocator = pairPage?.pairs.find((pair) => pair.pairId === pairId)?.pairLocator;
      const next = await loadDatasetPair(datasetDir, pairId, pairLocator);
      onDataLoaded(next);
      setStatus(`Loaded ${pairId}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function applyOverride() {
    const targetPairId = selectedPairId || currentPairId;
    if (!targetPairId) return;
    setStatus("Applying YAML override...");
    try {
      const next = await exportViewerDataFromServer({
        datasetDir,
        pairId: targetPairId,
        overridePairId: targetPairId,
        overrideEditSequencePath: overrideYaml,
      });
      onDataLoaded(next);
      setStatus(`Applied override to ${targetPairId}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  const total = pairPage?.total ?? 0;
  const canPrev = offset > 0;
  const canNext = pairPage ? offset + limit < pairPage.total : false;
  const isPacked = summary?.datasetFormat === "packed" || data?.dataset_format === "packed";

  return (
    <section className="panel-section">
      <div className="section-title">Dataset</div>
      <label className="control-block">
        <span>Dataset directory</span>
        <input className="text-input" value={datasetDir} onChange={(event) => setDatasetDir(event.target.value)} />
      </label>
      <div className="button-row">
        <button type="button" onClick={() => setPicker("dataset")}>Browse</button>
        <button type="button" onClick={() => loadSummary()} disabled={!datasetDir}>Load summary</button>
      </div>
      {summary ? (
        <div className="summary-grid">
          <span>kind</span><strong>{summary.datasetKind ?? "n/a"}</strong>
          <span>format</span><strong>{summary.datasetFormat ?? "raw"}</strong>
          <span>pairs</span><strong>{summary.pairTotal}</strong>
          <span>states</span><strong>{summary.stateTotal ?? summary.stageTotal ?? "n/a"}</strong>
          <span>conditions</span><strong>{summary.hasConditions ? "yes" : "no"}</strong>
        </div>
      ) : null}
      <div className="section-title">Pairs</div>
      <label className="control-block">
        <span>Split</span>
        <select value={split} onChange={(event) => { setSplit(event.target.value); setOffset(0); }}>
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
          <option value="all">all</option>
        </select>
      </label>
      <label className="control-block">
        <span>Pair search</span>
        <input className="text-input" value={pairQuery} onChange={(event) => setPairQuery(event.target.value)} placeholder="area_state_000000" />
      </label>
      <div className="button-row">
        <button type="button" onClick={() => loadPairList(0)} disabled={!datasetDir}>Search</button>
        <button type="button" onClick={() => currentPairId && setSelectedPairId(currentPairId)} disabled={!currentPairId}>Use current</button>
      </div>
      {pairPage ? (
        <div className="pair-browser">
          <div className="pair-browser-toolbar">
            <button type="button" onClick={() => loadPairList(Math.max(0, offset - limit))} disabled={!canPrev}>Prev</button>
            <span>{total === 0 ? "0" : `${offset + 1}-${Math.min(offset + limit, total)}`} / {total}</span>
            <button type="button" onClick={() => loadPairList(offset + limit)} disabled={!canNext}>Next</button>
          </div>
          <div className="pair-list">
            {pairPage.pairs.map((pair) => (
              <button
                key={pair.pairLocator ?? pair.pairId}
                type="button"
                className={selectedPairId === pair.pairId ? "pair-row active" : "pair-row"}
                onClick={() => setSelectedPairId(pair.pairId)}
              >
                <strong>{pair.pairId}</strong>
                <span>{pair.sourceState ?? "?"}{" -> "}{pair.targetState ?? "?"}</span>
                <small>{pair.changeKind ?? (pair.isDemolitionPair ? "demolition" : "edit")} | {pair.validationOk === false ? "invalid" : pair.validationOk ? "valid" : "unchecked"} | pc {pair.conditionPointCount ?? "n/a"}{pair.pairHash ? ` | ${pair.pairHash.slice(0, 12)}` : ""}</small>
              </button>
            ))}
          </div>
          <button className="load-pair-button" type="button" onClick={() => loadSelectedPair()} disabled={!selectedPairId}>Load selected pair</button>
        </div>
      ) : null}
      {!isPacked ? (
        <>
          <label className="control-block">
            <span>Override YAML for current pair</span>
            <input className="text-input" value={overrideYaml} onChange={(event) => setOverrideYaml(event.target.value)} placeholder={currentPairId ?? ""} />
          </label>
          <div className="button-row">
            <button type="button" onClick={() => setPicker("yaml")}>Browse YAML</button>
            <button type="button" onClick={applyOverride} disabled={!datasetDir || !overrideYaml}>Apply to current pair</button>
          </div>
        </>
      ) : null}
      {status ? <div className="server-data-status">{status}</div> : null}
      {picker === "dataset" ? (
        <ServerPathPicker
          title="Select dataset directory"
          mode="directory"
          initialPath={datasetDir}
          onCancel={() => setPicker(null)}
          onSelect={(selected) => {
            setDatasetDir(selected);
            onDatasetSummaryLoaded?.(null);
            setSummary(null);
            setPairPage(null);
            setPicker(null);
          }}
        />
      ) : null}
      {picker === "yaml" ? (
        <ServerPathPicker
          title="Select edit sequence YAML"
          mode="file"
          initialPath={overrideYaml || `${datasetDir}/edit_sequences_v2`}
          onCancel={() => setPicker(null)}
          onSelect={(selected) => {
            setOverrideYaml(selected);
            setPicker(null);
          }}
        />
      ) : null}
    </section>
  );
}

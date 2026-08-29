import type { ConditionPointCloud, DatasetSummary, PairListPage, ViewerData } from "./types";
import { validateViewerData } from "./loadViewerData";

export type ServerFsEntry = {
  name: string;
  path: string;
  type: "directory" | "file";
};

export type ServerFsListing = {
  path: string;
  parent: string;
  entries: ServerFsEntry[];
};

export type ServerPairListing = {
  datasetDir: string;
  total: number;
  matched: number;
  limit: number;
  pairs: string[];
};

async function readApiJson(response: Response): Promise<unknown> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof payload === "object" && payload && "error" in payload ? String((payload as { error: unknown }).error) : response.statusText;
    throw new Error(message);
  }
  return payload;
}

export async function listServerPath(path: string): Promise<ServerFsListing> {
  const response = await fetch(`/api/server-fs/list?path=${encodeURIComponent(path)}`);
  return (await readApiJson(response)) as ServerFsListing;
}

export async function listViewerPairs(datasetDir: string, query = "", limit = 500): Promise<ServerPairListing> {
  const params = new URLSearchParams({ datasetDir, query, limit: String(limit) });
  const response = await fetch(`/api/viewer/pairs?${params.toString()}`);
  return (await readApiJson(response)) as ServerPairListing;
}

export async function exportViewerDataFromServer(options: {
  datasetDir: string;
  pairId?: string;
  pairLocator?: string;
  overridePairId?: string;
  overrideEditSequencePath?: string;
}): Promise<ViewerData> {
  const response = await fetch("/api/viewer/export", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(options),
  });
  return validateViewerData(await readApiJson(response));
}

export async function loadDatasetSummary(datasetDir: string): Promise<DatasetSummary> {
  const params = new URLSearchParams({ datasetDir });
  const response = await fetch(`/api/datasets/summary?${params.toString()}`);
  return (await readApiJson(response)) as DatasetSummary;
}

export async function loadDatasetPairs(options: {
  datasetDir: string;
  split: string;
  query?: string;
  offset?: number;
  limit?: number;
}): Promise<PairListPage> {
  const params = new URLSearchParams({
    datasetDir: options.datasetDir,
    split: options.split,
    query: options.query ?? "",
    offset: String(options.offset ?? 0),
    limit: String(options.limit ?? 100),
  });
  const response = await fetch(`/api/datasets/pairs?${params.toString()}`);
  return (await readApiJson(response)) as PairListPage;
}

export async function loadDatasetPair(datasetDir: string, pairId: string, pairLocator?: string | null): Promise<ViewerData> {
  const params = new URLSearchParams({ datasetDir, pairId });
  if (pairLocator) params.set("pairLocator", pairLocator);
  const response = await fetch(`/api/datasets/pair?${params.toString()}`);
  return validateViewerData(await readApiJson(response));
}

export async function loadDatasetCondition(datasetDir: string, pairId: string, maxPoints: number, pairLocator?: string | null): Promise<ConditionPointCloud> {
  const params = new URLSearchParams({ datasetDir, pairId, maxPoints: String(maxPoints) });
  if (pairLocator) params.set("pairLocator", pairLocator);
  const response = await fetch(`/api/datasets/condition?${params.toString()}`);
  return (await readApiJson(response)) as ConditionPointCloud;
}

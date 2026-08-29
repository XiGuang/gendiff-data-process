import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import { fileURLToPath } from "node:url";

type ServerFsEntry = {
  name: string;
  path: string;
  type: "directory" | "file";
};

const configDir = path.dirname(fileURLToPath(import.meta.url));
const LARGE_PAIR_COUNT_LIMIT = 1000;
const PAIR_LIST_LIMIT = 500;
const browserApiCache = new Map<string, unknown>();

function sendJson(res: ServerResponse, status: number, payload: unknown) {
  res.statusCode = status;
  res.setHeader("content-type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) return {};
  const text = Buffer.concat(chunks).toString("utf-8");
  return JSON.parse(text) as Record<string, unknown>;
}

function repoRoot(): string {
  return path.resolve(configDir, "..");
}

function resolveServerPath(rawPath: string): string {
  return path.isAbsolute(rawPath) ? path.resolve(rawPath) : path.resolve(repoRoot(), rawPath);
}

async function listServerPath(rawPath: string | undefined) {
  const requested = rawPath && rawPath.trim() ? rawPath : repoRoot();
  const resolved = resolveServerPath(requested);
  const stat = await fs.stat(resolved);
  const dirPath = stat.isDirectory() ? resolved : path.dirname(resolved);
  const dirents = await fs.readdir(dirPath, { withFileTypes: true });
  const entries: ServerFsEntry[] = dirents
    .filter((entry) => entry.isDirectory() || entry.isFile())
    .filter((entry) => entry.isDirectory() || /\.ya?ml$/i.test(entry.name) || /\.json$/i.test(entry.name))
    .map((entry) => ({
      name: entry.name,
      path: path.join(dirPath, entry.name),
      type: entry.isDirectory() ? "directory" : "file",
    }))
    .sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === "directory" ? -1 : 1));
  return {
    path: dirPath,
    parent: path.dirname(dirPath),
    entries,
  };
}

async function listViewerPairs(datasetDir: string | undefined, query: string | undefined, limitText: string | undefined) {
  const resolvedDatasetDir = datasetDir && datasetDir.trim() ? resolveServerPath(datasetDir) : "";
  if (!resolvedDatasetDir) {
    throw new Error("datasetDir is required");
  }
  const editSeqDir = path.join(resolvedDatasetDir, "edit_sequences_v2");
  const dirents = await fs.readdir(editSeqDir, { withFileTypes: true });
  const allPairs = dirents
    .filter((entry) => entry.isFile() && /\.ya?ml$/i.test(entry.name))
    .map((entry) => entry.name.replace(/\.ya?ml$/i, ""))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const normalizedQuery = (query ?? "").trim().toLowerCase();
  const filtered = normalizedQuery ? allPairs.filter((pair) => pair.toLowerCase().includes(normalizedQuery)) : allPairs;
  const parsedLimit = Number(limitText);
  const limit = Number.isFinite(parsedLimit) && parsedLimit > 0 ? Math.min(parsedLimit, 5000) : PAIR_LIST_LIMIT;
  return {
    datasetDir: resolvedDatasetDir,
    total: allPairs.length,
    matched: filtered.length,
    limit,
    pairs: filtered.slice(0, limit),
  };
}

async function exportViewerData(body: Record<string, unknown>) {
  const datasetDir = typeof body.datasetDir === "string" ? body.datasetDir.trim() : "";
  const pairId = typeof body.pairId === "string" ? body.pairId.trim() : "";
  const pairLocator = typeof body.pairLocator === "string" ? body.pairLocator.trim() : "";
  const overridePairId = typeof body.overridePairId === "string" ? body.overridePairId.trim() : "";
  const overrideEditSequencePath = typeof body.overrideEditSequencePath === "string" ? body.overrideEditSequencePath.trim() : "";
  if (!datasetDir) {
    throw new Error("datasetDir is required");
  }
  if (!pairId) {
    const editSeqDir = path.join(resolveServerPath(datasetDir), "edit_sequences_v2");
    const entries = await fs.readdir(editSeqDir).catch(() => []);
    const yamlCount = entries.filter((entry) => /\.ya?ml$/i.test(entry)).length;
    if (yamlCount > LARGE_PAIR_COUNT_LIMIT) {
      throw new Error(`This dataset has ${yamlCount} pair YAML files. Fill Pair id and load one pair at a time.`);
    }
  }
  const outputPath = path.join(os.tmpdir(), `edit-animation-viewer-${Date.now()}-${Math.random().toString(16).slice(2)}.json`);
  const scriptPath = path.resolve(configDir, "../tools/export_edit_animation_viewer_data.py");
  const args = [scriptPath, "--dataset-dir", resolveServerPath(datasetDir), "--output", outputPath];
  if (pairId) {
    args.push("--pair-id", pairId);
  }
  if (pairLocator) {
    args.push("--pair-locator", pairLocator);
  }
  if (overridePairId || overrideEditSequencePath) {
    if (!overridePairId || !overrideEditSequencePath) {
      throw new Error("overridePairId and overrideEditSequencePath must be provided together");
    }
    args.push("--override-pair-id", overridePairId, "--override-edit-sequence-yaml", overrideEditSequencePath);
  }
  await new Promise<void>((resolve, reject) => {
    const child = spawn(process.env.PYTHON ?? "python", args, { cwd: repoRoot() });
    let stderr = "";
    let stdout = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error((stderr || stdout || `exporter exited with code ${code}`).trim()));
    });
  });
  try {
    const text = await fs.readFile(outputPath, "utf-8");
    return JSON.parse(text);
  } finally {
    await fs.rm(outputPath, { force: true });
  }
}

async function browserCacheFingerprint(datasetDir: string, split?: string | null) {
  const parts: string[] = [];
  for (const candidate of [
    datasetDir,
    path.join(datasetDir, "dataset_meta.yaml"),
    path.join(datasetDir, "dataset_meta.pt"),
    path.join(datasetDir, "states.pt"),
    path.join(datasetDir, "edit_sequences_v2"),
    split && split !== "all" ? path.join(datasetDir, `${split}.yaml`) : "",
    split && split !== "all" ? path.join(datasetDir, `${split}_index.pt`) : "",
    split && split !== "all" ? path.join(datasetDir, `${split}_index.yaml`) : "",
  ]) {
    if (!candidate) continue;
    const stat = await fs.stat(candidate).catch(() => null);
    if (stat) {
      parts.push(`${candidate}:${stat.mtimeMs}:${stat.size}`);
    }
  }
  return parts.join("|");
}

async function runBrowserApi(args: string[], cacheKey?: string): Promise<unknown> {
  if (cacheKey && browserApiCache.has(cacheKey)) {
    return browserApiCache.get(cacheKey);
  }
  const scriptPath = path.resolve(configDir, "../tools/dataset_browser_api.py");
  const payload = await new Promise<unknown>((resolve, reject) => {
    const child = spawn(process.env.PYTHON ?? "python", [scriptPath, ...args], { cwd: repoRoot() });
    let stderr = "";
    let stdout = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error((stderr || stdout || `dataset browser helper exited with code ${code}`).trim()));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`dataset browser helper returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`));
      }
    });
  });
  if (cacheKey) {
    browserApiCache.set(cacheKey, payload);
    if (browserApiCache.size > 32) {
      const oldest = browserApiCache.keys().next().value;
      if (oldest) browserApiCache.delete(oldest);
    }
  }
  return payload;
}

async function datasetSummary(datasetDirText: string | undefined) {
  const datasetDir = datasetDirText && datasetDirText.trim() ? resolveServerPath(datasetDirText) : "";
  if (!datasetDir) throw new Error("datasetDir is required");
  const fingerprint = await browserCacheFingerprint(datasetDir);
  return runBrowserApi(["summary", "--dataset-dir", datasetDir], `summary:${fingerprint}`);
}

async function datasetPairs(params: URLSearchParams) {
  const datasetDirText = params.get("datasetDir") ?? "";
  const datasetDir = datasetDirText.trim() ? resolveServerPath(datasetDirText) : "";
  if (!datasetDir) throw new Error("datasetDir is required");
  const split = params.get("split") || "all";
  const query = params.get("query") || "";
  const offset = params.get("offset") || "0";
  const limit = params.get("limit") || "100";
  const fingerprint = await browserCacheFingerprint(datasetDir, split);
  return runBrowserApi(
    ["pairs", "--dataset-dir", datasetDir, "--split", split, "--query", query, "--offset", offset, "--limit", limit],
    `pairs:${fingerprint}:${split}:${query}:${offset}:${limit}`,
  );
}

async function datasetPair(params: URLSearchParams) {
  const datasetDir = params.get("datasetDir") ?? "";
  const pairId = params.get("pairId") ?? "";
  const pairLocator = params.get("pairLocator") ?? "";
  if (!datasetDir.trim()) throw new Error("datasetDir is required");
  if (!pairId.trim()) throw new Error("pairId is required");
  return exportViewerData({ datasetDir, pairId, pairLocator });
}

async function datasetCondition(params: URLSearchParams) {
  const datasetDirText = params.get("datasetDir") ?? "";
  const datasetDir = datasetDirText.trim() ? resolveServerPath(datasetDirText) : "";
  const pairId = params.get("pairId") ?? "";
  const pairLocator = params.get("pairLocator") ?? "";
  const maxPoints = params.get("maxPoints") || "8192";
  if (!datasetDir) throw new Error("datasetDir is required");
  if (!pairId.trim()) throw new Error("pairId is required");
  const args = ["condition", "--dataset-dir", datasetDir, "--pair-id", pairId, "--max-points", maxPoints];
  if (pairLocator) args.push("--pair-locator", pairLocator);
  return runBrowserApi(args);
}

function installServerApi(server: { middlewares: { use: (fn: (req: IncomingMessage, res: ServerResponse, next: () => void) => void) => void } }) {
  server.middlewares.use((req, res, next) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (req.method === "GET" && url.pathname === "/api/server-fs/list") {
      listServerPath(url.searchParams.get("path") ?? undefined)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/viewer/pairs") {
      listViewerPairs(url.searchParams.get("datasetDir") ?? undefined, url.searchParams.get("query") ?? undefined, url.searchParams.get("limit") ?? undefined)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "POST" && url.pathname === "/api/viewer/export") {
      readJsonBody(req)
        .then(exportViewerData)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/datasets/summary") {
      datasetSummary(url.searchParams.get("datasetDir") ?? undefined)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/datasets/pairs") {
      datasetPairs(url.searchParams)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/datasets/pair") {
      datasetPair(url.searchParams)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/datasets/condition") {
      datasetCondition(url.searchParams)
        .then((payload) => sendJson(res, 200, payload))
        .catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }));
      return;
    }
    next();
  });
}

export default defineConfig({
  server: {
    watch: {
      usePolling: true,
      interval: 500,
    },
  },
  plugins: [
    react(),
    {
      name: "edit-animation-viewer-server-api",
      configureServer(server) {
        installServerApi(server);
      },
      configurePreviewServer(server) {
        installServerApi(server);
      },
    },
  ],
  test: {
    environment: "jsdom",
    globals: true,
  },
});

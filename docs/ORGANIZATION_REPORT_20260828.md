# Organization execution report — 2026-08-28

## Outcome

The legacy data repository has been stabilized without moving or deleting any
dataset. Previously uncommitted source is backed up, grouped into functional
commits, pushed to an isolated branch, and verified from a fresh clone. Code,
datasets, generated indexes, outputs, and unresolved provenance now have
separate catalogs.

## Git branch and commits

Branch: `organize/catalog-and-code-snapshot-20260828`

```text
2259890 docs: add code and dataset lineage catalog
e83c6f6 chore: preserve dependency snapshots
7dbed97 checkpoint: preserve exploratory diagnostic script
258db17 checkpoint: preserve obj-to-language changes
b4c15a8 feat(data): preserve construction sequence generators
3ebbb12 feat(data): preserve polygon proxy pipeline
```

The branch was pushed to `origin` after both the code checkpoint and catalog
commit.

## Repository-external backup

Backup root:

`/mnt/d/artifacts/gendiff-data-process/snapshots/20260828_pre_organization`

```text
tracked_worktree.patch
  sha256 90b2b04d5aa66eb7f4e3bce36f7bfbe99b2dc13e70659e03806f79da13e9c271

status.txt
  sha256 c5acfd2f4e2402a4975da47211f7280051aa741d55030ab005e576b0cb396231

source_snapshot.tar.gz
  sha256 89074dec873302a903a64f571a8b0f5e7654e50de1f2390c8da26d017f91df94
  gzip verification passed; 25 archived source entries

newly_revealed_yaml.tar.gz
  sha256 6bb4ab3392447692eeaebf3c3cce2a744b54ca58c0deeafb9e022ceb465a76bc
  gzip verification passed
```

The second archive contains historical generated YAML indexes that became
visible after removing the blanket `*.yaml` ignore.

## Catalog results

- 119 `data/<category>/<dataset>` directories cataloged.
- 5,672,789 files observed.
- 3,587,926,590,672 bytes observed (about 3.59 TB decimal).
- 119 per-dataset manifests plus one aggregate dataset index generated.
- Every tracked Python path has exactly one entry in
  `catalog/code_inventory.yaml`.
- Four machine-readable pipeline descriptions distinguish candidate, legacy,
  and unresolved target flows.
- Historical `output/`, `outputs/`, and `pipeline/output/` roots are inventoried
  but remain untouched and ignored.
- Nine generated YAML pair indexes are cataloged by size, row count, schema,
  and SHA-256. `data.yaml` is byte-identical to
  `config/yuehai_with_remove.yaml`.

All 127 catalog YAML files parsed successfully.

## Verification

Fresh clone:

`/mnt/d/projects/gendiff-data-process-clean-20260828`

Verified at commit `22598906871384ad00a46e44492ec2ddbfa36a09`:

- local HEAD equals the pushed remote branch;
- worktree is clean;
- no `data/` or `output/` tree is present;
- fresh `.git` is about 568 KiB and the checkout about 2.4 MiB;
- all selected Python files compile;
- five polygon proxy synthetic tests pass;
- `uv lock --check` passes;
- all catalog YAML files parse;
- dataset index count equals the 119 manifests;
- code inventory has no missing or extra Python path.

## Intentionally not changed

- No dataset or historical output was moved, renamed, edited, or deleted.
- The approximately 17.09 GiB temporary pack in the old `.git` directory was
  not removed.
- No algorithm implementation was refactored.
- No pipeline was promoted to `current` without downstream evidence.

## Highest-priority next work

1. Link the exact GenDiff training repository commit, loader, config, and schema
   to the candidate canonical dataset.
2. Add canonicalizer invariance, determinism, collision, ambiguity, and
   round-trip tests before generating more training data.
3. Reconcile `pyproject.toml`/`uv.lock` with the historical Conda/Pip snapshots
   and create a runnable data-generation environment.
4. Add a run-manifest writer that records dataset IDs, commit, clean/dirty
   state, command, config, seed, environment, hashes, and validation reports.
5. Execute one small frozen end-to-end run and a training-loader smoke/overfit
   test.
6. Only after those gates pass, migrate small candidate outputs to the external
   artifact layout; address the old temporary Git pack last.

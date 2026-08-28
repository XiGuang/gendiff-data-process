# Project organization acceptance

Date: 2026-08-28

Scope: repository organization and lineage metadata only. This report does not
approve canonicalizer/adapter implementation, data generation, migration,
training, or changes to `/mnt/d/projects/GenDiff`.

## Audit state

- Branch: `organize/finalize-project-organization-20260828`
- Code snapshot commit inventoried: `ddd7a68685042dc93331a105d0f8b7449c8e467a`
- Phase 1 files preserved from the initially detached worktree:
  `catalog/training_consumer_manifest.yaml`,
  `docs/TRAINING_CONSUMER_AUDIT.md`, and
  `docs/CANONICALIZER_TEST_PLAN.md`.
- Legacy data root checked by directory metadata only: `/mnt/d/data/data`.
- GenDiff baseline used to detect accidental writes: commit
  `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`, dirty-status SHA-256
  `6f46f8383742f8cda5a30916fcc2c909de56f32a69d19e9082402cd8c595ecb3`.

## Acceptance matrix

| ID | Result | Acceptance item | Evidence | Remaining human confirmation |
|---|---|---|---|---|
| ORG-01 | PASS | Work is isolated on the requested branch without overwriting an existing ref. | `git show-ref --verify` failed for local/remote refs; `git ls-remote --heads origin organize/finalize-project-organization-20260828` returned empty; branch created from `ddd7a686...`. | None. |
| ORG-02 | PASS | Git tracks code/metadata only, not legacy data, outputs, or environments. | `.gitignore`; `git ls-files | rg '^(data|output|outputs|pipeline/output|\.venv)(/|$)'` returned no paths. | Confirm future contributors keep source configs outside ignored `/config/`. |
| ORG-03 | PASS | Every tracked Python file has one purpose and valid status. | `git ls-files '*.py'` = 67; parsed `catalog/code_inventory.yaml` = 67 unique entries; missing/extra/duplicate/blank/invalid-status sets all empty. | Purpose classifications remain reviewable judgments; no script was promoted to `current`. |
| ORG-04 | PASS | Every observed `/mnt/d/data/data/<category>/<dataset>` directory has dataset ID, lifecycle, and status. | Bounded two-level `Path.iterdir()` found 119 directories; `catalog/datasets/index.yaml` and 119 linked manifests cover the same physical-path set; required-field gaps = 0. | Directory contents and content hashes were intentionally not rescanned. |
| ORG-05 | PASS | Dataset index and manifests agree structurally and arithmetically. | Parsed YAML comparison: 119 rows/files, no missing/unlisted IDs; aggregate 5,672,789 files and 3,587,926,590,672 bytes; category counts/sizes match. | Observed counts/sizes remain metadata snapshots dated 2026-08-28. |
| ORG-06 | PASS | Candidate/current data records have producer and consumer evidence or explicit `unknown`. | One candidate `data/` manifest has explicit producer fields and `consumers: [unknown]`; all nine `candidate_run` entries in `catalog/legacy_outputs.yaml` now have producer plus `consumers: [unknown]`; no dataset is `current`. | Exact command/config/commit/consumer for candidate runs remains unknown. |
| ORG-07 | FAIL | Legacy and observed training datasets are fully reproducible from immutable lineage. | `docs/UNRESOLVED_PROVENANCE.md`; `catalog/training_consumer_manifest.yaml` records unknown historical run commit/diff/environment and packed producer command/commit. | Locate immutable CityEngine source, generation/packing argv, commit/diff, environment, hashes, and validation reports. |
| ORG-08 | FAIL | The construction canonical candidate is compatible with the actual training consumer. | `docs/TRAINING_CONSUMER_AUDIT.md`; `producer_compatibility.direct_loader_compatibility: blocked`; 12 mismatches in `catalog/training_consumer_manifest.yaml`. | Separate review must approve adapter semantics and test plan before implementation. |
| ORG-09 | PASS | Actual consumer, candidate construction pipeline, and legacy pipelines are separated without false promotion. | `docs/CURRENT_PIPELINE.md`; `catalog/pipelines/construction_sequence_v1.yaml`; `catalog/pipelines/history_area_edit_v2.yaml`; no pipeline status is `current`. | Review the `blocked_target` classification; it is not a release claim. |
| ORG-10 | PASS | Entry navigation answers code, data, lineage, consumer, and status questions without duplicating catalogs. | `README.md` and `catalog/README.md` link the code inventory, dataset index/manifests, legacy outputs, training consumer evidence, pipeline decision, unresolved register, and this report. | None. |
| ORG-11 | PASS | Final bounded validation passes after all edits. | 128 catalog YAML files parsed; manifest evidence 29/29 and mismatches 12; code inventory 67/67; dataset catalog 119/119; 67 Python files compiled with `tokenize.open()` and no pyc writes; 5 synthetic tests passed; `uv lock --check`, Pandoc rendering, Git boundary, and `git diff --check` passed. | Real-data and canonicalizer tests remain out of scope and must not be inferred from this PASS. |
| ORG-12 | PASS | Requested code-only clone exists at `/mnt/d/projects/gendiff-data-process`, matches the pushed branch, excludes forbidden trees, and is clean. | Target absence was confirmed before clone. Fresh clone HEAD/upstream both equaled `e884896f99371b3ccfdd7d31c8034388efc23c55`; porcelain status was empty; `data`, `output`, `outputs`, `pipeline/output`, and `.venv` were absent; clone validation reported 128 YAML, 67 Python, 119 dataset rows, 5 passing synthetic tests, and a valid lock. | `/mnt/d/data` remains the legacy compatibility root; no migration or symlink change was made. |

FAIL is intentional for unresolved lineage and compatibility. Project
organization can be accepted while those technical gates remain blocked, but no
pipeline or dataset may be promoted to `current`.

## Bounded validation commands

These checks read source/catalog metadata only and do not generate data:

```bash
git status --short --branch
git diff --check
git ls-files | rg '^(data|output|outputs|pipeline/output|\.venv)(/|$)'
python3 -c '<parse every catalog YAML with yaml.safe_load>'
python3 -c '<compare tracked Python paths with code_inventory entries>'
python3 -c '<compare dataset index/manifests with two-level directory metadata>'
python3 -c '<decode with tokenize.open and compile every tracked Python source, without pyc writes>'
python3 -m unittest tests.test_polygon_proxy.PolygonProxySyntheticTests
uv lock --check
pandoc --from=gfm --to=html <modified-doc> -o /dev/null
```

The exact inline Python assertions and their counts belong in the final handoff
command summary. Real-data tests, full hashing/scanning, generation, dependency
installation, and training are out of scope.

## Exact next gate

Record and push this acceptance result, fast-forward the fresh clone to the
recording commit, verify clean status and forbidden-path absence once more, then
stop after organization handoff. Canonicalizer, adapter, generation, and
training work require a new reviewed task and remain blocked by ORG-07/ORG-08.

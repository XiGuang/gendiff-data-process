# gendiff-data-process

Data-generation, conversion, validation, and catalog utilities for DiffGen.
This repository stores code and small metadata only. Multi-terabyte datasets
and generated artifacts stay outside Git and are referenced through manifests.

## Project map

| Question | Source of truth |
|---|---|
| Where is the code and what does each script do? | `catalog/code_inventory.yaml` lists every tracked Python path with one `purpose` and `status`. |
| Where is each legacy dataset and what is its lifecycle? | `catalog/datasets/index.yaml` links every observed `/mnt/d/data/data/<category>/<dataset>` directory to a manifest under `catalog/datasets/`. |
| Who produced and consumes a dataset or output? | Per-dataset `producer`/`consumers` fields, `catalog/legacy_outputs.yaml`, and `catalog/training_consumer_manifest.yaml`. Unknown evidence is written as `unknown`. |
| Which pipelines are legacy, candidate, or blocked? | `docs/CURRENT_PIPELINE.md` and `catalog/pipelines/`. |
| Which claims still cannot be reproduced? | `docs/UNRESOLVED_PROVENANCE.md`. |
| Did the repository organization pass its checks? | `docs/ORGANIZATION_ACCEPTANCE.md`. |

See `catalog/README.md` for field meanings and lookup order, and
`docs/DATA_LAYOUT.md` for the code/data/artifact boundary. Historical execution
details remain in `docs/ORGANIZATION_REPORT_20260828.md`.

## Pipeline status

No data-generation pipeline in this repository is marked `current`. The actual
observed GenDiff training consumer is now documented, but it consumes packed
`area_v2_packed_v1` PT data rather than this repository's loose canonical
candidate output. Direct compatibility is blocked; see:

- `catalog/training_consumer_manifest.yaml`: commit/config/command, loader
  contract, paths, evidence IDs, mismatches, and explicit unknowns.
- `docs/TRAINING_CONSUMER_AUDIT.md`: human-readable evidence chain.
- `docs/CANONICALIZER_TEST_PLAN.md`: planned gates only; no implementation has
  started.

## Candidate pipeline entry points

```text
polygon_proxy/core.py + tools/build_polygon_proxy*.py
building_process/batch_polygon_proxy_flat.py
building_process/generate_construction_sequence.py
building_process/generate_construction_sequence_canonical_edit_dataset.py
```

The repository contains two polygon proxy implementations. Read
`docs/CURRENT_PIPELINE.md` before combining their outputs.

## Data and artifact boundary

- `/mnt/d/data` remains the legacy compatibility root and is not moved by this
  repository organization work.
- `/mnt/d/data/data`, `/mnt/d/data/output`, `/mnt/d/data/outputs`, and
  `/mnt/d/data/pipeline/output` remain outside Git and are cataloged in place.
- A code-only clone must not contain `data/`, `output/`, `outputs/`,
  `pipeline/output/`, or `.venv/`.

## Refresh the observational data catalog

```bash
python tools/build_data_catalog.py \
  --data-root data \
  --output-dir catalog/datasets \
  --cataloged-at YYYY-MM-DD
```

The scanner is read-only with respect to datasets: it records filesystem
metadata and writes only the catalog manifests.

## Copy files while preserving relative paths

Script: `copy_by_relative_path.py`

Copy selected file types from an input directory to an output directory while
keeping the same relative folder structure.

```bash
python copy_by_relative_path.py \
  --input data/condition/yuehai_building_and_ground_combinations \
  --output /tmp/filtered \
  --ext .ply .npz
```

Common options:

- `--ext`: extensions to include (case-insensitive); empty means all files.
- `--overwrite`: overwrite existing files in the output.
- `--dry-run`: print statistics without copying.

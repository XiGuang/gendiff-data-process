# gendiff-data-process

Data-generation, conversion, validation, and catalog utilities for DiffGen.
This repository stores code and small metadata only. Multi-terabyte datasets
and generated artifacts stay outside Git and are referenced through manifests.

## Start here

- `docs/CURRENT_PIPELINE.md`: current decision and candidate pipeline gates.
- `catalog/code_inventory.yaml`: purpose and status of every Python file.
- `catalog/datasets/index.yaml`: observed legacy datasets, sizes, and manifests.
- `catalog/pipelines/`: machine-readable candidate and legacy DAGs.
- `docs/DATA_LAYOUT.md`: code/data/artifact separation and migration order.
- `docs/UNRESOLVED_PROVENANCE.md`: evidence and reproducibility gaps.

No generation pipeline is marked `current` yet. The construction canonicalizer
is a candidate pending determinism, uniqueness, round-trip, collision, and
training-consumer validation.

## Candidate pipeline entry points

```text
polygon_proxy/core.py + tools/build_polygon_proxy*.py
building_process/batch_polygon_proxy_flat.py
building_process/generate_construction_sequence.py
building_process/generate_construction_sequence_canonical_edit_dataset.py
```

The repository contains two polygon proxy implementations. Read
`docs/CURRENT_PIPELINE.md` before combining their outputs.

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

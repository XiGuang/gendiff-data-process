# Data and code catalog

This directory is the source of truth for understanding the legacy `data/`
repository without moving its multi-terabyte contents.

## What is recorded

- `code_inventory.yaml`: one primary status and a plain-language purpose for
  every tracked Python file.
- `datasets/index.yaml`: aggregate size/count summary and links to 119 observed
  `data/<category>/<dataset>` manifests.
- `datasets/*.yaml`: physical path, file count, byte size, extensions, mtime
  range, lifecycle, validation state, and provenance confidence.
- `pipelines/*.yaml`: candidate and legacy production DAGs.
- `legacy_outputs.yaml`: inventory of generated `output/`, `outputs/`, and
  `pipeline/output/` roots that remain outside Git.
- `legacy_yaml_indexes.yaml`: hashes and row counts for large generated YAML
  pair indexes that were previously hidden by the global `*.yaml` ignore.

## Confidence rules

`high` requires a recorded command/config and an immutable producer commit.
`medium` means code defaults, names, or timestamps strongly suggest a relation.
`low` means only a broad script-family/category relation is known. Unknown
fields stay `unknown`; they are not filled by guesswork.

## Updating the dataset inventory

Run from the repository root:

```bash
python tools/build_data_catalog.py \
  --data-root data \
  --output-dir catalog/datasets \
  --cataloged-at YYYY-MM-DD
```

The scanner reads metadata only. It does not move, rename, edit, or delete
dataset files. Review the generated diff before committing it.

## Status policy

- `current`: confirmed production path with downstream evidence.
- `candidate`: recent path that still needs end-to-end validation.
- `legacy`: explains existing historical datasets.
- `support`: inspection, conversion, copying, reporting, or catalog tooling.
- `experiment`: smoke tests and one-off diagnostics.
- `deprecated`: superseded, but retained until migration validation.
- `unknown`: evidence is insufficient.

At the time of this catalog, no data-generation pipeline is promoted to
`current`. The canonical construction pipeline is a candidate.

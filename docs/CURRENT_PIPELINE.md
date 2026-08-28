# Current and candidate pipeline

## Decision

No pipeline is marked `current` yet. The most recent coherent candidate is:

```text
component/tile OBJ
  -> polygon proxy YAML
  -> construction sequence
  -> canonical edit objects and v2 sequences
  -> GenDiff training loader (not yet confirmed)
```

The candidate entry point for the last stage is
`building_process/generate_construction_sequence_canonical_edit_dataset.py`.
Its presence and recency are not enough to call it production-ready.

## Two polygon proxy implementations

The repository currently contains two distinct implementations:

1. `polygon_proxy/core.py`, used by `tools/build_polygon_proxy*.py`, emits
   proxy JSON/OBJ/metrics.
2. `building_process/polygon_proxy.py`, used by
   `building_process/batch_polygon_proxy*.py`, can emit the flat YAML consumed
   by construction scripts.

Their schemas and geometry behavior have not been proven equivalent. A run
manifest must name the exact entry point and producer commit; “polygon proxy”
alone is ambiguous.

## Verification performed during organization

- All newly preserved Python files passed bytecode compilation.
- Five synthetic tests for `polygon_proxy/core.py` passed.
- `uv.lock` passed `uv lock --check`.
- Construction CLIs were not integration-tested because the active Python
  environments lack some of `torch`, `trimesh`, and `shapely`.
- Real-data polygon tests depend on ignored local datasets and are therefore not
  self-contained in a fresh clone.

## Promotion checklist

Before marking the canonical pipeline `current`:

1. Select one polygon proxy schema and provide an explicit adapter if needed.
2. Record an immutable input dataset ID, commit, command, config, seed, and
   dependency environment for every run.
3. Add canonical uniqueness, symmetry, permutation, round-trip, and collision
   tests.
4. Validate the complete pipeline on a small frozen dataset.
5. Confirm the actual GenDiff loader can read it and complete a small overfit
   experiment.

See `catalog/pipelines/construction_sequence_v1.yaml` and
`catalog/pipelines/history_area_edit_v2.yaml` for the machine-readable gates.

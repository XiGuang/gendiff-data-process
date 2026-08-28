# Current and candidate pipeline

## Decision

No data-generation pipeline in this repository is marked `current`. Phase 1
confirmed an actual GenDiff training consumer, but it is a separate historical
area-v2 path and does not establish consumption of this repository's candidate
canonical outputs.

The two evidence chains must remain separate.

## Observed GenDiff training consumer

```text
/mnt/d/projects/GenDiff/datasets/history_stages_origin
  -> area-v2 loose generation (exact argv/commit unknown)
  -> /mnt/d/projects/GenDiff/datasets/history_area_edit_v2_packed
       schema: area_v2_packed_v1
       edit schema: area_v2_absolute_target_coord_no_anchor
  -> configs/packed_area_edit_v2_tiny_overfit.yaml
  -> craftsman/data/packed_area_edit_v2_data_module.py
  -> BuildingLayerEditV2System / BackboneBuildingLayerEditV2
  -> outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1
```

Evidence:

- inspected GenDiff checkout: branch `structured_proxy`, commit
  `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`, dirty before and after the
  read-only audit;
- observed run command:
  `python train.py --config configs/packed_area_edit_v2_tiny_overfit.yaml --train --gpu 0,1,2,3,4,5,6`;
- run config SHA-256:
  `6b61d262991fec4ec3b1787385a3f0fb0d074dddd87f3f0dd1acb9317ae40441`;
- exact paths, selected file hashes, bounded sample results, and evidence IDs:
  `catalog/training_consumer_manifest.yaml`;
- human-readable trace: `docs/TRAINING_CONSUMER_AUDIT.md`.

The historical run commit/diff, exact data-generation command and environment
are `unknown`. Train, validation, and test all alias the same 20,000 samples, so
the observed metrics are not held-out evidence.

## Construction canonical candidate

```text
component/tile OBJ
  -> polygon proxy YAML
  -> construction sequence
  -> loose canonical edit objects and v2 sequences
  -X-> observed packed GenDiff loader
```

The candidate entry point is
`building_process/generate_construction_sequence_canonical_edit_dataset.py`,
introduced by commit `b4c15a89852df01c836dded8aef75a6d5b320bb2` and cataloged as
`candidate`. Phase 1 found direct compatibility **blocked**: the candidate does
not emit the required packed container/edit schema, uses incompatible
normalization and edit-value semantics, aliases its splits, and has no reviewed
`canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor` adapter.

The observed `/mnt/d/data/outputs/canonical` and `canonical_obj` directories are
candidate runs with known script-family producer and `unknown` exact
command/config/commit/consumer. Their names do not prove training use; see
`catalog/legacy_outputs.yaml`.

## Two polygon proxy implementations

1. `polygon_proxy/core.py`, used by `tools/build_polygon_proxy*.py`, emits proxy
   JSON/OBJ/metrics.
2. `building_process/polygon_proxy.py`, used by
   `building_process/batch_polygon_proxy*.py`, can emit the flat YAML consumed
   by construction scripts.

Their schemas and geometry behavior have not been proven equivalent. A run
manifest must name the exact entry point and producer commit; "polygon proxy"
alone is ambiguous.

## Verification state

- Code inventory coverage and synthetic polygon proxy tests were verified
  during organization; see `docs/ORGANIZATION_ACCEPTANCE.md`.
- The Phase 1 training audit was bounded and read-only. It did not generate
  data, train, install dependencies, or modify `/mnt/d/projects/GenDiff`.
- Canonicalizer tests in `docs/CANONICALIZER_TEST_PLAN.md` are planned, not
  implemented or passed.

## Exact promotion gate

Organization and its fresh-clone verification must finish first. After separate
review approval, implement tests and a versioned v3-to-area-v2 adapter, then the
canonical core. Unit/golden/property/adapter/loader-smoke gates must all pass
before a small versioned pilot; pilot determinism, round-trip, collision,
capacity, and building-level split gates must pass before bounded overfit. No
bulk generation or training is authorized by this document.

Machine-readable pipeline state is in
`catalog/pipelines/construction_sequence_v1.yaml` and
`catalog/pipelines/history_area_edit_v2.yaml`.

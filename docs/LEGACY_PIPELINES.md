# Legacy pipelines

The historical code is preserved because it explains the existing 3.59 TB
observed data tree. It is not mixed with the newer polygon/construction
candidate path.

## Block and condition family

The broad historical relation is:

```text
data/origin + data/component
  -> cut/merge/process_change scripts
  -> data/block
  -> process_combination and rotation scripts
  -> data/condition
  -> external render/encoder steps
  -> data/images and data/latents
  -> gen_yaml scripts
  -> data/yaml
```

This is a low-confidence family-level relation. Exact commands, configs,
commits, and input versions were not retained, so individual manifests keep
those fields as `unknown`.

## Point-cloud pipeline outputs

`pipeline/select_t1_and_t2.py` and
`pipeline/segment_and_normalize_point_cloud.py` are associated with
`pipeline/output/{blocks,global,metrics.json}`. These outputs occupy about 5.96
GB by apparent file size and remain ignored by Git.

## OBJ/YAML language experiments

`obj_to_language/` contains converters whose component footprint schema is not
the same as the current GenDiff CityEngine stage schema. They remain useful for
inspection and round-trip experiments, but they are not confirmed producers of
current `history_stages_*` or `area` training data.

## Retention rule

Names such as `tmp`, `test`, `new`, `fixed`, `fast`, or `exact` are not deletion
criteria. A legacy directory can be archived only after its consumer status,
content hash, retention need, and replacement have been recorded.

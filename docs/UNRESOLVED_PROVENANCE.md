# Unresolved provenance and risk register

## P0: blocks safe training claims

- The observed GenDiff consumer is now linked in
  `catalog/training_consumer_manifest.yaml`: inspected checkout commit
  `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`, loader
  `/mnt/d/projects/GenDiff/craftsman/data/packed_area_edit_v2_data_module.py`,
  run command/config, and required `area_v2_packed_v1` /
  `area_v2_absolute_target_coord_no_anchor` schemas are known. The historical
  run commit, dirty diff hash, exact Python environment, packed-data generation
  command, and producer commit remain `unknown`.
- The observed packed release is not a valid held-out release: train, validation,
  and test alias the same 20,000 samples. Evidence is
  `docs/TRAINING_CONSUMER_AUDIT.md` and manifest evidence IDs `E14`/`E15`.
- This repository's canonical construction candidate is not directly compatible
  with the observed loader. The missing versioned
  `canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor` adapter,
  normalization/schema differences, and 12 classified mismatches block a
  producer-to-consumer claim.
- Canonical edit uniqueness is unmeasured. Cyclic polygon starts, winding,
  symmetric shapes, layer ordering, nearest matches, and float tolerances can
  produce multiple sequences for equivalent geometry.
- The canonical candidate has no full-dataset round-trip, collision, ambiguity,
  or duplicate-key report.
- Construction CLIs cannot currently run in the active environment because the
  declared project environment and historical Conda/Pip snapshots disagree;
  `torch` is absent from the uv project dependencies.

## P1: blocks reliable regeneration

- Nearly all legacy datasets lack an exact command, config, seed, and producer
  commit.
- The current CityEngine stage YAML producer and its immutable source/config are
  `unknown`; the local `obj_to_language` schema is not evidence for that link.
- `data/images` and `data/latents` likely cross repository boundaries; their
  producers are unknown.
- The two polygon proxy implementations have not been schema- or
  geometry-equivalence tested.
- Real-data tests use ignored local paths and have no small versioned fixture.
- Several one-off scripts contain hard-coded `/mnt/d/...` paths.
- Historical generated YAML pair indexes contain absolute `/mnt/d/data/...`
  paths; `data.yaml` is byte-identical to `config/yuehai_with_remove.yaml`.
- `requirements.txt`/`requirements_pip.txt` are historical environment exports,
  while `pyproject.toml`/`uv.lock` describe a narrower code environment.

## P2: storage and hygiene

- The legacy `.git` directory is about 18 GB and contains an approximately
  17.09 GiB temporary pack. It was intentionally not deleted before backup and
  fresh-clone verification.
- Historical outputs mix candidate runs, diagnostics, and scratch data. They
  are inventoried in `catalog/legacy_outputs.yaml` but not yet assigned immutable
  run IDs. Candidate output consumers are explicitly `unknown`; the observed
  packed GenDiff run is not evidence that it consumed those directories.
- Dataset manifests record apparent byte sizes and file metadata, not content
  hashes; hashing all 3.59 TB is a separate planned operation.

## Evidence needed to close an item

For a pipeline/dataset relation to reach high confidence, attach:

1. immutable producer commit and clean-worktree proof;
2. exact command/config/seed/environment;
3. immutable input dataset IDs and content hashes;
4. output counts/sizes/hashes;
5. validation report, including canonical collisions and round-trip results;
6. downstream consumer commit/config and a successful smoke/overfit result.

Unknown fields must remain `unknown`; directory names and mtimes are supporting
clues, not proof.

## Confirmed relations that are no longer unknown

- Observed run to command/config: high confidence, manifest evidence `E04/E08`.
- Observed run to selected consumer file bytes: high confidence, `E05`.
- Packed dataset to source-stage paths and recorded generation parameters: high
  confidence, `E14`.
- Candidate producer to the observed training run: still `unknown`, `E26/E27`.

These closures are scoped to the inspected paths and bytes. They do not promote
either pipeline or reconstruct the missing historical run manifest.

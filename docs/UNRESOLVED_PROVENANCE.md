# Unresolved provenance and risk register

## P0: blocks safe training claims

- The actual GenDiff training repository commit, loader, config, and schema that
  consume `history_stages_*`/`area` data are not linked from this repository.
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
  are inventoried but not yet assigned immutable run IDs.
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

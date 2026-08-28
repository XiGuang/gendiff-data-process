# Code, data, and artifact layout

## Present state

The legacy checkout at `/mnt/d/data` contains both the Git repository and 3.59
TB of observed datasets. This organization pass deliberately does not move
those datasets. It introduces clear logical boundaries first.

```text
Git repository
  code + tests + configs + catalog + docs

/mnt/d/data/data
  legacy physical datasets (ignored, cataloged in place)

/mnt/d/data/output, /mnt/d/data/outputs, /mnt/d/data/pipeline/output
  historical scratch and run artifacts (ignored, cataloged in place)

/mnt/d/artifacts/gendiff-data-process
  repository-external backups and future immutable run artifacts
```

## Git boundary

`.gitignore` uses root directory boundaries for data and output trees. YAML is
no longer globally ignored, so configs, schemas, fixtures, and catalog
manifests can be versioned. Large OBJ/PLY/PT/NPZ/image artifacts remain ignored
as a safety net.

The existing `config/*.yaml`, root `data.yaml`, and
`pipeline/pairs_t1_t2.yaml` are large generated pair indexes with absolute data
paths, not source configuration. They are cataloged by hash and remain ignored.
New source configuration should use a clearly versioned `configs/` directory.

## Target state for new runs

New production work should write outside the code checkout:

```text
/mnt/d/datasets/<dataset_id>/...
/mnt/d/artifacts/gendiff-data-process/runs/<run_id>/
  run.yaml
  config.yaml
  logs/
  reports/
  outputs/ (or links to immutable datasets)
```

Every `run.yaml` should record:

- input dataset IDs and hashes;
- repository URL and immutable commit;
- clean/dirty state and diff SHA-256 when dirty;
- full command, config, seed, environment, host, start/end time;
- output dataset IDs, counts, sizes, hashes, and validation reports.

## Migration order

1. Fresh-clone and verify the pushed code/catalog branch.
2. Create the external dataset/artifact roots and permissions.
3. Route one small new run to the new artifact layout.
4. Migrate small `output/`/`outputs/` candidates with manifests and reversible
   links.
5. Migrate large datasets only after consumers and hashes are confirmed.

No bulk move or deletion is authorized by this document.

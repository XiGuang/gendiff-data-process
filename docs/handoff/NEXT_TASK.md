# Server Codex handoff: training consumer and canonicalizer audit

Date: 2026-08-28

## Objective

Establish the exact contract between the current GenDiff training code and the
candidate canonical construction dataset before changing algorithms or
generating more data.

This is an evidence-gathering task. The output must make it possible to answer:

1. Which repository commit, loader, config, command, and physical data paths are
   actually used by training?
2. Which fields, shapes, token semantics, coordinate normalization, IDs, and
   split conventions does the loader require?
3. Does the candidate canonicalizer emit that contract exactly?
4. Where can equivalent geometry still produce multiple supervision sequences?
5. What small deterministic test suite must pass before implementation or bulk
   regeneration begins?

## Starting state

- Data repository branch:
  `organize/catalog-and-code-snapshot-20260828`
- Verified data-repository commit before this handoff:
  `0fb8505f37fe7709830be2e0e9949fc29243e307`
- Clean verification checkout:
  `/mnt/d/projects/gendiff-data-process-clean-20260828`
- Legacy mixed code/data checkout:
  `/mnt/d/data`
- Candidate canonical implementation:
  `building_process/generate_construction_sequence_canonical_edit_dataset.py`
- Expected training repository from the earlier audit:
  `/mnt/d/projects/GenDiff`

The legacy dataset inventory contains 119 dataset directories, 5,672,789 files,
and 3,587,926,590,672 observed bytes. Do not rescan or hash the full tree in this
phase.

## Phase 1 procedure

### 1. Record repository state

For the GenDiff training repository, record without modifying it:

- resolved path and remote URL;
- branch, HEAD commit, upstream, and dirty status;
- relevant untracked or modified files without exposing secrets;
- Python/environment descriptors and the command actually used for training.

If more than one GenDiff checkout exists, identify which one produced the
latest logs/checkpoints and explain the evidence.

### 2. Trace the real training consumer

Locate and trace:

- dataset/DataLoader entry point;
- config composition and selected dataset section;
- sample manifest paths and split files;
- collate/tokenization/normalization code;
- loss targets and masks;
- model forward inputs;
- inference/decoding/apply path;
- latest training command, logs, and checkpoint metadata.

Record exact file paths and line/function names. Distinguish code defaults from
the configuration that was actually used.

### 3. Inspect a bounded sample

Inspect only a small representative sample sufficient to report:

- field names, dtypes, shapes, ranges, and missing-value behavior;
- whether train/validation/test samples or source buildings overlap;
- presence and stability of layer/point/entity IDs;
- duplicate canonical targets for equivalent source/target geometry;
- no-op, demolition, reconstruction, ambiguity, and invalid-sample handling.

Do not mutate sample files. Do not scan or hash the entire 3.59 TB tree.

### 4. Compare producer and consumer contracts

Compare the current loader requirements with:

- the unified canonicalizer v1 specification;
- `generate_construction_sequence_canonical_edit_dataset.py` outputs;
- pipeline contracts under `catalog/pipelines/`.

Classify every mismatch as blocking, compatibility risk, or documentation gap.

### 5. Define the next executable test gate

Write a canonicalizer test plan covering at least:

- cyclic polygon start invariance;
- winding reversal invariance;
- layer/component permutation invariance;
- symmetric-geometry deterministic tie breaking;
- floating-point quantization boundaries;
- stable layer/edge/point identities;
- ambiguous correspondence fail-closed behavior;
- canonicalize/compile/apply round trip;
- repeated-run byte identity;
- duplicate canonical-key and conflicting-target detection;
- training loader smoke test and bounded overfit test.

Specify fixtures, metrics, commands, and pass/fail thresholds. Do not implement
the full canonicalizer in Phase 1.

## Required outputs

### `catalog/training_consumer_manifest.yaml`

Must contain:

- schema version and audit timestamp;
- training repository state;
- actual config/command evidence;
- loader and model consumer paths;
- input/output schema and normalization;
- dataset/split paths;
- producer compatibility status;
- provenance confidence per material relation;
- unresolved fields and blockers.

### `docs/TRAINING_CONSUMER_AUDIT.md`

Must lead with conclusions, then document the evidence chain, mismatches,
leakage/uniqueness risks, and recommended gate.

### `docs/CANONICALIZER_TEST_PLAN.md`

Must provide a test matrix detailed enough for a later Codex task to implement
without redefining semantics.

## Completion criteria

Phase 1 is complete only when:

- the actual training consumer is traced end to end;
- the loader contract is stated precisely enough to build fixtures;
- the candidate producer/consumer mismatches are classified;
- the three required deliverables exist and parse/render correctly;
- all blockers and inaccessible evidence are explicit;
- no dataset, output, training code, or legacy Git storage was modified.

Stop after reporting Phase 1. Wait for review before implementing the
canonicalizer or running bulk generation/training.

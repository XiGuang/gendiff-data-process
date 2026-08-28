# Repository operating instructions

## Required context

Before making changes, read these files completely:

1. `docs/handoff/NEXT_TASK.md`
2. `docs/handoff/GenDiff_unified_canonicalizer_spec_v1.md`
3. `docs/handoff/GenDiff_server_audit_2026-08-28.md`
4. `docs/handoff/Data_project_organization_and_lineage_plan.md`
5. `docs/CURRENT_PIPELINE.md`
6. `docs/UNRESOLVED_PROVENANCE.md`
7. `catalog/pipelines/construction_sequence_v1.yaml`
8. `catalog/pipelines/history_area_edit_v2.yaml`

Treat catalog fields marked `unknown` as unknown. Do not fill them from names or
timestamps alone.

## Current phase

The current task is Phase 1: a read-only audit of the actual GenDiff training
consumer and its contract with generated canonical edit data. Produce evidence
and tests/plans first. Do not start bulk regeneration or model training.

## Safety boundaries

- Do not move, rename, edit, or delete files under legacy `data/`, `output/`,
  `outputs/`, or `pipeline/output/` trees.
- Do not run `git clean`, `git gc`, destructive resets, or remove the legacy
  repository's temporary Git pack.
- Do not change the GenDiff training repository during Phase 1; inspect it
  read-only unless the user explicitly expands scope.
- Do not commit credentials, tokens, passwords, private keys, or SSH config.
- Do not run large-scale generation, hashing, training, or dependency installs.
- Do not promote a pipeline or dataset to `current` without command, commit,
  config, validation, and downstream-consumer evidence.
- Preserve existing user changes. Use explicit file lists with `git add`; do not
  use `git add .`.

## Required deliverables

Write Phase 1 results into this repository:

- `catalog/training_consumer_manifest.yaml`
- `docs/TRAINING_CONSUMER_AUDIT.md`
- `docs/CANONICALIZER_TEST_PLAN.md`

Unknown or inaccessible evidence must be recorded explicitly. Every material
claim should name the inspected path, commit/config, or command output that
supports it.

## Verification and handoff

- Validate every new YAML file with a parser.
- Run only small, targeted, non-destructive checks during Phase 1.
- Keep the working tree reviewable and report any pre-existing dirtiness before
  editing.
- At completion, summarize evidence, blockers, files changed, commands/tests,
  and the exact next gate. Do not silently continue into implementation.

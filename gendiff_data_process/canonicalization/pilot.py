from __future__ import annotations

import hashlib
import platform
import shutil
import socket
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import yaml  # type: ignore[import-untyped]

from .adapters.area_v2 import (
    AreaNormalizationStats,
    AreaV2Adapter,
    AreaV2Capacity,
    validate_area_v2_edit_capacity,
)
from .collision import audit_supervision_collisions
from .condition import CanonicalCondition, build_canonical_condition
from .config import CanonicalizerBundle, load_bundle
from .core import canonicalize_building_sequence
from .edit_v3 import apply_canonical_edit, build_canonical_edit
from .errors import CanonicalizationError
from .history_adapter import SourceFileEvidence, load_history_sequence
from .packed_contract import validate_packed_release_meta, validate_packed_sample
from .release_contracts import (
    assign_building_splits,
    building_uid_from_key,
    compute_train_normalization_profile,
)
from .serialize import canonical_hash, canonical_value
from .solid_partition import classify_stage_change
from .types import CanonicalBuildingSequence, CanonicalEdit


@dataclass(frozen=True)
class PilotTask:
    dataset_root: str
    building_key: str
    split: str
    stage_indices: tuple[int, ...]
    config_path: str


@dataclass(frozen=True)
class PilotPairResult:
    source_position: int
    target_position: int
    change_kind: str
    edit: CanonicalEdit
    condition: CanonicalCondition


@dataclass(frozen=True)
class PilotPairFailure:
    source_position: int
    target_position: int
    change_kind: str
    error_code: str
    error_message: str
    error_context: dict


@dataclass(frozen=True)
class PilotBuildingResult:
    building_key: str
    building_uid: str
    split: str
    source_files: tuple[SourceFileEvidence, ...]
    sequence: CanonicalBuildingSequence | None
    pairs: tuple[PilotPairResult, ...]
    noop_transition_indices: tuple[int, ...]
    pair_failures: tuple[PilotPairFailure, ...]
    error_code: str | None = None
    error_message: str | None = None
    error_context: dict | None = None


def building_keys_from_range(start: int, count: int) -> tuple[str, ...]:
    if start < 0 or not 1 <= count <= 100:
        raise ValueError("pilot building range 要求 start >= 0 且 1 <= count <= 100")
    return tuple(f"building_{index:04d}" for index in range(start, start + count))


def _process_task(task: PilotTask) -> PilotBuildingResult:
    building_uid = building_uid_from_key(task.building_key)
    source_files: tuple[SourceFileEvidence, ...] = ()
    try:
        bundle = load_bundle(task.config_path)
        loaded = load_history_sequence(
            task.dataset_root,
            task.building_key,
            task.stage_indices,
        )
        source_files = loaded.source_files
        sequence = canonicalize_building_sequence(loaded.sequence, bundle)
        pairs: list[PilotPairResult] = []
        noops: list[int] = []
        pair_failures: list[PilotPairFailure] = []
        bidirectional = bundle.validation_profile.mode == "bidirectional_monotonic"
        area_capacity = AreaV2Capacity(
            bundle.validation_profile.max_layers,
            bundle.validation_profile.max_points_per_layer,
            bundle.validation_profile.max_buildings_per_tile,
        )
        for pair_index, edit in enumerate(sequence.adjacent_edits):
            source = sequence.stages[pair_index]
            target = sequence.stages[pair_index + 1]
            change = classify_stage_change(
                source,
                target,
                volume_tolerance_q3=bundle.validation_profile.removed_volume_tolerance_q3,
            )
            if change.change_kind == "noop":
                noops.append(pair_index)
                continue
            if bidirectional and change.change_kind == "mixed":
                context = {
                    "added_volume2_q3": change.added_volume2_q3,
                    "removed_volume2_q3": change.removed_volume2_q3,
                }
                for source_position, target_position in (
                    (pair_index, pair_index + 1),
                    (pair_index + 1, pair_index),
                ):
                    pair_failures.append(
                        PilotPairFailure(
                            source_position,
                            target_position,
                            "mixed",
                            "E_MIXED_CHANGE_UNSUPPORTED",
                            "同一 transition 同时新增和删除体积",
                            context,
                        )
                    )
                continue

            directed_edits = [(pair_index, pair_index + 1, edit)]
            if bidirectional:
                reverse_edit = build_canonical_edit(
                    target, source, bundle.canonicalizer
                )
                directed_edits.append((pair_index + 1, pair_index, reverse_edit))
            for source_position, target_position, directed_edit in directed_edits:
                directed_source = sequence.stages[source_position]
                directed_target = sequence.stages[target_position]
                try:
                    condition = build_canonical_condition(
                        directed_source,
                        directed_target,
                        bundle.condition_sampling,
                        volume_tolerance_q3=bundle.validation_profile.removed_volume_tolerance_q3,
                    )
                    applied = apply_canonical_edit(
                        directed_source,
                        directed_edit,
                        bundle.canonicalizer,
                    )
                    if applied.stage_hash != directed_target.stage_hash:
                        raise CanonicalizationError(
                            "E_ROUNDTRIP_MISMATCH",
                            "directed pair edit target hash 不一致",
                        )
                    validate_area_v2_edit_capacity(
                        directed_source,
                        directed_edit,
                        area_capacity,
                    )
                    pairs.append(
                        PilotPairResult(
                            source_position,
                            target_position,
                            condition.change_kind,
                            directed_edit,
                            condition,
                        )
                    )
                except CanonicalizationError as exc:
                    if not bidirectional:
                        raise
                    directed_change = classify_stage_change(
                        directed_source,
                        directed_target,
                        volume_tolerance_q3=bundle.validation_profile.removed_volume_tolerance_q3,
                    )
                    pair_failures.append(
                        PilotPairFailure(
                            source_position,
                            target_position,
                            directed_change.change_kind,
                            exc.code,
                            exc.message,
                            dict(exc.context),
                        )
                    )
        return PilotBuildingResult(
            task.building_key,
            building_uid,
            task.split,
            source_files,
            sequence,
            tuple(pairs),
            tuple(noops),
            tuple(pair_failures),
        )
    except CanonicalizationError as exc:
        return PilotBuildingResult(
            task.building_key,
            building_uid,
            task.split,
            source_files,
            None,
            (),
            (),
            (),
            exc.code,
            exc.message,
            dict(exc.context),
        )
    except Exception as exc:
        return PilotBuildingResult(
            task.building_key,
            building_uid,
            task.split,
            source_files,
            None,
            (),
            (),
            (),
            "E_INPUT_ADAPTER",
            str(exc),
            {"exception_type": type(exc).__name__},
        )


def process_pilot_tasks(
    tasks: tuple[PilotTask, ...], workers: int
) -> tuple[PilotBuildingResult, ...]:
    if workers <= 0:
        raise ValueError("workers 必须为正数")
    if workers == 1:
        return tuple(_process_task(task) for task in tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(_process_task, tasks))


def pilot_fingerprint(results: Iterable[PilotBuildingResult]) -> str:
    payload = []
    for result in results:
        payload.append(
            {
                "building_key": result.building_key,
                "building_uid": result.building_uid,
                "split": result.split,
                "source_sha256": [item.sha256 for item in result.source_files],
                "sequence_hash": (
                    result.sequence.sequence_hash if result.sequence else None
                ),
                "stage_hashes": (
                    [stage.stage_hash for stage in result.sequence.stages]
                    if result.sequence
                    else []
                ),
                "edit_hashes": (
                    [edit.edit_hash for edit in result.sequence.adjacent_edits]
                    if result.sequence
                    else []
                ),
                "directed_pairs": [
                    {
                        "source_position": pair.source_position,
                        "target_position": pair.target_position,
                        "change_kind": pair.change_kind,
                        "edit_hash": pair.edit.edit_hash,
                        "condition_hash": pair.condition.condition_hash,
                    }
                    for pair in result.pairs
                ],
                "noop_transition_indices": result.noop_transition_indices,
                "pair_failures": [asdict(failure) for failure in result.pair_failures],
                "error_code": result.error_code,
                "error_context": result.error_context,
            }
        )
    return canonical_hash(payload)


def pair_accounting(
    results: tuple[PilotBuildingResult, ...],
    stage_indices: tuple[int, ...],
    *,
    emitted_sample_count: int,
    duplicate_row_count: int,
    conflicting_row_count: int,
    directions_per_transition: int = 1,
) -> dict[str, int]:
    if directions_per_transition not in {1, 2}:
        raise ValueError("directions_per_transition 必须是 1 或 2")
    pair_slots_per_building = max(0, len(stage_indices) - 1) * directions_per_transition
    attempted = len(results) * pair_slots_per_building
    noop_skipped = sum(
        len(result.noop_transition_indices) * directions_per_transition
        for result in results
    )
    failed_building_pair_slots = sum(
        pair_slots_per_building for result in results if result.sequence is None
    )
    pair_generation_failures = sum(len(result.pair_failures) for result in results)
    explicit_failures = (
        failed_building_pair_slots + pair_generation_failures + conflicting_row_count
    )
    accounted = (
        emitted_sample_count + noop_skipped + duplicate_row_count + explicit_failures
    )
    return {
        "attempted": attempted,
        "emitted": emitted_sample_count,
        "noop_skipped": noop_skipped,
        "duplicates_deduplicated": duplicate_row_count,
        "explicit_failures": explicit_failures,
        "failed_building_pair_slots": failed_building_pair_slots,
        "pair_generation_failures": pair_generation_failures,
        "collision_conflicting_rows": conflicting_row_count,
        "silent_drop_count": attempted - accounted,
    }


def make_pilot_tasks(
    dataset_root: str | Path,
    config_path: str | Path,
    building_keys: tuple[str, ...],
    stage_indices: tuple[int, ...],
    bundle: CanonicalizerBundle,
) -> tuple[PilotTask, ...]:
    if (
        len(stage_indices) < 2
        or any(index < 0 for index in stage_indices)
        or tuple(sorted(stage_indices)) != stage_indices
    ):
        raise ValueError("pilot stage_indices 必须包含至少两个严格递增的非负索引")
    assignments = assign_building_splits(building_keys, bundle.split)
    return tuple(
        PilotTask(
            str(Path(dataset_root).resolve()),
            building_key,
            assignments[building_key],
            stage_indices,
            str(Path(config_path).resolve()),
        )
        for building_key in building_keys
    )


def _write_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(canonical_value(value), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_output_hash(root: Path) -> str:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
        )
    return canonical_hash(records)


def _tree_summary(root: Path) -> dict:
    records = []
    total_size_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size_bytes = path.stat().st_size
        total_size_bytes += size_bytes
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "file_count": len(records),
        "total_size_bytes": total_size_bytes,
        "tree_hash": canonical_hash(records),
    }


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _pilot_dataset_name(bundle: CanonicalizerBundle) -> str:
    if bundle.validation_profile.mode == "bidirectional_monotonic":
        return "canonicalizer_pilot_bidirectional_v1"
    return "canonicalizer_pilot_v1"


def _source_manifest(results: tuple[PilotBuildingResult, ...]) -> dict:
    return {
        "schema_version": "canonicalizer_pilot_source_manifest_v1",
        "buildings": [
            {
                "building_key": result.building_key,
                "building_uid": result.building_uid,
                "split": result.split,
                "files": [asdict(item) for item in result.source_files],
            }
            for result in results
        ],
    }


def _pack_samples(
    run_root: Path,
    bundle: CanonicalizerBundle,
    profile,
    results: tuple[PilotBuildingResult, ...],
    producer: dict,
) -> tuple[dict, dict]:
    import torch

    dataset_root = run_root / "outputs" / _pilot_dataset_name(bundle)
    dataset_root.mkdir(parents=True)
    normalization = AreaNormalizationStats(
        profile.profile_id,
        profile.center_x,
        profile.center_z,
        profile.scale_xz,
        profile.center_y,
        profile.scale_y,
    )
    adapter = AreaV2Adapter(bundle, normalization)
    states: list[dict] = []
    state_indices: dict[tuple[str, str], int] = {}
    samples_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    collision_records: list[dict[str, str]] = []
    observed_supervision: dict[tuple[str, str], tuple[str, str]] = {}
    duplicate_rows = 0
    conflicting_rows: list[dict[str, object]] = []
    change_kind_counts: Counter[str] = Counter()

    canonical_dir = run_root / "canonical"
    for result in results:
        if result.sequence is None:
            continue
        sequence = result.sequence
        _write_yaml(
            canonical_dir / f"{result.building_key}.yaml",
            {
                "sequence": sequence,
                "directed_pairs": result.pairs,
                "noop_transition_indices": result.noop_transition_indices,
                "pair_failures": result.pair_failures,
            },
        )
        for directed_pair in result.pairs:
            source = sequence.stages[directed_pair.source_position]
            target = sequence.stages[directed_pair.target_position]
            edit = directed_pair.edit
            condition = directed_pair.condition
            key = (source.stage_hash, condition.condition_hash)
            supervision = (target.stage_hash, edit.edit_hash)
            previous = observed_supervision.get(key)
            if previous is not None:
                if previous != supervision:
                    conflicting_rows.append(
                        {
                            "building_key": result.building_key,
                            "source_position": directed_pair.source_position,
                            "target_position": directed_pair.target_position,
                            "change_kind": directed_pair.change_kind,
                            "source_stage_hash": key[0],
                            "condition_hash": key[1],
                            "previous": previous,
                            "current": supervision,
                        }
                    )
                    continue
                duplicate_rows += 1
                continue
            observed_supervision[key] = supervision
            collision_records.append(
                {
                    "source_stage_hash": source.stage_hash,
                    "condition_hash": condition.condition_hash,
                    "target_stage_hash": target.stage_hash,
                    "edit_hash": edit.edit_hash,
                }
            )

            normalized_condition = [
                normalization.normalize_xyz(
                    point,
                    bundle.canonicalizer.grid_xz,
                    bundle.canonicalizer.grid_y,
                )
                for point in condition.points_q
            ]
            pair = adapter.adapt_transition(
                sequence,
                directed_pair.source_position,
                directed_pair.target_position,
                edit,
                normalized_condition,
                dense_building_id=0,
                condition_hash=condition.condition_hash,
                split=result.split,
            )
            source_key = (sequence.building_uid, source.stage_hash)
            target_key = (sequence.building_uid, target.stage_hash)
            if source_key not in state_indices:
                state_indices[source_key] = len(states)
                states.append(pair["source_state"])
            if target_key not in state_indices:
                state_indices[target_key] = len(states)
                states.append(pair["target_state"])
            sample = pair["sample"]
            if sample["canonical_metadata"]["change_kind"] != directed_pair.change_kind:
                raise CanonicalizationError(
                    "E_PACKED_CANONICAL_METADATA",
                    "condition、pair 与 adapter 的 change_kind 不一致",
                )
            sample["source_state_index"] = state_indices[source_key]
            sample["target_state_index"] = state_indices[target_key]
            sample["condition"] = torch.tensor(sample["condition"], dtype=torch.float32)
            validate_packed_sample(sample)
            samples_by_split[result.split].append(sample)
            change_kind_counts[sample["canonical_metadata"]["change_kind"]] += 1

    collision_report = asdict(audit_supervision_collisions(collision_records))
    collision_report["duplicate_row_count"] = duplicate_rows
    collision_report["conflicting_key_count"] = len(
        {
            (record["source_stage_hash"], record["condition_hash"])
            for record in conflicting_rows
        }
    )
    collision_report["conflicting_row_count"] = len(conflicting_rows)
    collision_report["conflicts"] = conflicting_rows
    contract = adapter.canonical_contract()
    contract["producer_commit"] = producer["commit"]
    contract["package_wheel_sha256"] = producer["package_wheel_sha256"]
    contract["normalization_profile_hash"] = canonical_hash(profile.to_mapping())
    meta = {
        "schema_version": adapter.PACK_SCHEMA,
        "edit_schema_version": adapter.EDIT_SCHEMA,
        "states_path": str(dataset_root / "states.pt"),
        "canonical_contract": contract,
        "producer": producer,
        "split_sample_counts": {
            split: len(samples) for split, samples in samples_by_split.items()
        },
    }
    validate_packed_release_meta(meta)
    torch.save(meta, dataset_root / "dataset_meta.pt")
    _write_yaml(dataset_root / "dataset_meta.yaml", meta)
    torch.save(
        {
            "schema_version": adapter.PACK_SCHEMA,
            "states": states,
            "normalization_stats_tensor": list(profile.tensor_order),
        },
        dataset_root / "states.pt",
    )

    shard_size = 32
    for split, samples in samples_by_split.items():
        shard_records = []
        for shard_index, start in enumerate(range(0, len(samples), shard_size)):
            shard_samples = samples[start : start + shard_size]
            shard_dir = dataset_root / "shards" / split
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"{split}_{shard_index:05d}.pt"
            torch.save(
                {
                    "schema_version": adapter.PACK_SCHEMA,
                    "sample_count": len(shard_samples),
                    "sample_offset": start,
                    "samples": shard_samples,
                },
                shard_path,
            )
            shard_records.append(
                {
                    "path": str(shard_path),
                    "split": split,
                    "sample_count": len(shard_samples),
                }
            )
        torch.save({"shards": shard_records}, dataset_root / f"{split}_index.pt")
        _write_yaml(dataset_root / f"{split}_index.yaml", {"shards": shard_records})

    return (
        {
            "dataset_root": str(dataset_root),
            "state_count": len(states),
            "split_sample_counts": {
                split: len(samples) for split, samples in samples_by_split.items()
            },
            "emitted_sample_count": sum(
                len(samples) for samples in samples_by_split.values()
            ),
            "change_kind_counts": dict(change_kind_counts),
            "duplicate_row_count_before_dedup": duplicate_rows,
            "conflicting_row_count": len(conflicting_rows),
            "collision_report": collision_report,
        },
        meta,
    )


def build_pilot(
    *,
    run_root: str | Path,
    dataset_root: str | Path,
    config_path: str | Path,
    building_start: int,
    building_count: int,
    stage_indices: tuple[int, ...],
    determinism_workers: tuple[int, ...],
    repository_path: str | Path,
    repository_remote: str,
    producer_commit: str,
    package_wheel: str | Path,
    command: tuple[str, ...],
) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    output_root = Path(run_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"pilot run 目录已存在，拒绝覆盖: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "reports").mkdir()
    package_dir = output_root / "package"
    package_dir.mkdir()

    config_source = Path(config_path).resolve()
    bundle = load_bundle(config_source)
    keys = building_keys_from_range(building_start, building_count)
    tasks = make_pilot_tasks(dataset_root, config_source, keys, stage_indices, bundle)
    results = process_pilot_tasks(tasks, workers=1)
    baseline_fingerprint = pilot_fingerprint(results)
    worker_fingerprints = {1: baseline_fingerprint}
    for workers in sorted(set(determinism_workers)):
        if workers == 1:
            continue
        worker_fingerprints[workers] = pilot_fingerprint(
            process_pilot_tasks(tasks, workers)
        )
    deterministic = len(set(worker_fingerprints.values())) == 1

    successful = tuple(result for result in results if result.sequence is not None)
    train_sequences = tuple(
        result.sequence
        for result in successful
        if result.split == "train" and result.sequence is not None
    )
    profile = compute_train_normalization_profile(
        train_sequences,
        bundle.normalization,
        grid_xz=bundle.canonicalizer.grid_xz,
        grid_y=bundle.canonicalizer.grid_y,
    )

    wheel_source = Path(package_wheel).resolve()
    wheel_target = package_dir / wheel_source.name
    shutil.copy2(wheel_source, wheel_target)
    wheel_hash = _sha256_file(wheel_target)
    config_target = output_root / "config.yaml"
    shutil.copy2(config_source, config_target)
    producer = {
        "repository": str(Path(repository_path).resolve()),
        "repository_remote": repository_remote,
        "commit": producer_commit,
        "dirty": False,
        "package_wheel": str(wheel_target),
        "package_wheel_sha256": wheel_hash,
        "command": list(command),
        "config": str(config_target),
        "config_source": str(config_source),
        "config_sha256": _sha256_file(config_target),
    }
    pack_report, _ = _pack_samples(output_root, bundle, profile, results, producer)

    source_manifest = _source_manifest(results)
    source_manifest["manifest_hash"] = canonical_hash(source_manifest["buildings"])
    _write_yaml(output_root / "source_manifest.yaml", source_manifest)
    _write_yaml(output_root / "normalization_profile.yaml", profile)

    errors = Counter(result.error_code for result in results if result.error_code)
    pair_errors = Counter(
        failure.error_code for result in results for failure in result.pair_failures
    )
    warnings = Counter(
        warning
        for result in successful
        for warning in (result.sequence.warnings if result.sequence else ())
    )
    directions_per_transition = (
        2 if bundle.validation_profile.mode == "bidirectional_monotonic" else 1
    )
    pairs = pair_accounting(
        results,
        stage_indices,
        emitted_sample_count=pack_report["emitted_sample_count"],
        duplicate_row_count=pack_report["duplicate_row_count_before_dedup"],
        conflicting_row_count=pack_report["conflicting_row_count"],
        directions_per_transition=directions_per_transition,
    )
    split_buildings = Counter(result.split for result in results)
    successful_split_buildings = Counter(result.split for result in successful)
    required_change_coverage = (
        bundle.validation_profile.mode != "bidirectional_monotonic"
        or all(
            pack_report["change_kind_counts"].get(kind, 0) > 0
            for kind in ("construction", "demolition")
        )
    )
    generation_pass = (
        deterministic
        and not errors
        and not pair_errors
        and pairs["silent_drop_count"] == 0
        and all(
            pack_report["split_sample_counts"].get(split, 0) > 0
            for split in ("train", "val", "test")
        )
        and pack_report["collision_report"]["conflicting_key_count"] == 0
        and required_change_coverage
    )
    report = {
        "schema_version": "canonicalizer_pilot_generation_report_v1",
        "status": "pass" if generation_pass else "fail",
        "pilot_fingerprint": baseline_fingerprint,
        "determinism": {
            "worker_fingerprints": worker_fingerprints,
            "mismatch_count": (
                0 if deterministic else len(set(worker_fingerprints.values())) - 1
            ),
        },
        "selection": {
            "building_start": building_start,
            "building_count": building_count,
            "stage_indices": stage_indices,
            "directions_per_transition": directions_per_transition,
            "selected_split_building_counts": dict(split_buildings),
            "successful_split_building_counts": dict(successful_split_buildings),
        },
        "building_results": {
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "errors_by_code": dict(errors),
            "warnings_by_code": dict(warnings),
            "failures": [
                {
                    "building_key": result.building_key,
                    "split": result.split,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                    "error_context": result.error_context,
                }
                for result in results
                if result.error_code
            ],
        },
        "pair_results": {
            "errors_by_code": dict(pair_errors),
            "required_change_coverage": required_change_coverage,
            "change_kind_counts": pack_report["change_kind_counts"],
            "failures": [
                {
                    "building_key": result.building_key,
                    "split": result.split,
                    **asdict(failure),
                }
                for result in results
                for failure in result.pair_failures
            ],
        },
        "pairs": pairs,
        "normalization_profile_id": profile.profile_id,
        "normalization_profile_hash": canonical_hash(profile.to_mapping()),
        "pack": pack_report,
        "producer": producer,
    }
    _write_yaml(output_root / "reports" / "generation_report.yaml", report)

    run_manifest = {
        "schema_version": "gendiff_data_process_run_v1",
        "run_id": output_root.name,
        "status": (
            "generated_pending_consumer_validation"
            if generation_pass
            else "failed_generation_gate"
        ),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "source_manifest": str(output_root / "source_manifest.yaml"),
            "source_manifest_hash": source_manifest["manifest_hash"],
        },
        "producer": producer,
        "config_hashes": {
            "geometry_config_hash": bundle.canonicalizer.geometry_config_hash,
            "canonicalizer_config_hash": bundle.canonicalizer.canonicalizer_config_hash,
            "validation_profile_hash": bundle.validation_profile.config_hash,
            "condition_config_hash": bundle.condition_sampling.config_hash,
            "normalization_config_hash": bundle.normalization.config_hash,
            "split_config_hash": bundle.split.config_hash,
            "package_config_hash": bundle.package.config_hash,
        },
        "outputs": {
            "dataset_root": pack_report["dataset_root"],
            "canonical_output_hash": _canonical_output_hash(output_root / "canonical"),
            "canonical_tree": _tree_summary(output_root / "canonical"),
            "packed_tree": _tree_summary(Path(pack_report["dataset_root"])),
            "generation_report": str(
                output_root / "reports" / "generation_report.yaml"
            ),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "pyyaml": _package_version("PyYAML"),
            "shapely": _package_version("shapely"),
            "geos": __import__("shapely").geos_version_string,
            "numpy": _package_version("numpy"),
            "fpsample": _package_version("fpsample"),
            "torch": _package_version("torch"),
        },
        "training_run": False,
    }
    _write_yaml(output_root / "run.yaml", run_manifest)
    return run_manifest

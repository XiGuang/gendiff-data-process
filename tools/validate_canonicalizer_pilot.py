from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml  # type: ignore[import-untyped]

from gendiff_data_process.canonicalization.collision import audit_supervision_collisions
from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.packed_contract import (
    validate_packed_release_meta,
    validate_packed_sample,
)
from gendiff_data_process.canonicalization.release_contracts import (
    building_uid_from_key,
    split_for_building_uid,
)
from gendiff_data_process.canonicalization.serialize import (
    canonical_hash,
    canonical_value,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证 canonicalizer pilot 与真实 GenDiff loader 合同"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gendiff-repo", required=True)
    parser.add_argument("--repository", default=str(REPO_ROOT))
    parser.add_argument("--hash-seed-workers", default="0:1,1:2,9876:8")
    return parser


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, value) -> None:
    path.write_text(
        yaml.safe_dump(canonical_value(value), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_dir_hash(root: Path) -> str:
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return canonical_hash(records)


def _tree_hash(root: Path) -> str:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return canonical_hash(records)


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def _validate_wheel(path: Path, expected_name: str, expected_version: str) -> dict:
    if _sha256(path) == "":
        raise ValueError("wheel hash 为空")
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        package_files = [
            name
            for name in archive.namelist()
            if name.startswith("gendiff_data_process/")
        ]
        if len(metadata_names) != 1 or not package_files:
            raise ValueError("wheel 缺少唯一 METADATA 或 package 文件")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, value)
    if fields.get("Name") != expected_name or fields.get("Version") != expected_version:
        raise ValueError(
            f"wheel identity 不匹配: {fields.get('Name')} {fields.get('Version')}"
        )
    return {
        "name": fields["Name"],
        "version": fields["Version"],
        "file_count": len(package_files),
    }


def _fingerprint_checks(
    repository: Path,
    run_manifest: dict,
    generation: dict,
    seed_workers: str,
) -> dict:
    selection = generation["selection"]
    config_path = Path(run_manifest["producer"]["config"])
    command_base = [
        sys.executable,
        str(repository / "tools/build_canonicalizer_pilot.py"),
        "--fingerprint-only",
        "--dataset-root",
        run_manifest["input"]["dataset_root"],
        "--config",
        str(config_path),
        "--building-start",
        str(selection["building_start"]),
        "--building-count",
        str(selection["building_count"]),
        "--stage-indices",
        ",".join(str(item) for item in selection["stage_indices"]),
    ]
    records = []
    for item in seed_workers.split(","):
        seed, workers = item.split(":", 1)
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        output = subprocess.check_output(
            [*command_base, "--workers", workers],
            cwd=repository,
            env=environment,
            text=True,
        )
        payload = json.loads(output)
        records.append(
            {
                "python_hash_seed": int(seed),
                "workers": int(workers),
                "fingerprint": payload["fingerprint"],
            }
        )
    mismatch_count = sum(
        record["fingerprint"] != generation["pilot_fingerprint"] for record in records
    )
    return {"runs": records, "mismatch_count": mismatch_count}


def _validate_packed(run_root: Path, bundle) -> dict:
    run_manifest = _load_yaml(run_root / "run.yaml")
    dataset_root = Path(run_manifest["outputs"]["dataset_root"]).resolve()
    expected_name = (
        "canonicalizer_pilot_bidirectional_v1"
        if bundle.validation_profile.mode == "bidirectional_monotonic"
        else "canonicalizer_pilot_v1"
    )
    if dataset_root != (run_root / "outputs" / expected_name).resolve():
        raise ValueError("packed dataset 路径与 task contract 不一致")
    profile = _load_yaml(run_root / "normalization_profile.yaml")
    meta = torch.load(dataset_root / "dataset_meta.pt", weights_only=False)
    validate_packed_release_meta(meta)
    task_contract_id = (
        "bidirectional_monotonic_v1"
        if bundle.validation_profile.mode == "bidirectional_monotonic"
        else "construction_only_v1"
    )
    expected_contract = {
        "task_contract_id": task_contract_id,
        "validation_mode": bundle.validation_profile.mode,
        "condition_surface_mode": bundle.condition_sampling.surface_mode,
        "canonicalizer_version": bundle.canonicalizer.canonicalizer_version,
        "geometry_version": bundle.canonicalizer.geometry_version,
        "geometry_config_hash": bundle.canonicalizer.geometry_config_hash,
        "canonicalizer_config_hash": bundle.canonicalizer.canonicalizer_config_hash,
        "validation_profile_hash": bundle.validation_profile.config_hash,
        "condition_config_hash": bundle.condition_sampling.config_hash,
        "normalization_config_hash": bundle.normalization.config_hash,
        "split_config_hash": bundle.split.config_hash,
        "package_config_hash": bundle.package.config_hash,
        "normalization_profile_id": profile["profile_id"],
    }
    contract = meta["canonical_contract"]
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"dataset contract 字段不匹配: {field}")
    if contract.get("producer_commit") != run_manifest["producer"]["commit"]:
        raise ValueError("dataset contract producer commit 不匹配")
    if (
        contract.get("package_wheel_sha256")
        != run_manifest["producer"]["package_wheel_sha256"]
    ):
        raise ValueError("dataset contract wheel SHA256 不匹配")
    if contract.get("normalization_profile_hash") != canonical_hash(profile):
        raise ValueError("dataset contract normalization profile hash 不匹配")
    if canonical_value(meta.get("producer")) != canonical_value(
        run_manifest["producer"]
    ):
        raise ValueError("dataset meta producer 与 run manifest 不匹配")
    if Path(meta.get("states_path", "")).resolve() != dataset_root / "states.pt":
        raise ValueError("dataset meta states_path 越界或不匹配")

    state_bank = torch.load(dataset_root / "states.pt", weights_only=False)
    if state_bank.get("schema_version") != "area_v2_packed_v1":
        raise ValueError("state bank schema 不匹配")
    states = state_bank.get("states") or []
    records = []
    split_uids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    pair_names: set[str] = set()
    counts: Counter[str] = Counter()
    change_kind_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        index = torch.load(dataset_root / f"{split}_index.pt", weights_only=False)
        for shard_record in index.get("shards") or []:
            shard_path = Path(shard_record["path"]).resolve()
            if (
                not shard_path.is_relative_to(dataset_root)
                or shard_record.get("split") != split
            ):
                raise ValueError(f"{split} shard 路径越界或 split 不匹配")
            shard = torch.load(shard_path, weights_only=False)
            if int(shard.get("sample_count", -1)) != len(shard.get("samples") or []):
                raise ValueError(f"{split} shard sample_count 不一致")
            for sample in shard.get("samples") or []:
                validate_packed_sample(sample)
                metadata = sample["canonical_metadata"]
                if (
                    bundle.validation_profile.mode == "bidirectional_monotonic"
                    and not all(
                        metadata.get(field) for field in ("change_kind", "pair_hash")
                    )
                ):
                    raise ValueError("双向 sample 缺少 change_kind 或 pair_hash")
                for field, expected in expected_contract.items():
                    if metadata.get(field) != expected:
                        raise ValueError(
                            f"sample canonical metadata 字段不匹配: {field}"
                        )
                if metadata["split"] != split:
                    raise ValueError("sample split metadata 与 index 不一致")
                if (
                    split_for_building_uid(metadata["building_uid"], bundle.split)
                    != split
                ):
                    raise ValueError("sample split 与冻结 building split 算法不一致")
                condition = sample["condition"]
                if tuple(condition.shape) != (bundle.condition_sampling.point_count, 3):
                    raise ValueError(
                        f"condition shape 不匹配: {tuple(condition.shape)}"
                    )
                if not torch.isfinite(condition).all():
                    raise ValueError("condition 包含 NaN/Inf")
                if sample["pair_name"] in pair_names:
                    raise ValueError("packed pair_name 重复")
                source_index = int(sample.get("source_state_index", -1))
                target_index = int(sample.get("target_state_index", -1))
                if not 0 <= source_index < len(states) or not 0 <= target_index < len(
                    states
                ):
                    raise ValueError("sample state index 越界")
                source_state = states[source_index]
                target_state = states[target_index]
                source_meta = source_state.get("meta") or {}
                target_meta = target_state.get("meta") or {}
                building_key = source_meta.get("building_key")
                if not building_key or target_meta.get("building_key") != building_key:
                    raise ValueError("sample source/target building 不一致")
                if building_uid_from_key(building_key) != metadata["building_uid"]:
                    raise ValueError("sample building UID 与 state building key 不一致")
                if source_meta.get("stage_hash") != metadata["source_stage_hash"]:
                    raise ValueError("sample source stage hash 与 state bank 不一致")
                if target_meta.get("stage_hash") != metadata["target_stage_hash"]:
                    raise ValueError("sample target stage hash 与 state bank 不一致")
                expected_pair_name = (
                    f"{building_key}_{source_state['state_name'].rsplit('/', 1)[-1]}"
                    f"_to_{target_state['state_name'].rsplit('/', 1)[-1]}"
                )
                if sample["pair_name"] != expected_pair_name:
                    raise ValueError("sample pair_name 与 state bank 不一致")
                pair_names.add(sample["pair_name"])
                split_uids[split].add(metadata["building_uid"])
                records.append(metadata)
                counts[split] += 1
                change_kind_counts[metadata["change_kind"]] += 1
    overlaps = {
        "train_val": len(split_uids["train"] & split_uids["val"]),
        "train_test": len(split_uids["train"] & split_uids["test"]),
        "val_test": len(split_uids["val"] & split_uids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"building split 泄漏: {overlaps}")
    collision = audit_supervision_collisions(records)
    observed_counts = {
        split: int(counts.get(split, 0)) for split in ("train", "val", "test")
    }
    expected_counts = {
        split: int(meta["split_sample_counts"].get(split, 0))
        for split in ("train", "val", "test")
    }
    if observed_counts != expected_counts:
        raise ValueError("packed sample count 与 dataset meta 不一致")
    return {
        "sample_counts": observed_counts,
        "state_count": len(states),
        "split_building_counts": {
            split: len(uids) for split, uids in split_uids.items()
        },
        "split_overlap": overlaps,
        "change_kind_counts": dict(change_kind_counts),
        "collision_report": canonical_value(collision),
    }


def _validate_real_loader(run_root: Path, gendiff_repo: Path, bundle) -> dict:
    sys.path.insert(0, str(gendiff_repo))
    from craftsman.data.packed_area_edit_v2_data_module import (
        PackedAreaEditV2DataModule,
    )

    run_manifest = _load_yaml(run_root / "run.yaml")
    dataset_root = Path(run_manifest["outputs"]["dataset_root"]).resolve()
    module = PackedAreaEditV2DataModule(
        {
            "dataset_folder": str(dataset_root),
            "max_layers": bundle.validation_profile.max_layers,
            "max_points_per_layer": bundle.validation_profile.max_points_per_layer,
            "strict_area_capacity": True,
            "replica": 1,
            "batch_size": 8,
            "num_workers": 0,
            "condition_point_num": bundle.condition_sampling.point_count,
            "train_iterate_shards": False,
            "shuffle_shards": False,
            "shuffle_samples_in_shard": False,
            "persistent_workers": False,
            "max_val_samples": 0,
            "max_test_samples": 0,
            "repeat_eval_to_world_size": False,
        }
    )
    module.setup(None)
    observed = {}
    for split, loader in (
        ("train", module.train_dataloader()),
        ("val", module.val_dataloader()),
        ("test", module.test_dataloader()),
    ):
        names = []
        for batch in loader:
            if tuple(batch["source_point_coords"].shape[1:]) != (
                bundle.validation_profile.max_layers,
                bundle.validation_profile.max_points_per_layer,
                2,
            ):
                raise ValueError("loader source_point_coords shape 不匹配")
            if tuple(batch["change_point_clouds"].shape[1:]) != (
                bundle.condition_sampling.point_count,
                3,
            ):
                raise ValueError("loader condition shape 不匹配")
            names.extend(batch["pair_name"])
        observed[split] = len(names)
    return {"sample_counts": observed, "num_workers": 0, "batch_size": 8}


def main() -> int:
    args = _parser().parse_args()
    run_root = Path(args.run_root).resolve()
    repository = Path(args.repository).resolve()
    gendiff_repo = Path(args.gendiff_repo).resolve()
    run_manifest = _load_yaml(run_root / "run.yaml")
    generation = _load_yaml(run_root / "reports" / "generation_report.yaml")
    bundle = load_bundle(run_manifest["producer"]["config"])
    checks: dict[str, object] = {}
    failures: list[str] = []

    try:
        actual_commit = _git(repository, "rev-parse", "HEAD")
        actual_remote = _git(repository, "remote", "get-url", "origin")
        status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
        if (
            actual_commit != run_manifest["producer"]["commit"]
            or actual_remote != run_manifest["producer"]["repository_remote"]
            or status
        ):
            raise ValueError(
                "validator 要求 producer commit/remote 一致且 worktree clean"
            )
        checks["git"] = {
            "commit": actual_commit,
            "remote": actual_remote,
            "clean": True,
        }
    except Exception as exc:
        failures.append(f"git: {exc}")

    try:
        config_artifact = Path(run_manifest["producer"]["config"])
        config_source = Path(run_manifest["producer"]["config_source"])
        expected_config_hash = run_manifest["producer"]["config_sha256"]
        if config_artifact != run_root / "config.yaml":
            raise ValueError("run manifest 未引用 artifact 内配置副本")
        if _sha256(config_artifact) != expected_config_hash:
            raise ValueError("artifact config SHA256 不匹配")
        if _sha256(config_source) != expected_config_hash:
            raise ValueError("producer source config SHA256 不匹配")
        actual_hashes = {
            "geometry_config_hash": bundle.canonicalizer.geometry_config_hash,
            "canonicalizer_config_hash": bundle.canonicalizer.canonicalizer_config_hash,
            "validation_profile_hash": bundle.validation_profile.config_hash,
            "condition_config_hash": bundle.condition_sampling.config_hash,
            "normalization_config_hash": bundle.normalization.config_hash,
            "split_config_hash": bundle.split.config_hash,
            "package_config_hash": bundle.package.config_hash,
        }
        if actual_hashes != run_manifest["config_hashes"]:
            raise ValueError("run config 子树 hash 不匹配")
        checks["config"] = {
            "sha256": expected_config_hash,
            "config_hashes": actual_hashes,
        }
    except Exception as exc:
        failures.append(f"config: {exc}")

    try:
        wheel = Path(run_manifest["producer"]["package_wheel"])
        if _sha256(wheel) != run_manifest["producer"]["package_wheel_sha256"]:
            raise ValueError("wheel SHA256 不匹配")
        checks["wheel"] = _validate_wheel(
            wheel,
            bundle.package.distribution_name,
            bundle.package.version,
        )
    except Exception as exc:
        failures.append(f"wheel: {exc}")

    try:
        source_manifest = _load_yaml(run_root / "source_manifest.yaml")
        actual_manifest_hash = canonical_hash(source_manifest["buildings"])
        if (
            actual_manifest_hash != source_manifest["manifest_hash"]
            or actual_manifest_hash != run_manifest["input"]["source_manifest_hash"]
        ):
            raise ValueError("source manifest hash 不匹配")
        checked = 0
        for building in source_manifest["buildings"]:
            for source in building["files"]:
                if _sha256(Path(source["path"])) != source["sha256"]:
                    raise ValueError(f"source SHA256 不匹配: {source['path']}")
                checked += 1
        checks["source_files"] = {"checked": checked, "mismatches": 0}
    except Exception as exc:
        failures.append(f"source_files: {exc}")

    try:
        actual = _canonical_dir_hash(run_root / "canonical")
        expected = run_manifest["outputs"]["canonical_output_hash"]
        if actual != expected:
            raise ValueError("canonical output hash 不匹配")
        checks["canonical_output_hash"] = actual
    except Exception as exc:
        failures.append(f"canonical_output: {exc}")

    try:
        packed_root = Path(run_manifest["outputs"]["dataset_root"])
        actual_packed_hash = _tree_hash(packed_root)
        if actual_packed_hash != run_manifest["outputs"]["packed_tree"]["tree_hash"]:
            raise ValueError("packed tree hash 不匹配")
        checks["packed_tree_hash"] = actual_packed_hash
    except Exception as exc:
        failures.append(f"packed_tree: {exc}")

    try:
        profile = _load_yaml(run_root / "normalization_profile.yaml")
        source_manifest = _load_yaml(run_root / "source_manifest.yaml")
        failed_keys = {
            item["building_key"] for item in generation["building_results"]["failures"]
        }
        expected_train_uids = sorted(
            building["building_uid"]
            for building in source_manifest["buildings"]
            if building["split"] == "train"
            and building["building_key"] not in failed_keys
        )
        if sorted(profile["train_building_uids"]) != expected_train_uids:
            raise ValueError(
                "normalization profile 含非 train building 或遗漏成功 train building"
            )
        checks["normalization_train_only"] = {
            "train_building_count": len(expected_train_uids),
            "profile_id": profile["profile_id"],
        }
    except Exception as exc:
        failures.append(f"normalization: {exc}")

    try:
        determinism_check = _fingerprint_checks(
            repository,
            run_manifest,
            generation,
            args.hash_seed_workers,
        )
        if determinism_check["mismatch_count"]:
            raise ValueError("不同 Python hash seed/worker 的 fingerprint 不一致")
        checks["determinism"] = determinism_check
    except Exception as exc:
        failures.append(f"determinism: {exc}")

    try:
        packed_check = _validate_packed(run_root, bundle)
        if (
            packed_check["change_kind_counts"]
            != generation["pair_results"]["change_kind_counts"]
        ):
            raise ValueError("packed change_kind 计数与 generation report 不一致")
        checks["packed"] = packed_check
    except Exception as exc:
        failures.append(f"packed: {exc}")

    try:
        checks["real_gendiff_loader"] = _validate_real_loader(
            run_root, gendiff_repo, bundle
        )
    except Exception as exc:
        failures.append(f"real_gendiff_loader: {type(exc).__name__}: {exc}")

    passed = not failures and generation["status"] == "pass"
    report = {
        "schema_version": "canonicalizer_pilot_validation_report_v1",
        "status": "pass" if passed else "fail",
        "generation_status": generation["status"],
        "checks": checks,
        "failures": failures,
        "training_run": False,
    }
    _write_yaml(run_root / "reports" / "validation_report.yaml", report)
    run_manifest["status"] = "pass" if passed else "fail"
    run_manifest["outputs"]["validation_report"] = str(
        run_root / "reports" / "validation_report.yaml"
    )
    _write_yaml(run_root / "run.yaml", run_manifest)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

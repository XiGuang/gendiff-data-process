#!/usr/bin/env python3
"""Build an observational catalog for legacy data/<category>/<dataset> trees.

The scanner does not move, rename, or modify any dataset files. It records only
facts visible from the filesystem and marks inferred provenance explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CATEGORY_METADATA = {
    "origin": {
        "status": "legacy",
        "lifecycle": "raw",
        "producer": "unknown",
        "notes": "原始源数据；本仓库内未找到足够证据确认其生成程序。",
    },
    "component": {
        "status": "unknown",
        "lifecycle": "intermediate",
        "producer": "legacy.component_processing_family",
        "notes": "可能由切分或合并脚本产生；具体命令和 commit 未保留。",
    },
    "block": {
        "status": "legacy",
        "lifecycle": "processed",
        "producer": "legacy.block_processing_family",
        "notes": "属于旧 block 处理链；具体 producer 仅能从命名和脚本用途推断。",
    },
    "condition": {
        "status": "legacy",
        "lifecycle": "processed",
        "producer": "legacy.condition_processing_family",
        "notes": "属于旧 condition/rotation/combination 链；缺少原始运行命令。",
    },
    "images": {
        "status": "unknown",
        "lifecycle": "derived",
        "producer": "external_or_unknown",
        "notes": "图像 producer 很可能跨仓库或依赖 Blender；当前不强行归属。",
    },
    "latents": {
        "status": "unknown",
        "lifecycle": "derived",
        "producer": "external_or_unknown",
        "notes": "latent producer 很可能在训练或编码仓库；当前仓库证据不足。",
    },
    "yaml": {
        "status": "legacy",
        "lifecycle": "manifest",
        "producer": "legacy.yaml_generation_family",
        "notes": "可能由 gen_yaml 脚本族产生；具体输入、命令和 commit 未保留。",
    },
    "test": {
        "status": "experiment",
        "lifecycle": "scratch",
        "producer": "unknown",
        "notes": "测试或人工检查数据；不得仅凭目录名删除。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan legacy datasets and write provenance-safe YAML manifests."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("catalog/datasets")
    )
    parser.add_argument("--cataloged-at", default=datetime.now().date().isoformat())
    return parser.parse_args()


def yaml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unnamed"


def utc_timestamp(value: float | None) -> str:
    if value is None:
        return "unknown"
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def scan_tree(path: Path) -> dict[str, object]:
    size_bytes = 0
    file_count = 0
    earliest_mtime: float | None = None
    latest_mtime: float | None = None
    extensions: Counter[str] = Counter()
    errors: list[str] = []

    for current_root, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current = Path(current_root)
        for filename in filenames:
            file_path = current / filename
            try:
                stat = file_path.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(f"{file_path}: {exc.__class__.__name__}")
                continue
            file_count += 1
            size_bytes += stat.st_size
            earliest_mtime = (
                stat.st_mtime
                if earliest_mtime is None
                else min(earliest_mtime, stat.st_mtime)
            )
            latest_mtime = (
                stat.st_mtime
                if latest_mtime is None
                else max(latest_mtime, stat.st_mtime)
            )
            suffix = file_path.suffix.lower() or "<no_extension>"
            extensions[suffix] += 1

    return {
        "size_bytes": size_bytes,
        "file_count": file_count,
        "earliest_mtime": utc_timestamp(earliest_mtime),
        "latest_mtime": utc_timestamp(latest_mtime),
        "extensions": dict(sorted(extensions.items())),
        "errors": errors[:20],
        "error_count": len(errors),
    }


def manifest_text(
    *,
    dataset_id: str,
    relative_path: Path,
    physical_path: Path,
    category: str,
    observed: dict[str, object],
    cataloged_at: str,
) -> str:
    metadata = CATEGORY_METADATA.get(
        category,
        {
            "status": "unknown",
            "lifecycle": "unknown",
            "producer": "unknown",
            "notes": "未识别的数据类别。",
        },
    )
    status = metadata["status"]
    producer = metadata["producer"]
    confidence = "low"
    notes = metadata["notes"]

    if category == "block" and "polygon_proxy" in relative_path.name.lower():
        status = "candidate"
        producer = "polygon_proxy.candidate_family"
        confidence = "medium"
        notes = "名称与 polygon proxy 候选链一致；仍缺原始 command/commit 证明。"

    lines = [
        "schema_version: dataset_catalog_v1",
        f"dataset_id: {yaml_string(dataset_id)}",
        f"status: {status}",
        f"lifecycle: {metadata['lifecycle']}",
        f"physical_path: {yaml_string(physical_path)}",
        f"legacy_relative_path: {yaml_string(relative_path.as_posix())}",
        f"cataloged_at: {yaml_string(cataloged_at)}",
        "immutable: unknown",
        "",
        "observed:",
        f"  size_bytes: {observed['size_bytes']}",
        f"  file_count: {observed['file_count']}",
        f"  earliest_mtime_utc: {yaml_string(observed['earliest_mtime'])}",
        f"  latest_mtime_utc: {yaml_string(observed['latest_mtime'])}",
        f"  scan_error_count: {observed['error_count']}",
        "  extensions:",
    ]
    extensions = observed["extensions"]
    if extensions:
        for extension, count in extensions.items():
            lines.append(f"    {yaml_string(extension)}: {count}")
    else:
        lines.append("    {}")
    if observed["errors"]:
        lines.append("  scan_errors:")
        for error in observed["errors"]:
            lines.append(f"    - {yaml_string(error)}")

    lines.extend(
        [
            "",
            "producer:",
            f"  script_id: {yaml_string(producer)}",
            "  script_path: unknown",
            "  git_commit: unknown",
            "  command: unknown",
            "  config: unknown",
            "  dirty_worktree: unknown",
            "",
            "inputs:",
            "  - unknown",
            "consumers:",
            "  - unknown",
            "validation:",
            "  status: not_cataloged",
            "  report: unknown",
            f"provenance_confidence: {confidence}",
            f"notes: {yaml_string(notes)}",
            "",
        ]
    )
    return "\n".join(lines)


def index_text(
    entries: list[dict[str, object]], cataloged_at: str, data_root: Path
) -> str:
    category_counts: defaultdict[str, int] = defaultdict(int)
    category_bytes: defaultdict[str, int] = defaultdict(int)
    total_bytes = 0
    total_files = 0
    for entry in entries:
        category = str(entry["category"])
        category_counts[category] += 1
        category_bytes[category] += int(entry["size_bytes"])
        total_bytes += int(entry["size_bytes"])
        total_files += int(entry["file_count"])

    lines = [
        "schema_version: dataset_index_v1",
        f"cataloged_at: {yaml_string(cataloged_at)}",
        f"data_root: {yaml_string(data_root)}",
        f"dataset_count: {len(entries)}",
        f"total_file_count: {total_files}",
        f"total_size_bytes: {total_bytes}",
        "categories:",
    ]
    for category in sorted(category_counts):
        lines.extend(
            [
                f"  {yaml_string(category)}:",
                f"    dataset_count: {category_counts[category]}",
                f"    size_bytes: {category_bytes[category]}",
            ]
        )
    lines.append("datasets:")
    for entry in entries:
        lines.extend(
            [
                f"  - dataset_id: {yaml_string(entry['dataset_id'])}",
                f"    manifest: {yaml_string(entry['manifest'])}",
                f"    legacy_path: {yaml_string(entry['relative_path'])}",
                f"    status: {entry['status']}",
                f"    size_bytes: {entry['size_bytes']}",
                f"    file_count: {entry['file_count']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, Path, Path]] = []
    for category_path in sorted(path for path in data_root.iterdir() if path.is_dir()):
        child_dirs = sorted(path for path in category_path.iterdir() if path.is_dir())
        for dataset_path in child_dirs:
            relative_path = dataset_path.relative_to(data_root)
            dataset_id = f"legacy__{slug(category_path.name)}__{slug(dataset_path.name)}"
            datasets.append((dataset_id, relative_path, dataset_path))

    entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for dataset_id, relative_path, dataset_path in datasets:
        if dataset_id in seen_ids:
            raise SystemExit(f"dataset_id collision: {dataset_id}")
        seen_ids.add(dataset_id)
        observed = scan_tree(dataset_path)
        category = relative_path.parts[0]
        metadata = CATEGORY_METADATA.get(category, {"status": "unknown"})
        status = metadata["status"]
        if category == "block" and "polygon_proxy" in relative_path.name.lower():
            status = "candidate"
        manifest_name = f"{dataset_id}.yaml"
        (output_dir / manifest_name).write_text(
            manifest_text(
                dataset_id=dataset_id,
                relative_path=relative_path,
                physical_path=dataset_path.resolve(),
                category=category,
                observed=observed,
                cataloged_at=args.cataloged_at,
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "dataset_id": dataset_id,
                "manifest": manifest_name,
                "relative_path": relative_path.as_posix(),
                "category": category,
                "status": status,
                "size_bytes": observed["size_bytes"],
                "file_count": observed["file_count"],
            }
        )
        print(
            f"{dataset_id}: {observed['file_count']} files, "
            f"{observed['size_bytes']} bytes"
        )

    (output_dir / "index.yaml").write_text(
        index_text(entries, args.cataloged_at, data_root), encoding="utf-8"
    )
    print(f"wrote {len(entries)} dataset manifests to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gendiff_data_process.canonicalization.config import load_bundle
from gendiff_data_process.canonicalization.pilot import (
    build_pilot,
    building_keys_from_range,
    make_pilot_tasks,
    pilot_fingerprint,
    process_pilot_tasks,
)


def _indices(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not output or len(output) != len(set(output)):
        raise argparse.ArgumentTypeError("stage/workers 列表不能为空或重复")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构造最多 100-building 的 canonicalizer pilot")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--building-start", type=int, default=1)
    parser.add_argument("--building-count", type=int, default=100)
    parser.add_argument("--stage-indices", type=_indices, default=(0, 1, 2, 3))
    parser.add_argument("--workers", type=int, default=1, help="fingerprint-only 使用的 worker 数")
    parser.add_argument("--fingerprint-only", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--repository", default=str(REPO_ROOT))
    parser.add_argument("--producer-commit")
    parser.add_argument("--package-wheel")
    parser.add_argument("--determinism-workers", type=_indices, default=(1, 2, 8))
    return parser


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def main() -> int:
    args = _parser().parse_args()
    bundle = load_bundle(args.config)
    building_keys = building_keys_from_range(args.building_start, args.building_count)
    tasks = make_pilot_tasks(
        args.dataset_root,
        args.config,
        building_keys,
        tuple(args.stage_indices),
        bundle,
    )
    if args.fingerprint_only:
        results = process_pilot_tasks(tasks, args.workers)
        errors = Counter(result.error_code for result in results if result.error_code)
        print(
            json.dumps(
                {
                    "fingerprint": pilot_fingerprint(results),
                    "workers": args.workers,
                    "building_count": len(results),
                    "successful_buildings": sum(result.sequence is not None for result in results),
                    "errors_by_code": dict(errors),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    required = {
        "--run-root": args.run_root,
        "--producer-commit": args.producer_commit,
        "--package-wheel": args.package_wheel,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"build 模式缺少参数: {', '.join(missing)}")
    repository = Path(args.repository).resolve()
    actual_commit = _git(repository, "rev-parse", "HEAD")
    if actual_commit != args.producer_commit:
        raise SystemExit(f"producer commit 不匹配: actual={actual_commit} expected={args.producer_commit}")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SystemExit("pilot 必须从 clean worktree 运行")

    manifest = build_pilot(
        run_root=args.run_root,
        dataset_root=args.dataset_root,
        config_path=args.config,
        building_start=args.building_start,
        building_count=args.building_count,
        stage_indices=tuple(args.stage_indices),
        determinism_workers=tuple(args.determinism_workers),
        repository_path=repository,
        repository_remote=_git(repository, "remote", "get-url", "origin"),
        producer_commit=args.producer_commit,
        package_wheel=args.package_wheel,
        command=tuple(sys.argv),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

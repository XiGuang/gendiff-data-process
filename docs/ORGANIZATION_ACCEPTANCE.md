# 项目整理验收

日期：2026-08-28

范围：仅验收仓库整理和 lineage 元数据。本文不批准 canonicalizer/adapter 实现、
数据生成、迁移、训练或修改 `/mnt/d/projects/GenDiff`。

## 审计状态

- 分支：`organize/finalize-project-organization-20260828`
- 已登记的代码 snapshot commit：`ddd7a68685042dc93331a105d0f8b7449c8e467a`
- 从最初 detached worktree 保留的 Phase 1 文件：
  `catalog/training_consumer_manifest.yaml`、
  `docs/TRAINING_CONSUMER_AUDIT.md` 和
  `docs/CANONICALIZER_TEST_PLAN.md`。
- 只通过目录元数据检查的 legacy 数据根目录：`/mnt/d/data/data`。
- 用于检测意外写入的 GenDiff 基线：commit
  `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`，dirty status SHA-256
  `6f46f8383742f8cda5a30916fcc2c909de56f32a69d19e9082402cd8c595ecb3`.

## 验收矩阵

| ID | 结果 | 验收项 | 证据 | 仍需人工确认 |
|---|---|---|---|---|
| ORG-01 | PASS | 工作隔离在指定分支，未覆盖已有 ref。 | 本地和 remote ref 的 `git show-ref --verify` 均失败；`git ls-remote --heads origin organize/finalize-project-organization-20260828` 返回空；分支从 `ddd7a686...` 创建。 | 无。 |
| ORG-02 | PASS | Git 只跟踪代码和元数据，不跟踪 legacy 数据、输出或 environment。 | `.gitignore`；`git ls-files | rg '^(data|output|outputs|pipeline/output|\.venv)(/|$)'` 未返回路径。 | 后续贡献者需确保源 config 不放入被 ignore 的 `/config/`。 |
| ORG-03 | PASS | 每个 tracked Python 文件都有一个用途和有效状态。 | `git ls-files '*.py'` = 67；解析 `catalog/code_inventory.yaml` 得到 67 个唯一条目；missing/extra/duplicate/blank/invalid-status 集合均为空。 | purpose 分类仍是可审阅判断；没有脚本被提升为 `current`。 |
| ORG-04 | PASS | 每个已观测 `/mnt/d/data/data/<category>/<dataset>` 目录都有 dataset ID、lifecycle 和 status。 | 有界两级 `Path.iterdir()` 找到 119 个目录；`catalog/datasets/index.yaml` 和 119 个链接 manifest 覆盖相同物理路径集合；必填字段缺口为 0。 | 有意不重新扫描目录内容或计算内容 hash。 |
| ORG-05 | PASS | Dataset index 和 manifest 在结构与聚合数值上完全一致。 | 解析 YAML 对比得到 119 行/文件，无 missing/unlisted ID；聚合为 5,672,789 个文件、3,587,926,590,672 bytes；分类数量和大小一致。 | 观测数量和大小仍是日期为 2026-08-28 的元数据 snapshot。 |
| ORG-06 | PASS | Candidate/current 数据记录有 producer、consumer 证据或显式 `unknown`。 | 唯一 candidate `data/` manifest 有显式 producer 字段和 `consumers: [unknown]`；`catalog/legacy_outputs.yaml` 中 9 个 `candidate_run` 条目均有 producer 和 `consumers: [unknown]`；没有 dataset 为 `current`。 | candidate run 的精确 command/config/commit/consumer 仍为 unknown。 |
| ORG-07 | FAIL | Legacy 和已观测训练数据集可由不可变 lineage 完整复现。 | `docs/UNRESOLVED_PROVENANCE.md`；`catalog/training_consumer_manifest.yaml` 记录历史 run commit/diff/environment 和 packed producer command/commit 为 unknown。 | 需要定位不可变 CityEngine source、generation/packing argv、commit/diff、environment、hash 和 validation report。 |
| ORG-08 | FAIL | Construction canonical candidate 与实际训练 consumer 兼容。 | `docs/TRAINING_CONSUMER_AUDIT.md`；`producer_compatibility.direct_loader_compatibility: blocked`；`catalog/training_consumer_manifest.yaml` 中有 12 项 mismatch。 | 实现前必须单独审阅并批准 adapter 语义和测试计划。 |
| ORG-09 | PASS | 实际 consumer、candidate construction pipeline 和 legacy pipeline 已分开，且没有错误提升。 | `docs/CURRENT_PIPELINE.md`；`catalog/pipelines/construction_sequence_v1.yaml`；`catalog/pipelines/history_area_edit_v2.yaml`；没有 pipeline status 为 `current`。 | 需审阅 `blocked_target` 分类；它不是 release 声明。 |
| ORG-10 | PASS | 入口导航可回答代码、数据、lineage、consumer 和状态问题，且不重复 catalog 内容。 | `README.md` 和 `catalog/README.md` 链接 code inventory、dataset index/manifest、legacy output、训练 consumer 证据、pipeline 决策、未解决登记和本报告。 | 无。 |
| ORG-11 | PASS | 所有修改完成后的最终有界验证通过。 | 128 个 catalog YAML 解析通过；manifest evidence 29/29、mismatch 12；code inventory 67/67；dataset catalog 119/119；67 个 Python 文件通过 `tokenize.open()` + `compile()` 且不写 pyc；5 个 synthetic test 通过；`uv lock --check`、Pandoc 渲染、Git 边界和 `git diff --check` 通过。 | real-data 和 canonicalizer test 不在范围内，不能从本项 PASS 推断其已通过。 |
| ORG-12 | PASS | 指定 code-only clone 位于 `/mnt/d/projects/gendiff-data-process`，匹配已推送分支、不含禁用目录且状态 clean。 | clone 前已确认目标不存在；fresh clone HEAD/upstream 均为 `e884896f99371b3ccfdd7d31c8034388efc23c55`；porcelain status 为空；不存在 `data`、`output`、`outputs`、`pipeline/output` 和 `.venv`；clone 验证得到 128 YAML、67 Python、119 dataset 行、5 个 synthetic test 通过且 lock 有效。 | `/mnt/d/data` 仍是 legacy compatibility root；未做迁移或 symlink 修改。 |

未解决 lineage 和兼容性对应的 FAIL 是有意保留的。即使这些技术 gate 仍为 blocked，
项目整理本身仍可验收，但任何 pipeline 或 dataset 都不能提升为 `current`。

## 有界验证命令

以下检查只读取 source/catalog 元数据，不生成数据：

```bash
git status --short --branch
git diff --check
git ls-files | rg '^(data|output|outputs|pipeline/output|\.venv)(/|$)'
python3 -c '<用 yaml.safe_load 解析每个 catalog YAML>'
python3 -c '<比较 tracked Python 路径与 code_inventory 条目>'
python3 -c '<比较 dataset index/manifest 与两级目录元数据>'
python3 -c '<用 tokenize.open 解码并 compile 每个 tracked Python source，不写 pyc>'
python3 -m unittest tests.test_polygon_proxy.PolygonProxySyntheticTests
uv lock --check
pandoc --from=gfm --to=html <modified-doc> -o /dev/null
```

精确 inline Python 断言及其数量记录在最终交接命令摘要中。real-data test、全量
hash/scan、生成、依赖安装和训练均不在范围内。

## 精确下一门槛

记录并推送本验收结果，将 fresh clone fast-forward 到记录该结果的 commit，再次确认
clean status 和禁用路径不存在，随后停止。Canonicalizer、adapter、生成和训练需要新的
已审阅任务，并继续受 ORG-07/ORG-08 阻塞。

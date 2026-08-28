# 项目整理执行报告（2026-08-28）

## 执行结果

legacy 数据仓库已在不移动或删除任何数据集的前提下完成稳定化。此前未提交的 source
已备份、按功能组织为 commit、推送到隔离分支，并通过 fresh clone 验证。代码、数据集、
生成 index、输出和未解决 provenance 现已有独立 catalog。

## Git 分支与提交

分支：`organize/catalog-and-code-snapshot-20260828`

```text
2259890 docs: add code and dataset lineage catalog
e83c6f6 chore: preserve dependency snapshots
7dbed97 checkpoint: preserve exploratory diagnostic script
258db17 checkpoint: preserve obj-to-language changes
b4c15a8 feat(data): preserve construction sequence generators
3ebbb12 feat(data): preserve polygon proxy pipeline
```

code checkpoint 和 catalog commit 完成后，该分支已推送到 `origin`。

## 仓库外备份

备份根目录：

`/mnt/d/artifacts/gendiff-data-process/snapshots/20260828_pre_organization`

```text
tracked_worktree.patch
  sha256 90b2b04d5aa66eb7f4e3bce36f7bfbe99b2dc13e70659e03806f79da13e9c271

status.txt
  sha256 c5acfd2f4e2402a4975da47211f7280051aa741d55030ab005e576b0cb396231

source_snapshot.tar.gz
  sha256 89074dec873302a903a64f571a8b0f5e7654e50de1f2390c8da26d017f91df94
  gzip 验证通过；归档 25 个 source 条目

newly_revealed_yaml.tar.gz
  sha256 6bb4ab3392447692eeaebf3c3cce2a744b54ca58c0deeafb9e022ceb465a76bc
  gzip 验证通过
```

第二个归档包含移除全局 `*.yaml` ignore 后显现的历史生成 YAML index。

## 目录册结果

- 已登记 119 个 `data/<category>/<dataset>` 目录。
- 已观测 5,672,789 个文件。
- 已观测 3,587,926,590,672 bytes，十进制约 3.59 TB。
- 已生成 119 个单数据集 manifest 和一个聚合 dataset index。
- 每个 tracked Python 路径在 `catalog/code_inventory.yaml` 中恰有一个条目。
- 四份机器可读 pipeline 描述区分 candidate、legacy 和 unresolved target 链路。
- 历史 `output/`、`outputs/` 和 `pipeline/output/` 根目录已登记，但保持原样且继续被
  ignore。
- 九个生成 YAML pair index 已按大小、行数、schema 和 SHA-256 登记。
  `data.yaml` 与 `config/yuehai_with_remove.yaml` 字节完全相同。

全部 127 个 catalog YAML 文件解析成功。

## 验证

全新克隆：

`/mnt/d/projects/gendiff-data-process-clean-20260828`

在 commit `22598906871384ad00a46e44492ec2ddbfa36a09` 上验证：

- local HEAD 等于已推送 remote branch；
- worktree clean；
- 不存在 `data/` 或 `output/` 树；
- fresh `.git` 约 568 KiB，checkout 约 2.4 MiB；
- 所有选定 Python 文件 compile 通过；
- 五个 polygon proxy synthetic test 通过；
- `uv lock --check` 通过；
- 所有 catalog YAML 文件解析通过；
- dataset index 数量等于 119 个 manifest；
- code inventory 没有 missing 或 extra Python 路径。

## 有意保持不变的内容

- 没有移动、重命名、编辑或删除任何数据集或历史输出。
- 旧 `.git` 目录中约 17.09 GiB 的临时 pack 未被移除。
- 没有重构算法实现。
- 没有在缺少下游证据时将任何 pipeline 提升为 `current`。

## 后续最高优先级工作

1. 将精确 GenDiff 训练仓库 commit、loader、config 和 schema 与 candidate canonical
   数据集关联。
2. 在生成更多训练数据前增加 canonicalizer invariance、determinism、collision、
   ambiguity 和 round-trip test。
3. 对齐 `pyproject.toml`/`uv.lock` 与历史 Conda/Pip snapshot，建立可运行的数据生成环境。
4. 增加 run-manifest writer，记录 dataset ID、commit、clean/dirty 状态、command、
   config、seed、environment、hash 和 validation report。
5. 执行一次小型冻结端到端 run 和 training-loader smoke/overfit test。
6. 只有上述 gate 通过后，才把小型 candidate output 迁移到仓库外 artifact 布局；
   最后处理旧临时 Git pack。

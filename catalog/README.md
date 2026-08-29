# 数据与代码目录

本目录是在不移动 TB 级内容的前提下理解 legacy `data/` 仓库的事实来源。

## 记录内容

- `code_inventory.yaml`：为每个 tracked Python 文件记录唯一主状态和用途。
- `datasets/index.yaml`：汇总大小和数量，并链接 119 个已观测的
  `data/<category>/<dataset>` manifest。
- `datasets/*.yaml`：物理路径、文件数、字节数、扩展名、mtime 范围、生命周期、
  验证状态和 provenance 置信度。
- `pipelines/*.yaml`：candidate 与 legacy 生产 DAG。
- `training_consumer_manifest.yaml`：实际 GenDiff checkout、run
  command/config、packed loader 合同、producer 兼容性及未解决字段的 Phase 1
  只读证据。
- `canonicalizer_phase2_test_report.yaml`：2A 到 2E 的配置 hash、运行环境、定向测试、
  仓库校验和未解除阻塞项。
- `canonicalizer_bidirectional_test_report.yaml`：施工/拆除双向 candidate 的配置 hash、
  100-building 有界分类、定向测试与未执行 gate。
- `legacy_outputs.yaml`：仍位于 Git 外的 `output/`、`outputs/` 和
  `pipeline/output/` 生成产物目录。
- `legacy_yaml_indexes.yaml`：此前被全局 `*.yaml` ignore 隐藏的大型生成 YAML
  pair index 的 hash 和行数。
- 查看器本身不另建 catalog；其 Python 入口与 packed 读取层登记在
  `code_inventory.yaml`，迁移和兼容证据见 `docs/area_edit_v2_viewer_migration.md`。

## 置信度规则

`high` 要求有已记录的 command/config 和不可变 producer commit。
`medium` 表示代码默认值、名称或时间戳强烈暗示某种关系。
`low` 表示只知道脚本家族或类别级关系。证据不足的字段保持 `unknown`，不得猜填。

## 查询顺序

1. 在 `code_inventory.yaml` 中查找脚本路径，获取用途和状态。
2. 在 `datasets/index.yaml` 中查找 legacy 相对路径或 dataset ID，再打开链接的
   manifest 查看 lifecycle、producer、consumers 和 validation。
3. 查询 `output/`、`outputs/` 或 `pipeline/output/` 时使用
   `legacy_outputs.yaml`；candidate run 的 consumer 即使未知也会显式写出。
4. 查询外部 GenDiff 训练数据集和模型 schema 时使用
   `training_consumer_manifest.yaml`，并沿 evidence ID 取证。不得根据目录名或脚本名
   相似推断关系。
5. `pipelines/*.yaml` 表示 DAG 意图，`docs/CURRENT_PIPELINE.md` 表示当前决策。
   两者都不能在缺少 consumer 证据时提升数据集状态。
6. 查看某个 raw/packed pair 时，从 `viewer/README.md` 进入；查看器只读并校验 schema，
   但视觉可读性不能替代生成、collision、split 或训练 consumer gate。

## 更新数据目录

在仓库根目录运行：

```bash
python tools/build_data_catalog.py \
  --data-root data \
  --output-dir catalog/datasets \
  --cataloged-at YYYY-MM-DD
```

扫描器只读取元数据，不移动、重命名、编辑或删除数据文件。提交前必须审阅生成的 diff。

## 状态规则

- `current`：已有下游证据确认的生产路径。
- `candidate`：仍需端到端验证的近期路径。
- `legacy`：用于解释已有历史数据集的路径。
- `support`：检查、转换、复制、报告或 catalog 工具。
- `experiment`：smoke test 或一次性诊断。
- `deprecated`：已有替代路径，但在迁移验证前保留。
- `unknown`：证据不足。

当前没有任何数据生成 pipeline 被提升为 `current`。canonical construction
pipeline 仍是 candidate，与已观测 packed GenDiff loader 的直接兼容性为 blocked。

Canonicalizer Phase 2 的 candidate 实现边界见
`docs/CANONICALIZER_PHASE2_DECISIONS.md`，测试状态见
`docs/CANONICALIZER_TEST_PLAN.md`，验收证据见
`docs/CANONICALIZER_PHASE2_REPORT.md`；pilot 的冻结合同见
`docs/CANONICALIZER_PILOT_CONTRACT.md`，双向扩展见
`docs/CANONICALIZER_BIDIRECTIONAL_CONTRACT.md`。小型 loader smoke、fingerprint 或
partial pilot 通过不等于 pipeline 或 dataset 已提升为 `current`。

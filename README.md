# gendiff-data-process

DiffGen 的数据生成、转换、验证和目录工具。
本仓库只保存代码和小型元数据。TB 级数据集与生成产物保留在 Git 之外，
通过 manifest 引用。

## 项目导航

| 问题 | 事实来源 |
|---|---|
| 代码在哪里，每个脚本做什么？ | `catalog/code_inventory.yaml` 为每个 tracked Python 路径记录唯一的 `purpose` 和 `status`。 |
| 历史数据在哪里，生命周期是什么？ | `catalog/datasets/index.yaml` 将每个已观测的 `/mnt/d/data/data/<category>/<dataset>` 目录链接到 `catalog/datasets/` 下的 manifest。 |
| 数据或产物由谁产生、被谁消费？ | 查看单数据集的 `producer`/`consumers` 字段，以及 `catalog/legacy_outputs.yaml` 和 `catalog/training_consumer_manifest.yaml`；证据不足时明确写 `unknown`。 |
| 哪些 pipeline 是 legacy、candidate 或 blocked？ | 查看 `docs/CURRENT_PIPELINE.md` 和 `catalog/pipelines/`。 |
| 哪些结论仍无法复现？ | 查看 `docs/UNRESOLVED_PROVENANCE.md`。 |
| Canonicalizer Phase 2 如何实现？ | 查看 `docs/CANONICALIZER_PHASE2_DECISIONS.md` 和 `configs/canonicalizer_v1.yaml`。 |
| Canonicalizer Phase 2 验收结果是什么？ | 查看 `docs/CANONICALIZER_PHASE2_REPORT.md` 和 `catalog/canonicalizer_phase2_test_report.yaml`。 |
| 历史 construction-only pilot 使用什么合同？ | 查看 `docs/CANONICALIZER_PILOT_CONTRACT.md`。 |
| 新建与拆除如何统一生成？ | 查看 `docs/CANONICALIZER_BIDIRECTIONAL_CONTRACT.md`、`configs/canonicalizer_bidirectional_v1.yaml` 和 `catalog/canonicalizer_bidirectional_test_report.yaml`。 |
| 如何查看 raw 或 packed 区域编辑数据？ | 查看 `viewer/README.md` 和 `docs/area_edit_v2_viewer_migration.md`；运行入口在 `viewer/`，只读 Python helper 在 `tools/`。 |
| 项目整理是否通过验收？ | 查看 `docs/ORGANIZATION_ACCEPTANCE.md`。 |

字段含义和查询顺序见 `catalog/README.md`，代码、数据与产物边界见
`docs/DATA_LAYOUT.md`，历史整理执行细节见
`docs/ORGANIZATION_REPORT_20260828.md`。

## 处理链路状态

本仓库没有任何数据生成 pipeline 被标为 `current`。实际观测到的 GenDiff
训练 consumer 已有文档证据，但它消费的是打包后的 `area_v2_packed_v1` PT
数据，而不是本仓库的 loose canonical candidate 输出。直接兼容性仍为 blocked：

- `catalog/training_consumer_manifest.yaml`：commit/config/command、loader
  合同、路径、证据 ID、mismatch 和显式 unknown。
- `docs/TRAINING_CONSUMER_AUDIT.md`：供人工审阅的完整证据链。
- `docs/CANONICALIZER_TEST_PLAN.md`：定义 Phase 2 的测试 gate；2A 到 2E 的 candidate
  验收结果见 `docs/CANONICALIZER_PHASE2_REPORT.md`。
- `docs/CANONICALIZER_BIDIRECTIONAL_CONTRACT.md`：定义纯施工/纯拆除双向 pair、方向化
  condition、mixed 失败关闭和下一轮 100-building gate。

## 数据查看器

GenDiff 原 `construction_edit_animation_viewer` 已迁入本仓库：

```text
viewer/                                      # React/Three.js 前端
tools/dataset_browser_api.py                 # raw/packed 摘要、分页与 condition
tools/export_edit_animation_viewer_data.py   # raw/packed pair 转查看格式
gendiff_data_process/viewer_packed.py         # packed schema、index、shard 定位
tests/viewer/                                 # 双向 packed 查看器测试
```

查看器支持 legacy raw area-v2 和 candidate `area_v2_packed_v1`，并显示
`construction`/`demolition`、`pair_hash`、`INSERT_LAYER` 与 `DELETE_LAYER`。它是只读
诊断工具；查看成功不等于数据生成或训练 gate 已通过。

## 候选处理链路入口

```text
polygon_proxy/core.py + tools/build_polygon_proxy*.py
building_process/batch_polygon_proxy_flat.py
building_process/generate_construction_sequence.py
building_process/generate_construction_sequence_canonical_edit_dataset.py
```

仓库中存在两套 polygon proxy 实现。组合其输出前必须先阅读
`docs/CURRENT_PIPELINE.md`。

## 数据与产物边界

- `/mnt/d/data` 保持为 legacy compatibility root，本轮整理不移动该目录。
- `/mnt/d/data/data`、`/mnt/d/data/output`、`/mnt/d/data/outputs` 和
  `/mnt/d/data/pipeline/output` 保留在 Git 之外，仅原地登记。
- code-only clone 不得包含 `data/`、`output/`、`outputs/`、
  `pipeline/output/` 或 `.venv/`。

## 刷新观测型数据目录

```bash
python tools/build_data_catalog.py \
  --data-root data \
  --output-dir catalog/datasets \
  --cataloged-at YYYY-MM-DD
```

扫描器对数据集只读：仅记录文件系统元数据，并只写入 catalog manifest。

## 保留相对路径复制文件

脚本：`copy_by_relative_path.py`

从输入目录复制指定文件类型到输出目录，同时保持原相对目录结构。

```bash
python copy_by_relative_path.py \
  --input data/condition/yuehai_building_and_ground_combinations \
  --output /tmp/filtered \
  --ext .ply .npz
```

常用参数：

- `--ext`：需要包含的扩展名，不区分大小写；留空表示全部文件。
- `--overwrite`：覆盖输出目录中的已有文件。
- `--dry-run`：只打印统计，不复制文件。

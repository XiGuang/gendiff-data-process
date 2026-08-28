# gendiff-data-process 整理与数据 Lineage 方案

- 日期：2026-08-28
- 服务器代码仓库：`/mnt/d/data`
- Git 远端：`git@github.com:XiGuang/gendiff-data-process.git`
- 原则：**本轮不重构算法、不删除历史数据、不批量移动脚本。先保存代码，再建立目录册和生产关系。**

## 1. 当前问题

### 1.1 Git 状态

- 当前分支 `master` 与 `origin/master` 同步，远端最后提交是 2026-04-02。
- 仓库只有 56 个 tracked 文件。
- 有 4 个已修改 tracked 文件：
  - `README.md`
  - `building_process/batch_polygon_proxy_flat.py`
  - `obj_to_language/yaml_to_obj_new.py`
  - `uv.lock`
- 有 18 个顶层 untracked 条目，其中包含 2026-03 至 2026-05 的主要新代码、测试和工具。
- `.git` 约 18 GB，其中约 17.09 GiB 是未完成的临时 pack：`.git/objects/pack/tmp_pack_14JGZ5`。它不是正常 commit 历史，但在保存代码前不应直接删除。
- `.gitignore` 使用全局 `*.yaml`，导致新的配置、dataset manifest 和 pipeline manifest 默认无法进入 Git。

### 1.2 代码和数据混放

| 路径 | 规模 | 初步含义 |
|---|---:|---|
| `data/` | 3.3 TB | 历史原始数据、中间数据、训练数据混合 |
| `data/block/` | 1.3 TB | 建筑/区块处理结果，多套版本混合 |
| `data/images/` | 787 GB | 图像或渲染结果，具体 producer 尚未完全确认 |
| `data/latents/` | 729 GB | 模型 latent 数据，producer 可能跨仓库 |
| `data/condition/` | 411 GB | 变化条件点云及旋转/噪声版本 |
| `data/component/` | 141 GB | 建筑、地面、变化组件和 mesh 中间结果 |
| `data/origin/` | 1.6 GB | 名称上像原始数据，但尚未完成来源确认 |
| `data/test/` | 1.6 GB | polygon proxy 等实验/测试产物 |
| `data/yaml/` | 433 MB | 数据列表或训练描述 YAML |
| `pipeline/output/` | 5.6 GB | 旧 pipeline 的 global/block 输出 |
| `output/` | 38 MB | smoke、polygon proxy、construction 实验产物 |
| `outputs/` | 3.9 MB | 2026-05 canonical edit 实验产物 |
| `.venv/` | 1.9 GB | 本地环境，不应属于代码仓库 |

## 2. 整理后的目标

需要能快速回答四个问题：

1. 这个脚本做什么，目前还在不在用？
2. 这个数据目录由哪个脚本、哪个 commit、什么参数产生？
3. 这个数据又被哪个训练/处理流程消费？
4. 哪些内容是当前链路，哪些是历史、实验、未知或可清理产物？

目标不是一次性把所有代码移动到漂亮的新目录，而是建立一个受 Git 管理的 **catalog**，把现有路径映射成清晰的生产 DAG。

## 3. 三类内容必须分开

### 3.1 Code repo

建议最终使用：

```text
/mnt/d/projects/gendiff-data-process/
```

只包含：

- Python/Blender 脚本。
- `pyproject.toml`、lock file 和环境说明。
- 配置。
- tests 和小型 fixtures。
- 代码目录册、pipeline 文档、dataset manifests。
- 不包含真实 OBJ/PT/PLY/图片和批量运行输出。

### 3.2 Dataset root

建议最终使用：

```text
/mnt/d/datasets/gendiff/
  raw/
  intermediate/
  processed/
  training/
  legacy/
  unknown/
```

数据目录按“数据生命周期”组织，而不是按某个临时脚本名组织：

- `raw`：不可覆盖的外部原始数据。
- `intermediate`：可以由 raw 重建的中间结果。
- `processed`：有明确 schema、可供下游使用的数据。
- `training`：冻结的数据集 release，必须有 split 和 manifest。
- `legacy`：已知历史用途但当前不使用。
- `unknown`：尚未确认来源；只登记，不猜测、不删除。

### 3.3 Artifact root

建议最终使用：

```text
/mnt/d/artifacts/gendiff-data-process/
  runs/
  scratch/
  reports/
  legacy-output/
```

- `runs`：正式运行记录，每次运行一个 `run_id`。
- `scratch`：smoke、tmp、人工查看等可再生结果。
- `reports`：统计、可视化、审计报告。
- `legacy-output`：当前 `output/`、`outputs/`、`pipeline/output/` 的冻结备份。

## 4. 不立即移动代码：先加 Catalog

建议在代码仓库新增以下文件，不改变现有脚本位置：

```text
catalog/
  README.md
  code_inventory.yaml
  datasets/
    <dataset_id>.yaml
  pipelines/
    legacy_block_condition_v1.yaml
    polygon_proxy_v1.yaml
    construction_sequence_v1.yaml
    history_area_edit_v2.yaml

docs/
  CURRENT_PIPELINE.md
  LEGACY_PIPELINES.md
  DATA_LAYOUT.md
  UNRESOLVED_PROVENANCE.md
```

### 4.1 Script 状态枚举

每个脚本只能有一个主状态：

| 状态 | 含义 |
|---|---|
| `current` | 已确认属于当前生产链路 |
| `candidate` | 最近开发、可能接替旧链路，但尚未完成下游验证 |
| `legacy` | 历史流程仍需保留，但当前不使用 |
| `support` | 通用检查、转换、复制、统计工具 |
| `experiment` | smoke、test、一次性探索脚本 |
| `unknown` | 暂时无法确认用途或 producer/consumer |
| `deprecated` | 有明确替代者；保留到迁移验收后再归档 |

不能因为文件较新就直接标为 `current`。`current` 必须有下游 config、训练数据或正式运行记录作为证据。

### 4.2 `code_inventory.yaml` 条目

```yaml
- script_id: construction.canonical_edit_dataset.v1
  path: building_process/generate_construction_sequence_canonical_edit_dataset.py
  git_status: untracked
  status: candidate
  purpose: 从 construction sequence 生成 canonical edit 数据集
  inputs:
    - schema: construction_sequence
      path_pattern: <sequence_root>/building*/sequence*/stage_*
  outputs:
    - schema: canonical_edit_dataset
      files:
        - train.yaml
        - val.yaml
        - test.yaml
        - dataset_meta.yaml
        - edit_objects/
        - edit_sequences_v2/
        - conditions/
  known_output_paths:
    - /mnt/d/data/outputs/canonical
    - /mnt/d/data/outputs/canonical_obj
  consumers:
    - unknown
  replacement_for:
    - building_process/build_layer_edit_dataset_from_sequence.py
  last_verified: null
  notes: 当前尚未提交；是否用于 GenDiff 当前 area 数据仍需确认
```

### 4.3 Dataset manifest 条目

一批数据一个稳定 `dataset_id`，不要只靠路径名表达版本：

```yaml
schema_version: dataset_catalog_v1
dataset_id: yingrenshi_polygon_proxy_flat__v20260402
status: candidate
lifecycle: processed
physical_path: /mnt/d/data/output/yingrenshi_polygon_proxy_components_flat_new
size_bytes: null
created_at: 2026-04-02
immutable: true

producer:
  script_id: polygon_proxy.batch_flat.v1
  script_path: building_process/batch_polygon_proxy_flat.py
  git_commit: null
  dirty_worktree: true
  command: unknown
  config: unknown

inputs:
  - dataset_id: yingrenshi_change_components__unknown_version
    path: /mnt/d/data/data/component/yingrenshi_change/building

outputs:
  schemas:
    - polygon_proxy_yaml
    - polygon_proxy_obj

consumers:
  - pipeline_id: construction_sequence_v1

validation:
  status: not_run
  report: null

provenance_confidence: medium
notes: producer 由脚本默认路径和目录时间推断，尚缺原始 command
```

对于无法确认的字段必须写 `unknown`，不能为了让文档完整而猜测。

## 5. 当前代码的初步分类

### 5.1 最近的 polygon/construction 候选链路

| 路径 | 初步作用 | 状态建议 |
|---|---|---|
| `polygon_proxy/core.py` | 较新的 polygon proxy 核心实现 | `candidate`，当前 untracked |
| `tools/build_polygon_proxy.py` | 批量构造 polygon proxy | `candidate`，untracked |
| `tools/build_polygon_proxy_components.py` | 从 component OBJ 构造 proxy | `candidate`，untracked |
| `tools/check_polygon_proxy_samples.py` | polygon proxy 抽样检查 | `support`，untracked |
| `building_process/batch_polygon_proxy_flat.py` | 将 component proxy 扁平输出 | `candidate`，tracked 但有修改 |
| `building_process/generate_construction_proxy.py` | 从 proxy 产生单个 construction stage | `candidate`，untracked |
| `building_process/generate_construction_sequence.py` | 产生多阶段 construction sequence | `candidate`，untracked |
| `building_process/generate_construction_sequence_canonical_edit_dataset.py` | sequence + canonical edit dataset | `candidate`，2026-05 最新、untracked |
| `building_process/cut_dense_mesh_by_construction_sequence.py` | 按 construction stage 切原始 dense mesh | `candidate/support`，untracked |
| `building_process/build_layer_edit_dataset_from_sequence.py` | 较早的 layer edit dataset 构造 | `deprecated candidate`，需与 canonical 版本比较 |

已知实验输出大致对应：

```text
polygon proxy tools
  -> output/polygon_proxy*
  -> output/*polygon_proxy_components*

generate_construction_proxy.py
  -> output/construction_proxy/

generate_construction_sequence.py
  -> output/construction_sequence/

generate_construction_sequence_canonical_edit_dataset.py
  -> outputs/canonical/
  -> outputs/canonical_obj/
```

这些是“代码内默认路径 + 文件时间”得到的初步映射，正式 catalog 中应标记为 medium confidence，直到找到实际命令或运行日志。

### 5.2 `obj_to_language`

| 路径 | 作用 | 状态建议 |
|---|---|---|
| `obj_to_yaml_new.py` | OBJ 连通分量转 footprint/height YAML | `legacy/experiment` |
| `batch_obj_yaml_pipeline.py` | 批量 OBJ -> YAML -> 重建 OBJ | `legacy/experiment` |
| `rotate_yaml_bottom_contours.py` | 旋转 YAML footprint | `support` |
| `report_large_footprints.py` | 统计异常大 footprint | `support` |
| `yaml_to_obj_new.py` | YAML 重建 OBJ | `support`，tracked 但有修改 |

它的 schema 与当前 GenDiff CityEngine stage YAML 不一致，因此目前不能标为当前 `history_stages_*` 的确定 producer。

### 5.3 2025 年 block/condition/latent 旧链路

建议整体标为 `legacy`，保留 producer 关系但不与新 polygon/construction 链路混合：

- `process_change/`
- `process_combination/`
- `dora_preprocess/`
- `gen_yaml/`
- `pipeline/select_t1_and_t2.py`
- `pipeline/segment_and_normalize_point_cloud.py`
- 根目录的 `cut_building*`、`generate_t2.py`、`get_origin.py`、`merge_building.py`

初步数据关系：

```text
component/origin
  -> cut/merge/process_change
  -> data/block/*
  -> process_combination / rotate
  -> data/condition/*
  -> 外部编码或训练过程
  -> data/latents/*

block + condition + image/latent
  -> gen_yaml/*
  -> data/yaml/*
```

`data/images` 和 `data/latents` 的 producer 很可能跨越其他仓库或 Blender/模型流程。没有证据前应标为 `unknown producer`，不要强行归到本仓库某个脚本。

### 5.4 Support / experiment

- 文件检查：`check_obj_threshold.py`、`floating_poly_detect.py`、`list_obj_bboxes.py`。
- 数据搬运/统计：`copy_by_relative_path.py`、`count_obj_heights.py`、`stat_chunk_num.py`。
- 一次性脚本：根目录 `test.py`、`obj_to_language/test.py`、`watertight.py`。
- smoke 和 fixtures：`building_process/polygon_proxy_smoke_test.py`、`tests/test_polygon_proxy.py`。

一次性脚本不一定需要删除，但应标明输入路径、最后用途和是否仍可运行。

## 6. Git 止血方案

在移动任何数据前先保存代码。

### 6.1 禁止事项

- 不执行 `git add .`。
- 不把 `data/`、`output/`、`outputs/`、`.venv/` 或 `pipeline/output/` 加入 Git。
- 不在未备份 dirty worktree 前运行 `git clean`、`git gc` 或删除临时 pack。
- 不把全部 3.3 TB 数据放进 Git LFS；当前需求用 catalog 即可。

### 6.2 建议提交顺序

新建分支：

```text
organize/catalog-and-code-snapshot-20260828
```

按功能做代码快照，而不是一个巨大提交：

1. `checkpoint: preserve polygon proxy core, tools, and tests`
2. `checkpoint: preserve construction sequence and edit dataset tools`
3. `checkpoint: preserve obj-to-language changes`
4. `chore: update dependencies and lock file`
5. `docs: add code and dataset lineage catalog`

每个提交只 `git add` 明确文件列表。`requirements*.txt` 和 `uv.lock` 应单独确认来源，避免无意混入整个环境变更。

### 6.3 `.gitignore` 修正原则

使用目录边界忽略数据，不再全局忽略 YAML：

```gitignore
__pycache__/
*.py[cod]
.venv/
.vscode/

/data/
/output/
/outputs/
/pipeline/output/

*.obj
*.mtl
*.ply
*.pt
*.npz
*.png
*.jpg
```

删除全局 `*.yaml`，使以下内容能正常提交：

- `config/**/*.yaml`
- `catalog/**/*.yaml`
- tests fixtures YAML
- 小型 schema 示例

### 6.4 干净新 clone

代码快照推送后，建议在 `/mnt/d/projects/gendiff-data-process` 做一次 fresh clone。这样可以：

- 与 3.3 TB 数据彻底分开。
- 避开旧 `.git` 中 17.09 GiB 临时 pack。
- 验证所有需要的代码确实已经 push。

只有 fresh clone 中代码、tests、catalog 均完整后，旧 `/mnt/d/data` 才能降级为“数据兼容根目录”。

## 7. 数据整理顺序

### 阶段 A：原地登记，不移动

1. 扫描 `data/` 的二级目录，记录文件数、总字节、扩展名、最早/最新 mtime。
2. 每个二级目录分配 `dataset_id`。
3. 标记 `current/candidate/legacy/unknown/scratch`。
4. 尽量反查 producer：脚本默认路径、shell history、日志、周报、GenDiff config、mtime。
5. 生成 pipeline DAG；未知关系保留为 unknown。

这一阶段不会触碰 3.3 TB 文件内容，只生成轻量 catalog。

### 阶段 B：冻结明确的 release

对确认仍在使用的数据：

- 记录输入 dataset IDs、producer commit、完整 command/config。
- 标记 immutable。
- 记录文件数量、总大小和快速 inventory hash。
- 训练数据额外记录 schema、split、样本数和 validator 报告。

3.3 TB 全量 SHA256 成本很高。第一轮使用“相对路径 + 文件大小 + mtime”的 inventory fingerprint；正式训练 release 再做分块/Merkle hash。

### 阶段 C：物理迁移

只有 catalog 完成后才迁移：

1. 停止所有写入任务。
2. 确认源和目标位于同一 `/mnt/d` 文件系统；同盘 rename 比复制 3.3 TB 安全得多。
3. 把数据移到 `/mnt/d/datasets/gendiff/...`。
4. 把运行产物移到 `/mnt/d/artifacts/gendiff-data-process/...`。
5. 在旧路径建立只读兼容 symlink，暂时支持硬编码脚本。
6. 逐条验证 catalog 中的路径、样本数和关键消费者。
7. 至少保留一个观察周期后再移除旧兼容路径。

### 阶段 D：归档和清理

- `legacy`：移到 versioned legacy 目录，默认只读。
- `unknown`：继续保留，直到 producer/consumer 被确认。
- `scratch`：有 catalog 和 owner 批准后才可清理。
- `duplicate`：先比较 inventory/hash，再决定保留哪一份。

绝不能根据目录名中的 `tmp`、`test`、`new`、`fixed` 直接删除。

## 8. 新数据的运行记录

以后每次正式生成数据都创建一个 run manifest：

```yaml
schema_version: data_run_v1
run_id: 20260828T153000_polygon_proxy_v1
pipeline_id: polygon_proxy_v1
status: completed

code:
  repository: gendiff-data-process
  git_commit: <sha>
  dirty: false
  environment_lock_hash: <sha256>

command:
  argv: []
  config_path: configs/polygon_proxy_v1.yaml
  config_hash: <sha256>

inputs:
  - dataset_id: <id>
    inventory_hash: <hash>

outputs:
  - dataset_id: <id>
    physical_path: <path>
    file_count: 0
    total_bytes: 0
    inventory_hash: <hash>

runtime:
  host: mingfeng-208-208
  started_at: <iso8601>
  finished_at: <iso8601>

validation:
  report: <path>
  passed: true
```

如果确实需要在 dirty worktree 上运行，manifest 必须额外记录 `git diff` 的 SHA256；不能只写 `git_commit`。

## 9. 建议先完成的最小任务

### P0：保护现有代码

1. 建立整理分支。
2. 将 untracked 的 polygon proxy、construction、tests 和 tools 按功能提交。
3. 单独审查并提交 tracked 修改和依赖文件。
4. 推送分支并做 fresh clone 验证。

### P1：建立目录册

5. 新增 `code_inventory.yaml`，先覆盖全部 Python 文件。
6. 新增 4 个 pipeline manifest：旧 block/condition、polygon proxy、construction sequence、GenDiff history area edit。
7. 为 `data/` 的所有二级目录分配 dataset ID 和状态。
8. 建立 `UNRESOLVED_PROVENANCE.md`，集中记录无法确认的路径。

### P2：确认当前链路

9. 明确当前 CityEngine stage YAML 的真实 producer。
10. 明确 2026-05 canonical edit 输出是否被 GenDiff 实际消费。
11. 用 GenDiff config 反向补齐 current dataset 的 consumer 关系。
12. 把只有“文件时间推断”的 medium-confidence 关系升级为有 command/commit 证据的 high-confidence。

### P3：迁移数据

13. 先迁移小型 `output/`、`outputs/`，验证 artifact 目录和 symlink 方案。
14. 再迁移 `pipeline/output/`。
15. 最后按 dataset ID 分批迁移 3.3 TB `data/`，每批迁移后验证 inventory。

## 10. 验收标准

整理完成不以“目录看起来整齐”为标准，而以这些问题是否可回答为标准：

- Git fresh clone 能获得全部当前代码、配置、tests 和 catalog。
- Git status 不显示真实数据或运行产物。
- 每个 Python 脚本都有 purpose 和 status。
- 每个 `data/` 二级目录都有 dataset ID、生命周期和状态。
- 所有 current/candidate 数据都能追溯到 producer 或明确标记 unknown。
- 所有 current 训练数据都能追溯到输入、代码 commit、config 和 validation。
- 旧 pipeline 与新 polygon/construction pipeline 在文档和 catalog 中明确分开。
- 没有未经登记的数据被删除。
- 新运行不会继续向代码目录随意写 `output/new/fixed/tmp`。

## 11. 最重要的执行原则

1. **先 commit/push 代码，再移动数据。**
2. **先登记，后判断是否 legacy；先判断，后归档；最后才删除。**
3. **producer 不确定就写 unknown，不猜。**
4. **数据路径不是版本；dataset ID + manifest 才是版本。**
5. **代码仓库记录“如何生成”，数据根目录保存“生成了什么”，artifact 根目录保存“这次运行发生了什么”。**

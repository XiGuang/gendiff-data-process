# 代码、数据与产物布局

## 当前状态

legacy checkout `/mnt/d/data` 同时包含 Git 仓库和 3.59 TB 已观测数据集。
本轮整理明确不移动这些数据，只先建立清晰的逻辑边界。

```text
Git 仓库
  代码 + tests + configs + catalog + docs

/mnt/d/data/data
  legacy 物理数据集（Git ignore，原地登记）

/mnt/d/data/output, /mnt/d/data/outputs, /mnt/d/data/pipeline/output
  历史 scratch 与 run 产物（Git ignore，原地登记）

/mnt/d/artifacts/gendiff-data-process
  仓库外备份和未来不可变 run 产物
```

## 版本控制边界

`.gitignore` 使用根目录边界忽略数据和输出树。YAML 不再被全局 ignore，因此
config、schema、fixture 和 catalog manifest 可以进入版本控制。大型
OBJ/PLY/PT/NPZ/图像产物继续作为安全边界被 ignore。

现有 `config/*.yaml`、根目录 `data.yaml` 和 `pipeline/pairs_t1_t2.yaml`
是带绝对数据路径的大型生成 pair index，并非源配置。它们已按 hash 登记并继续被
ignore。新源配置应放入有明确版本的 `configs/` 目录。

## 新 run 的目标状态

新的正式运行应写到代码 checkout 之外：

```text
/mnt/d/datasets/<dataset_id>/...
/mnt/d/artifacts/gendiff-data-process/runs/<run_id>/
  run.yaml
  config.yaml
  logs/
  reports/
  outputs/（或指向不可变数据集的链接）
```

每个 `run.yaml` 应记录：

- 输入 dataset ID 和 hash；
- 仓库 URL 和不可变 commit；
- clean/dirty 状态，dirty 时还要记录 diff SHA-256；
- 完整 command、config、seed、environment、host 和起止时间；
- 输出 dataset ID、数量、大小、hash 和 validation report。

## 迁移顺序

1. Fresh clone 并验证已推送的 code/catalog 分支。
2. 创建仓库外 dataset/artifact 根目录和权限。
3. 先把一个小型新 run 路由到新的 artifact 布局。
4. 携带 manifest 和可逆链接迁移小型 `output/`/`outputs/` candidate。
5. 只有确认 consumer 和 hash 后才迁移大型数据集。

本文档不授权任何批量移动或删除。

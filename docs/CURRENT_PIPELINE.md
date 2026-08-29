# 当前与候选处理链路

## 当前决策

本仓库没有任何数据生成 pipeline 被标为 `current`。Phase 1 已确认一个实际的
GenDiff 训练 consumer，但它属于独立的历史 area-v2 路径，不能证明其消费了本仓库的
candidate canonical 输出。

以下两条证据链必须保持分离。

## 已观测的 GenDiff 训练消费端

```text
/mnt/d/projects/GenDiff/datasets/history_stages_origin
  -> area-v2 loose generation（精确 argv/commit 为 unknown）
  -> /mnt/d/projects/GenDiff/datasets/history_area_edit_v2_packed
       schema: area_v2_packed_v1
       edit schema: area_v2_absolute_target_coord_no_anchor
  -> configs/packed_area_edit_v2_tiny_overfit.yaml
  -> craftsman/data/packed_area_edit_v2_data_module.py
  -> BuildingLayerEditV2System / BackboneBuildingLayerEditV2
  -> outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1
```

证据：

- 已检查的 GenDiff checkout：branch `structured_proxy`，commit
  `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`；只读审计前后均为 dirty；
- 已观测 run command：
  `python train.py --config configs/packed_area_edit_v2_tiny_overfit.yaml --train --gpu 0,1,2,3,4,5,6`;
- run config SHA-256：
  `6b61d262991fec4ec3b1787385a3f0fb0d074dddd87f3f0dd1acb9317ae40441`;
- 精确路径、选定文件 hash、有界样本结果和 evidence ID：
  `catalog/training_consumer_manifest.yaml`；
- 供人工阅读的追踪报告：`docs/TRAINING_CONSUMER_AUDIT.md`。

历史 run commit/diff、精确数据生成 command 和 environment 均为 `unknown`。
train、validation 和 test 都 alias 同一批 20,000 个样本，因此已观测指标不能作为
held-out 证据。

## 施工/拆除规范化候选链路

```text
component/tile OBJ
  -> polygon proxy YAML
  -> stage sequence
  -> canonical edit + directional condition + area-v2 packed candidate
  -X-> observed packed GenDiff loader
```

candidate 入口为
`building_process/generate_construction_sequence_canonical_edit_dataset.py`,
由 commit `b4c15a89852df01c836dded8aef75a6d5b320bb2` 引入，catalog 状态为
`candidate`。Phase 1 确认其直接兼容性为 **blocked**：它不生成所需 packed
container/edit schema，normalization 和 edit value 语义不兼容，split 互为 alias，
Phase 1 时也缺少经过审阅的 adapter。Phase 2E 现已有
`canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor` candidate adapter 和
小型 loader smoke，但旧脚本尚未迁移，真实 release 兼容性仍为 blocked。

第一轮 `construction_only` 100-building pilot 已从 clean commit
`ca2a1ecb1e851f56506de9437d8e3598d9bc6efe` 执行。其 100 栋中 60 栋成功、39 栋因
`E_CONSTRUCTION_REMOVAL` 失败、1 栋因 `E_HOLE_UNSUPPORTED` 失败；validator 自身的
source/hash/split/collision/determinism/真实 loader 检查没有附加失败，但 generation gate
按合同为 FAIL。artifact 位于
`/mnt/d/artifacts/gendiff-data-process/runs/canonicalizer_pilot_v1_ca2a1ecb1e85_b0001_b0100`。

用户随后确认任务还包括从完整建筑逐步拆除。当前 candidate 分支
`codex/canonicalizer-bidirectional-v1` 新增独立
`configs/canonicalizer_bidirectional_v1.yaml` 和
`docs/CANONICALIZER_BIDIRECTIONAL_CONTRACT.md`：纯新增/纯删除都合法，每个单调 pair
生成 construction 与 demolition 两个方向，mixed 同步新增删除仍显式失败。现有 100 栋
的正式 pilot 固定在 clean commit
`f0de8c4de1cfe3f666d5f466de998c635ebdae0d` 和 wheel SHA-256
`d38fb04cb6f2e2319f4cf7da292e4faca77eadbabbb6377c67ed755f4aeabdb1`。99 栋完成
canonicalization，输出 191 条 construction 与 191 条 demolition；但 48 个 mixed
transition 和 `building_0032` 的 hole 使 generation gate 为 FAIL。原 39 栋
`E_CONSTRUCTION_REMOVAL` building 实际都包含 mixed，而不是 39 栋纯拆除。

第一次正式尝试的 validator 发现 sample 缺少 `task_contract_id`，对应 artifact 和修复 commit
`f0de8c4de1cfe3f666d5f466de998c635ebdae0d` 均已保留。第二次 validator 的
`failures: []`，并完成 400 个 source hash、determinism、split、collision 和真实 loader
全量检查；最终 FAIL 只继承上述 generation 数据门禁。精确路径、命令和 hash 见
`catalog/canonicalizer_bidirectional_test_report.yaml`。

已观测的 `/mnt/d/data/outputs/canonical` 和 `canonical_obj` 目录属于 candidate
run。其脚本家族级 producer 已知，但精确 command/config/commit/consumer 为
`unknown`。目录名不能证明训练使用关系，详见 `catalog/legacy_outputs.yaml`。

## 两套多边形代理实现

1. `polygon_proxy/core.py` 由 `tools/build_polygon_proxy*.py` 使用，输出 proxy
   JSON/OBJ/metrics。
2. `building_process/polygon_proxy.py` 由
   `building_process/batch_polygon_proxy*.py` 使用，可输出 construction 脚本消费的
   flat YAML。

两者的 schema 和几何行为尚未证明等价。run manifest 必须明确记录入口和 producer
commit；只写 "polygon proxy" 存在歧义。

## 只读查看工具

GenDiff commit `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc` 中的
`construction_edit_animation_viewer/` 已迁到本仓库 `viewer/`，运行时 helper 位于
`tools/dataset_browser_api.py`、`tools/export_edit_animation_viewer_data.py` 和
`gendiff_data_process/viewer_packed.py`。当前查看器同时支持 legacy raw area-v2 与
candidate `area_v2_packed_v1`，能区分 `construction`、`demolition` 并播放
`INSERT_LAYER`/`DELETE_LAYER`。

该工具不生产或修改数据。三样本 synthetic packed fixture、真实 GenDiff loader 和浏览器
视觉检查通过；随后又对正式双向 artifact 的 construction/demolition、2048 点 condition、
3D 和 edit playback 做了浏览器抽检，0 个 console error。查看器兼容性通过不等于数据 gate
通过，仍不能解除本节上方的 release/training 阻塞。迁移证据与命令见
`docs/area_edit_v2_viewer_migration.md`。

## 验证状态

- 整理阶段已验证 code inventory 覆盖率和 synthetic polygon proxy test，详见
  `docs/ORGANIZATION_ACCEPTANCE.md`。
- Phase 1 训练审计为有界只读检查；未生成数据、训练、安装依赖或修改
  `/mnt/d/projects/GenDiff`。
- construction-only clean commit 的 53 个 core/property/golden/adapter 测试和 2 个
  GenDiff loader smoke 已通过；正式 pilot 结果如上，不能提升为 release。
- 双向 candidate 已通过 60 个 canonicalization 定向测试（其中未设置显式 GenDiff 路径的
  1 个 loader suite 跳过）、2 个显式真实 GenDiff loader smoke、4 个 viewer Python 测试和
  24 个前端测试。正式 pilot 打包的 382 条样本已由真实 loader 全量读取；validator 无附加
  failure，但 generation 数据门禁仍为 FAIL。

## 精确提升门槛

项目整理、fresh clone、Phase 2A 到 2E、双向实现和正式 pilot 均已有证据。当前提升门槛是：
先确认并解决 48 个 mixed transition 的业务语义，或用独立版本化 replacement 合同接收；
同时修复 `building_0032` 的 hole。随后在同一 100 栋/400 YAML 边界和新 clean commit/wheel
上重跑，generation 与 validator 必须同时 PASS。PASS 后才可另行申请 bounded overfit；
本文档不授权全量生成或训练。

机器可读 pipeline 状态位于 `catalog/pipelines/construction_sequence_v1.yaml` 和
`catalog/pipelines/history_area_edit_v2.yaml`。

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

## 施工规范化候选链路

```text
component/tile OBJ
  -> polygon proxy YAML
  -> construction sequence
  -> loose canonical edit objects and v2 sequences
  -X-> observed packed GenDiff loader
```

candidate 入口为
`building_process/generate_construction_sequence_canonical_edit_dataset.py`,
由 commit `b4c15a89852df01c836dded8aef75a6d5b320bb2` 引入，catalog 状态为
`candidate`。Phase 1 确认其直接兼容性为 **blocked**：它不生成所需 packed
container/edit schema，normalization 和 edit value 语义不兼容，split 互为 alias，
并且缺少经过审阅的
`canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor` adapter。

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

## 验证状态

- 整理阶段已验证 code inventory 覆盖率和 synthetic polygon proxy test，详见
  `docs/ORGANIZATION_ACCEPTANCE.md`。
- Phase 1 训练审计为有界只读检查；未生成数据、训练、安装依赖或修改
  `/mnt/d/projects/GenDiff`。
- `docs/CANONICALIZER_TEST_PLAN.md` 中的 canonicalizer test 仍处于计划状态，
  尚未实现或通过。

## 精确提升门槛

必须先完成项目整理及 fresh clone 验证。经过单独审阅批准后，先实现 test 和有版本的
v3-to-area-v2 adapter，再实现 canonical core。unit/golden/property/adapter/
loader-smoke gate 全部通过后，才能构造小型 versioned pilot；pilot 的 determinism、
round-trip、collision、capacity 和 building-level split gate 全部通过后，才能进行
bounded overfit。本文档不授权批量生成或训练。

机器可读 pipeline 状态位于 `catalog/pipelines/construction_sequence_v1.yaml` 和
`catalog/pipelines/history_area_edit_v2.yaml`。

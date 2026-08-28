# 服务器 Codex 交接：训练消费端与规范化器审计

日期：2026-08-28

## 目标

在修改算法或生成更多数据前，明确当前 GenDiff 训练代码与 candidate canonical
construction 数据集之间的精确合同。

这是一项证据收集任务，输出必须能够回答：

1. 训练实际使用哪个仓库 commit、loader、config、command 和物理数据路径？
2. loader 要求哪些字段、shape、token 语义、坐标 normalization、ID 和 split 约定？
3. candidate canonicalizer 是否精确输出该合同？
4. 哪些地方仍会让等价几何产生多套 supervision sequence？
5. 在实现或批量再生成前，必须通过哪套小型确定性测试？

## 起始状态

- 数据仓库分支：
  `organize/catalog-and-code-snapshot-20260828`
- 本次交接前已验证的数据仓库 commit：
  `0fb8505f37fe7709830be2e0e9949fc29243e307`
- clean 验证 checkout：
  `/mnt/d/projects/gendiff-data-process-clean-20260828`
- legacy 代码/数据混合 checkout：
  `/mnt/d/data`
- candidate canonical 实现：
  `building_process/generate_construction_sequence_canonical_edit_dataset.py`
- 先前审计预期的训练仓库：
  `/mnt/d/projects/GenDiff`

legacy 数据目录包含 119 个 dataset 目录、5,672,789 个文件和
3,587,926,590,672 个已观测 bytes。本阶段不得重新扫描或 hash 全树。

## 第一阶段流程

### 1. 记录仓库状态

对 GenDiff 训练仓库做只读记录：

- resolved path 和 remote URL；
- branch、HEAD commit、upstream 和 dirty status；
- 不暴露 secret 的相关 untracked 或 modified 文件；
- Python/environment 描述以及实际训练 command。

如果存在多个 GenDiff checkout，识别哪个产生了最新 log/checkpoint，并说明证据。

### 2. 追踪实际训练消费端

定位并追踪：

- dataset/DataLoader 入口；
- config composition 和选中的 dataset section；
- sample manifest 路径和 split 文件；
- collate/tokenization/normalization 代码；
- loss target 和 mask；
- model forward 输入；
- inference/decoding/apply 路径；
- 最新 training command、log 和 checkpoint metadata。

记录精确文件路径和 line/function 名称，区分代码默认值与实际使用的配置。

### 3. 检查有界样本

只检查足以报告以下内容的小型代表性样本：

- field name、dtype、shape、range 和 missing-value 行为；
- train/validation/test 样本或 source building 是否重叠；
- layer/point/entity ID 是否存在且稳定；
- 等价 source/target 几何是否存在重复 canonical target；
- no-op、demolition、reconstruction、ambiguity 和 invalid-sample 处理。

不得修改样本文件，不得扫描或 hash 整个 3.59 TB 数据树。

### 4. 比较生产端与消费端合同

将当前 loader 要求与以下内容比较：

- unified canonicalizer v1 规范；
- `generate_construction_sequence_canonical_edit_dataset.py` 输出；
- `catalog/pipelines/` 下的 pipeline 合同。

将每项 mismatch 分类为阻塞项、兼容性风险或文档缺口。

### 5. 定义下一项可执行测试门槛

编写 canonicalizer 测试计划，至少覆盖：

- polygon 循环起点不变性；
- winding 反转不变性；
- layer/component 排列不变性；
- 对称几何的确定性 tie-break；
- 浮点量化边界；
- 稳定的 layer/edge/point identity；
- 歧义 correspondence 的失败关闭行为；
- canonicalize/compile/apply round trip；
- 重复 run 的字节一致性；
- duplicate canonical-key 和 conflicting-target 检测；
- training loader smoke test 和 bounded overfit test。

明确 fixture、metric、command 和 pass/fail 阈值。Phase 1 不实现完整 canonicalizer。

## 必需输出

### `catalog/training_consumer_manifest.yaml`

必须包含：

- schema version 和审计时间戳；
- 训练仓库状态；
- 实际 config/command 证据；
- loader 和 model consumer 路径；
- input/output schema 和 normalization；
- dataset/split 路径；
- producer 兼容状态；
- 每项关键关系的 provenance confidence；
- unresolved 字段和 blocker。

### `docs/TRAINING_CONSUMER_AUDIT.md`

必须先写结论，再记录证据链、mismatch、泄漏/唯一性风险和建议 gate。

### `docs/CANONICALIZER_TEST_PLAN.md`

必须提供足够详细的测试矩阵，使后续 Codex 任务无需重新定义语义即可实现。

## 完成标准

只有满足以下条件，Phase 1 才算完成：

- 实际训练 consumer 已完成端到端追踪；
- loader 合同足够精确，可用于构建 fixture；
- candidate producer/consumer mismatch 已分类；
- 三份必需交付物存在且可正确 parse/render；
- 所有 blocker 和不可访问证据均已显式记录；
- 未修改任何 dataset、output、训练代码或 legacy Git storage。

报告 Phase 1 后停止。实现 canonicalizer 或运行批量生成/训练前等待审阅。

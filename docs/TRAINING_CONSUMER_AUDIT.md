# GenDiff 训练消费端第一阶段审计

审计时间：2026-08-28

审计范围：只读检查 `/mnt/d/projects/GenDiff`，并与本仓库候选 producer 和 Unified Canonicalizer v1 规范比较。

## 结论

1. 已确认的实际训练消费者不是 loose YAML edit 文件，而是
   `area_v2_packed_v1` PT state bank/index/shard。实际 run 是
   `/mnt/d/projects/GenDiff/outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1`，
   命令、运行时 config、consumer 代码快照、metrics 和 checkpoint 均存在。
2. 该 run 使用 `area_v2_absolute_target_coord_no_anchor`：layer/point 动作携带显式
   source/target index，MOVE/INSERT 的值是 area-normalized XZ 中的绝对 target
   coordinate。模型输入最多 64 层、每层 32 点、16 栋楼，condition 先固定为
   2048 点，再由 condition encoder 取 256 token。
3. 当前 20k 数据的 `val`、`test` 都 alias `train` 的 20 个 shard；因此已有
   validation/test 指标不能解释为 held-out 泛化。证据是
   `/mnt/d/projects/GenDiff/datasets/history_area_edit_v2_packed/dataset_meta.yaml:97941`
   及对三个 index PT 的 bounded probe。
4. 本仓库候选
   `building_process/generate_construction_sequence_canonical_edit_dataset.py`
   不能直接或经现有 packer 无损喂给实际 loader。它没有所需 edit schema/version
   与 area normalization，MOVE 使用 delta、INSERT 使用 anchor-relative value，且
   只写 loose YAML/PT 和相同的 train/val/test pair list。
5. Unified Canonicalizer v1 目前只有规范，没有已审计实现，也没有
   `canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor` adapter。canonical
   hash/config hash、lineage、容量和 collision 信息也未进入 packed metadata 或训练
   batch，所以 producer/consumer compatibility 当前为 **blocked**。
6. 训练 run 的精确 Python 环境、历史 Git commit/diff hash、packed 数据生成命令和
   producer commit 均为 `unknown`。当前 checkout 的 selected consumer 文件与 run
   code snapshot 字节一致，但这不能替代历史 run manifest。

完整机器可读结论见 `catalog/training_consumer_manifest.yaml`。

## 仓库状态

审计仓库在编辑前为 clean detached HEAD：
`ddd7a68685042dc93331a105d0f8b7449c8e467a`。命令：

```bash
git -C /root/.codex/worktrees/8767/data status --short --branch
git -C /root/.codex/worktrees/8767/data rev-parse HEAD
```

训练仓库为 branch `structured_proxy`，upstream `origin/structured_proxy`，当前 HEAD：
`c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`，remote：
`git@github.com:XiGuang/GenDiff.git`。编辑前已存在以下 dirty 项，本审计未修改它们：

```text
 M configs/packed_area_edit_v2_tiny_overfit.yaml
 M docs/area_edit_v2_dataset_output.md
 M docs/area_edit_v2_viewer_migration.md
 M docs/packed_area_edit_v2_datamodule_design.md
 M tools/build_area_edit_v2_packed_dataset.py
 M tools/generate_area_sequence_v2_dataset.py
?? .playwright-cli/
?? craftsman/inference/
?? docs/area_layer_edit_v2_forward_pipeline_io.md
?? docs/area_layer_edit_v2_network.md
?? tools/generate_history_forward_cases.py
?? tools/run_area_layer_edit_v2_forward.py
```

证据命令：

```bash
git -C /mnt/d/projects/GenDiff status --short --branch
git -C /mnt/d/projects/GenDiff rev-parse HEAD
git -C /mnt/d/projects/GenDiff remote -v
git -C /mnt/d/projects/GenDiff rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
```

## 实际运行证据链

### 命令与配置

`/mnt/d/projects/GenDiff/outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1/cmd.txt`
记录的命令是：

```bash
python train.py --config configs/packed_area_edit_v2_tiny_overfit.yaml --train --gpu 0,1,2,3,4,5,6
```

运行时原始和解析后配置分别保存在：

- `/mnt/d/projects/GenDiff/outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1/configs/raw.yaml`
- `/mnt/d/projects/GenDiff/outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1/configs/parsed.yaml`

两者确认 `data_type=packed-area-edit-v2-datamodule`、
`system_type=building-layer-edit-v2-system`、7 GPU、dataset folder
`datasets/history_area_edit_v2_packed`。当前 dirty config、run raw config 和 run code
snapshot config 的 SHA256 都是：

```text
6b61d262991fec4ec3b1787385a3f0fb0d074dddd87f3f0dd1acb9317ae40441
```

该 config 与当前 HEAD blob 不同；
`git diff -- configs/packed_area_edit_v2_tiny_overfit.yaml` 显示 run tag 从
`tiny_overfit_10k` 改为 `tiny_overfit_20k_fixed_1`、dataset 从
`area_edit_v2_40k_packed_test` 改为 `history_area_edit_v2_packed`，并取消 resume。
因此只记录当前 commit 不能复现该 run。

`train.py:138` 的 `main()` 通过 `load_config()` 合并 YAML 和 CLI override，
`train.py:184-185` 从 registry 实例化 DataModule 与 system，`train.py:262-263`
执行 `trainer.fit()` 后立即 `trainer.test()`。

### 代码快照

run 自带 `code/` snapshot。以下当前文件与 run snapshot 的 SHA256 一致，且这些当前
文件相对 HEAD 没有 diff：

| 文件 | SHA256 |
|---|---|
| `train.py` | `57029c7d357d46338c5e7ee1646b3586027d4a1bc4c3c3c445a29d34202e8dcd` |
| `craftsman/data/packed_area_edit_v2_data_module.py` | `a0d6257fefcd4e79eefd3768145a813f0bb10e1c300e0514bc71d2d5ef8be254` |
| `craftsman/systems/building_layer_edit_v2_system.py` | `d96467a1cbf09079540b641ef8127186933e54d6e16d243a60cf8596f6948f85` |
| `craftsman/models/backbone_models/backbone_building_layer_edit_v2.py` | `9dac01fb39e57f91b9a622d71d9e1c79d61b51d5e478934978f0d66152e9f771` |

这建立了 run 到 selected file bytes 的高置信关系。run 当时的 Git HEAD、完整 dirty
diff hash 和 environment 仍为 `unknown`。

### 日志与检查点

最近 checkpoint 文件：

- `ckpts/epoch=170-step=122000.ckpt`，424,466,680 bytes，mtime
  `2026-06-11T21:00:02+08:00`
- `ckpts/last.ckpt`，424,466,680 bytes

`csv_logs/version_0/metrics.csv` 最后观测到 epoch 171、step 122409。Phase 1 没有
加载 424 MB checkpoint payload，因此其中的 optimizer/hyperparameter metadata 为
`unknown`；这里只使用路径、文件名、大小、mtime 和 metrics CSV，不声称 run 正常
完成。

## 数据加载合同

入口为
`/mnt/d/projects/GenDiff/craftsman/data/packed_area_edit_v2_data_module.py:24`
的 `PackedAreaEditV2Dataset`。`__init__()` 依次读取：

```text
<dataset_folder>/dataset_meta.pt
<dataset_folder>/states.pt
<dataset_folder>/<split>_index.pt
<dataset_folder>/shards/<split>/<split>_<index>.pt
```

`_validate_meta()` 强制：

```text
schema_version      == area_v2_packed_v1
edit_schema_version == area_v2_absolute_target_coord_no_anchor
```

不匹配时抛 `ValueError`。这意味着仅写 `train.yaml`、`val.yaml`、`test.yaml` 的
producer 不是该 consumer 的直接输入。

### 实际数据集与划分

`dataset_meta.yaml` 和 bounded PT probe 得到：

| 项目 | 实际值 |
|---|---:|
| 源阶段根目录 | `/mnt/d/projects/GenDiff/datasets/history_stages_origin` |
| 建筑/状态 | 5 / 1024 |
| 合法前向 pair | 98,976 |
| 抽样 pair | 20,000，seed 0 |
| packed 分片 | 20 个 train 分片 |
| train/val/test 样本 | 20,000 / 20,000 / 20,000 |
| val/test 划分 | 均 alias train，分片列表完全相同 |
| 已观测存储容量 | 21 层，每层 7 点 |
| 已存储 condition | float32 `[8192,3]` |

原始 dataset metadata 记录了生成参数，但没有 exact argv、Git commit、dirty diff
hash 或 dependency environment，所以 producer command/commit 为 `unknown`。

### 坐标与归一化

实际 metadata：

```yaml
coordinate_normalized: true
normalization_scope: area
normalization_stats_tensor: [-151.3255, 78.1355, 195.594, 97.797, 195.594]
normalization_stats_order: [center_x, center_z, scale_xz, center_y, scale_y]
point_value_semantics: absolute_target_coord
anchor_supervision: false
```

producer 函数
`tools/generate_area_sequence_v2_dataset.py:1139`、`:1191`、`:1229` 定义：

```text
x' = (x - center_x) / scale_xz
z' = (z - center_z) / scale_xz
y' = (y - center_y) / scale_y
```

`scale_xz == scale_y == max(span_x, span_z, span_y, min_scale)`。footprint 是 XZ
二维点，height 是 Y，condition 是 XYZ。当前 loader 不再归一化；它信任 packed
输入已经归一化。

### 单样本有界探查

只加载了 metadata、三个 index、`states.pt` 和第一个 train shard；没有遍历其余
19 个 shard。样本 0：

```text
pair_name: pair_000000_area_state_000000_to_area_state_000004
source/target state index: 0 / 4
source/target layer count: 0 / 2
condition: float32 [8192,3], finite, range [-0.49999427795410156, 0.25707948207855225]
edit_object: 2 INSERT layers
validation: layer/point counts match, max coord/height error 0, max AR tokens 6
```

该样本只证明 schema、dtype、shape 和一个纯 INSERT case；它不代表全数据 action
分布。更广泛的分布引用现有 server audit，不在 Phase 1 重扫 20k shard。

### 批次 Schema

配置固定 `L=64`、`P=32`、`T=P+1=33`、loader condition `N=2048`：

| key | 数据类型/形状 | 语义 |
|---|---|---|
| `source_point_coords` | float32 `[B,64,32,2]` | 已归一化 XZ |
| `source_point_mask` | bool `[B,64,32]` | source 点有效性 |
| `source_height_values` | float32 `[B,64,2]` | 已归一化 min/max Y |
| `source_layer_mask` | bool `[B,64]` | source 层有效性 |
| `target_*` | 对应 source 形状 | target supervision |
| `layer_actions` | int64 `[B,64]` | KEEP/MODIFY/DELETE/INSERT/PAD |
| `point_actions` | int64 `[B,64,32]` | KEEP/MOVE/DELETE/INSERT/PAD |
| `ar_action_targets` | int64 `[B,64,33]` | point action + EOS/PAD |
| `ar_source_index_targets` | int64 `[B,64,33]` | 显式 source index |
| `ar_target_index_targets` | int64 `[B,64,33]` | 显式 target index |
| `ar_value_targets` | float32 `[B,64,33,2]` | MOVE/INSERT 的绝对 target XZ |
| `ar_token_mask` | bool `[B,64,33]` | 包含 EOS 的有效 AR token |
| `source_building_ids` | int64 `[B,64]` | 0 表示缺失/pad；其他值为 source ID + 1 |
| `change_point_clouds` | float32 `[B,2048,3]` | 已归一化 XYZ |
| `normalization_stats` | float32 `[B,5]` | 反归一化元数据 |

`_fix_num_points()` 对超过 2048 的 condition 取前缀，对不足者重复，对空输入输出全
零。`BuildingChangeConditionV2.forward()` 再通过 `torch_cluster.fps` 将 2048 点降到
配置的 256 tokens。这个 deterministic prefix 不是空间覆盖采样保证。

## 模型、损失与执行合同

`BuildingLayerEditV2System._forward_batch()`
(`/mnt/d/projects/GenDiff/craftsman/systems/building_layer_edit_v2_system.py:135`)
把 condition tokens 与 source structure 送入
`BackboneBuildingLayerEditV2.forward()` (`.../backbone_building_layer_edit_v2.py:123`)。
训练时同时传入 layer/height/AR teacher-forcing targets。

动作 enum：

```text
LayerActionV2: KEEP=0 MODIFY=1 DELETE=2 INSERT=3 PAD=4
ARPointEditActionV2: KEEP=0 MOVE=1 DELETE=2 INSERT=3 EOS=4 PAD=5 BOS=6
```

`BuildingLayerEditV2System._compute_losses()` (`...system.py:180`) 使用：

- layer action CE，PAD 单独加权 0.2；
- target height value loss，mask 为 `target_layer_mask`；
- AR action/source-index/target-index/length losses；
- MOVE/INSERT 的 absolute target coordinate value loss。

source index 只监督 KEEP/MOVE/DELETE；target index 只监督 KEEP/MOVE/INSERT；value
只监督 MOVE/INSERT。metrics CSV header 与这些 loss/accuracy key 一致。

decode 路径是 `BuildingLayerEditV2System._decode_ar_predictions()` (`...system.py:512`)
→ `LayerwiseARGeneratedEditDecoderV2.decode_batch()` →
`BuildingLayerEditObjectApplierV2.apply_to_layers()`
(`/mnt/d/projects/GenDiff/craftsman/models/editing/building_layer_edit_v2/edit_object_applier.py:22`)。
applier 按显式 target point index 重建 footprint，并按 building/level/height/proxy ID
排序输出。

## 前向流程状态

大场景 runner 和 pipeline 当前都是 untracked：

- `/mnt/d/projects/GenDiff/tools/run_area_layer_edit_v2_forward.py`
- `/mnt/d/projects/GenDiff/craftsman/inference/area_layer_edit_v2_forward.py`

runner 读取 training config/checkpoint，按 tile 构造同一批 key，再调用 system 的
`_forward_batch()` 与 `_decode_predictions()`，最后 merge。只在
`/mnt/d/projects/GenDiff/outputs/history_forward_cases/cases/case_000000` 找到
source/target/condition case 输入；在该 output root 的 depth 3 内没有
`final_scene.yaml` 或 `summary.yaml`，因此 completed forward 证据为 `unknown`。

当前 forward 风险来自代码本身：point list 和 condition 超容量时取前缀；非 strict
模式下 capacity-overflow tile 会被跳过；building embedding map 在检查前已截断；
并且 forward 使用 per-tile normalization，而训练数据使用整个五栋 area 的一次性
normalization。两种 normalization 的等价性没有测试。

## 生产端对比

| 合同 | 实际 20k producer/packer | 本仓库 candidate | 统一 v1 规范 |
|---|---|---|---|
| 输出容器 | packed PT state/index/shard | loose YAML + condition PT | canonical sequence/edit v3 + manifest |
| 编辑 schema | `area_v2_absolute_target_coord_no_anchor` | 未声明；delta/anchor value | `canonical_edit_v3` 整数网格 |
| 坐标 | area-normalized float | raw float | 量化整数 world XZY |
| 身份 | building/proxy/point 整数 ID | `proxy_id*stride+point_index` 起步 | sequence lineage，不信任 raw index |
| 层匹配 | greedy，proxy/source proxy 优先 | greedy，同类逻辑 | 确定性全局最优 |
| 量化 | float + epsilon | Python `round()`/epsilon | half-away-from-zero 整数网格 |
| 划分 | train/val/test 相同 | train/val/test 相同 | building-level 隔离 |
| hash | 无 canonical config/hash | 无 canonical config/hash | geometry/edit/condition/config hash 必填 |
| 当前 loader | 已有成功 run | blocked | blocked，缺 adapter |

本仓库 candidate 的关键证据：

- `:69` 使用 Python `round()`；
- `:79-91` 用 `source_proxy_id * 100000 + point_index` 生成 source point ID；
- `:405` 是 greedy `_match_layers()`；
- `:502-557` MOVE value 是 delta，INSERT value 可能是 anchor-relative；
- `:1105-1230` 只写 loose dataset，未记录 consumer 所需 edit schema/normalization；
- `:1211-1212` 把同一 `pair_records` 写给三个 split。

因此脚本名中的 `canonical` 不能作为 v1 compatibility 证据。

## 不匹配项分类

### 阻塞项

- 缺少 `canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor` 的显式、可逆、
  versioned adapter，以及 adapter 到 packed tensors 的 golden tests。
- candidate 缺 schema、area normalization、packed container 和 consumer-required
  metadata；现有 packer会把缺失 `edit_schema_version` 写成 `None`，loader 随后拒绝。
- candidate/observed generator 的 fallback point ID 依赖数组下标，违反 ring/permutation
  invariance；layer matching 是 greedy，不满足统一 tie-break。
- val/test 与 train 完全重叠，任何 release/held-out claim 均被阻断。
- packed release 缺 producer command、commit/diff hash、environment 与 canonical config
  hashes，不能可靠再生。

### 兼容性风险

- data module 和 forward 的 tensor builder 都会 slice；strict training capacity check
  没有覆盖所有预计算 tensor/原始 source point 静默截断路径。
- condition 的 deterministic prefix 与 Unified v1 deterministic stratified/FPS 合同不同。
- forward per-tile normalization 与 training area normalization 可能让同一绝对几何得到
  不同数值问题。
- `canonicalizer.py` 中现存的 `BuildingLayerEditCanonicalizerV2` 只是按 layer/point slot
  和 `allclose` 比较，并非 Unified Canonicalizer v1。

### 文档缺口与未知项

- 历史 run commit、dirty diff hash、Python 可执行文件、依赖环境；
- packed 数据 exact generation/packing argv 与 code state；
- CityEngine stage YAML 生产端；
- checkpoint payload metadata；
- completed large-scene forward output 与 GT geometry metrics。

## 数据泄漏与唯一性风险

split leakage 是直接事实，不是抽样推断。uniqueness 风险同时由代码和既有审计支持：
candidate 与实际 area generator 都能按 point index 造 ID；同几何环的 cyclic start 或
winding 改变会改变 point identity 和 AR target。当前 20k 数据的单样本 reconstruction
report 只证明“按现有标签 apply 能回到现有 target”，不能证明等价几何只有一个标签。

需要在 canonical key 层检查：

```text
(source_stage_hash, condition_hash) -> exactly one (target_stage_hash, edit_hash)
```

任何 duplicate key with conflicting target/edit 必须 hard fail，而不是保留多个训练答案。

## 第一阶段验证命令

执行过的 checks 都是小范围只读操作：

```bash
git -C /mnt/d/projects/GenDiff status --short --branch
git -C /mnt/d/projects/GenDiff diff -- configs/packed_area_edit_v2_tiny_overfit.yaml
sha256sum <四个 selected current consumer files及对应run snapshot files>
head -n 1 /mnt/d/projects/GenDiff/outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1/csv_logs/version_0/metrics.csv
tail -n 5 /mnt/d/projects/GenDiff/outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1/csv_logs/version_0/metrics.csv
```

bounded sample 使用现成 `/mnt/d/anaconda3/envs/gendiff/bin/python` 和
`torch.load(..., map_location="cpu")`，只读取 metadata、三个 index、`states.pt`、
`train_00000.pt` 的 sample 0。仓库 `.venv` 因缺少 Python 3.12 stdlib `encodings`
无法启动；没有安装依赖。

## 精确下一门槛

Phase 1 到此停止。下一 gate 是人工审阅本报告、
`catalog/training_consumer_manifest.yaml` 与 `docs/CANONICALIZER_TEST_PLAN.md`；审阅通过
后，先实现测试与显式 v3→area-v2 adapter，再实现 canonical core。只有 unit/golden/
property/adapter/loader smoke 全部通过，才允许构造一个 versioned 小 pilot；pilot 的
determinism、round-trip、collision、capacity 和 building-level split gate 全通过后，
才允许 bounded overfit。当前不允许 bulk regeneration 或继续训练。

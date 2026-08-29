# Canonicalizer 双向单调合同

- 日期：2026-08-29
- 合同 ID：`bidirectional_monotonic_v1`
- 配置：`configs/canonicalizer_bidirectional_v1.yaml`
- 状态：candidate；代码、定向测试和只读 100-building fingerprint 已完成，正式双向
  pilot 尚待审阅与 clean commit
- 训练：未授权，本合同不包含训练

## 1. 业务目标

数据必须同时覆盖两类过程：

1. 从空或不完整建筑逐步增加体积，记为 `construction`。
2. 从完整或较完整建筑逐步删除已有体积，记为 `demolition`。

几何 canonical core、量化、ring、slab decomposition、lineage 和 edit 排序仍使用
`canonicalizer_v1` / `canonical_geometry_v1`。本次版本化的是任务、condition 和 pair
打包合同，不改写已经完成审阅的 v1 stage/edit hash。

历史 `configs/canonicalizer_v1.yaml` 保留为 `construction_only`，用于复现 commit
`ca2a1ecb1e851f56506de9437d8e3598d9bc6efe` 的第一轮 pilot，不得用新合同覆盖其产物。

## 2. Transition 分类

对有向阶段对 `source -> target`，先在整数 canonical solid 上计算：

```text
added   = Solid(target) - Solid(source)
removed = Solid(source) - Solid(target)
```

| `change_kind` | 判定 | 双向单调训练处理 |
|---|---|---|
| `construction` | `added > 0` 且 `removed == 0` | 接受 |
| `demolition` | `added == 0` 且 `removed > 0` | 接受 |
| `noop` | 二者均为 0 | 显式核算后跳过 |
| `mixed` | 二者均大于 0 | `E_MIXED_CHANGE_UNSUPPORTED`，不得写入训练 shard |

`mixed` 表示同一步中既添加又删除，属于替换、重建或上游阶段不一致。它不能仅凭“存在
removed volume”改名为 demolition。后续若支持 replacement，必须新增独立 condition
语义和版本，不能放宽本合同。

洞、自交、无效高度、容量超限和实体不一致仍按原 hard error 失败关闭。

## 3. Pair 生成

Split 始终先按 `building_uid` 分配，同一建筑的所有方向都留在同一 split。对每个相邻、
非 no-op、非 mixed 的无向 stage pair，固定产生两个有向样本：

```text
较小实体 -> 较大实体：construction
较大实体 -> 较小实体：demolition
```

即使原始 stage 下标方向本身是 demolition，也同时生成它的反向 construction。反向 edit
直接比较已经分配 lineage 的两个 canonical stage，调用 `build_canonical_edit()`，不得反转
token 数组或猜测 point identity。两个方向都必须分别通过 apply round-trip。

Pilot 对 mixed condition 错误采用 pair 级显式失败：同一 building 的其他合法 pair 仍可生成，
但整个 pilot gate 仍为 FAIL。任何 building 级 canonicalization error 则核算该 building 的
全部有向槽位。

双向核算式为：

```text
attempted = building_count * adjacent_transition_count * 2
attempted = emitted + noop_skipped + duplicates_deduplicated + explicit_failures
silent_drop_count = 0
```

## 4. Condition 合同

新配置固定 `surface_mode: directional_delta_exterior`：

```text
construction condition solid = target - source
demolition   condition solid = source - target
```

两类 condition 都继续输出 2048 个 XYZ 点，保持已审计 GenDiff area-v2 loader 的
`[N, 3]` 张量合同。采样仍使用 canonical delta surface、largest-remainder、Halton 候选、
字典序起点 FPS 和最终 XYZ 字典序排序。

以下字段必须进入 packed sample 的 `canonical_metadata`：

- `change_kind`；
- `condition_hash`，其 payload 包含 `change_kind`；
- `pair_hash`，覆盖 source/target/edit/condition hash 和 `change_kind`。

同一个无向 stage pair 的 construction/demolition 使用完全相同的 XYZ 点，避免用采样噪声
暗示方向。Seed material 固定为：

```text
min(source_stage_hash, target_stage_hash)
+ max(source_stage_hash, target_stage_hash)
+ condition_config_hash
```

`change_kind` 不进入 seed，但进入 condition hash；因此正反方向 points/seed 相同，condition
hash 不同。`noop` 不得用零点伪装，`mixed` 不得只采样其中一侧。

## 5. Consumer 边界

`area_v2_packed_v1` 和 `area_v2_absolute_target_coord_no_anchor` 保持不变；真实 GenDiff loader
会读取 XYZ、source state 和 edit target，额外 canonical metadata 不改变当前 tensor shape。

已知限制：已审计 loader 不会把 `change_kind` 作为独立模型输入通道。当前设计依靠 source
实体与 delta 点的位置关系区分施工和拆除，同时把方向写入 metadata/hash 供校验和统计。
“是否需要显式符号通道或双分支 condition encoder”仍为 `unknown`，必须通过后续独立
building-level overfit/held-out 实验判断；本阶段不训练，也不修改 `/mnt/d/projects/GenDiff`。

## 6. 100-building 有界证据

只读输入固定为：

- `/mnt/d/projects/GenDiff/datasets/history_stages_all_new/building_0001` 至
  `building_0100`；
- 每栋只读 `stage_0` 至 `stage_3` 的 YAML，共 400 个文件；
- GenDiff checkout commit：`c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`，保持 dirty 且未修改。
- 正式 packed 子目录名固定为 `outputs/canonicalizer_pilot_bidirectional_v1`，与历史
  `outputs/canonicalizer_pilot_v1` 分离。

在 2026-08-29 对上述边界运行整数 canonical solid 分类，300 个原始相邻槽位结果为：

| 类别 | transition 数 |
|---|---:|
| construction | 187 |
| demolition | 4 |
| noop | 58 |
| mixed | 48 |
| 因 `E_HOLE_UNSUPPORTED` 未能分类 | 1 栋 / 3 个槽位 |

39 栋 building 含至少一个 mixed transition；并非 39 栋都是合法 demolition。

当前 candidate 的有向核算为：600 个尝试槽位 = 打包去重前 382 个候选 pair + 116 个
no-op 跳过槽位 + 96 个 mixed pair failure + 6 个 hole building 失败槽位，
`silent_drop_count=0`。
workers 1/2/8 和 `PYTHONHASHSEED=0/1/9876` 的 fingerprint 均为
`3946e58c0b9b153909937571fa76d3188e7539ce14c2c846316f3ed80d5c6b0e`。这是 dirty
worktree 的只读 fingerprint，不是正式 artifact；机器可读证据见
`catalog/canonicalizer_bidirectional_test_report.yaml`。

可复现命令：

```bash
/mnt/d/anaconda3/envs/gendiff/bin/python tools/build_canonicalizer_pilot.py \
  --fingerprint-only \
  --dataset-root /mnt/d/projects/GenDiff/datasets/history_stages_all_new \
  --config configs/canonicalizer_bidirectional_v1.yaml \
  --building-start 1 --building-count 100 --stage-indices 0,1,2,3 \
  --workers 1
```

命令输出必须报告 `source_transition_counts`、有向 `change_kind_counts`、building error、
pair error 和 fingerprint。正式 artifact 仍必须在 clean commit 和对应 wheel SHA256 上生成。

## 7. Gate

进入训练前必须同时满足：

1. construction 和 demolition 均有非零覆盖；
2. 所有 emitted pair 的 condition kind、metadata kind 和几何分类一致；
3. 正反两个 edit 均 100% round-trip；
4. workers/Python hash seed fingerprint 一致；
5. building split overlap、supervision collision 和 silent drop 均为 0；
6. 真实 GenDiff loader 读取全部三个 split；
7. mixed、hole 和其他 hard error 被修复或由明确的新任务合同接收，不能筛掉后宣称 PASS。

当前精确下一 gate：审阅当前代码与配置 hash，补充双向 condition 黄金 hash；经授权提交并
推送 clean commit 后，重跑最多 100-building 双向 pilot。预计现有源数据仍会因 48 个 mixed
transition 和 `building_0032` 的 hole 标为 FAIL；这是数据/任务阻塞证据，不是实现失败。

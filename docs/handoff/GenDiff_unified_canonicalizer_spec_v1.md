# GenDiff 统一规范化器 v1 规范

- 状态：Phase 2 candidate implementation；尚未通过 pilot/release gate
- 版本：`canonicalizer_v1`
- 日期：2026-08-28
- 目标仓库：`/mnt/d/projects/GenDiff`
- 关联审计：`D:\Projects\DroneDiff\GenDiff_server_audit_2026-08-28.md`

2026-08-29 的实现归属决策将 canonical core 的唯一源调整为本仓库
`gendiff_data_process/canonicalization/`，GenDiff 通过版本化 adapter/依赖消费。
决策证据见 `docs/CANONICALIZER_PHASE2_DECISIONS.md`；本规范的几何语义不变。

## 1. 目的

Unified Canonicalizer 将同一个物理建筑实体的多种等价 proxy 表达，转换为唯一、可复现的规范表达，并生成唯一的跨阶段 lineage 和 edit supervision。

它必须解决以下现有问题：

1. 环起点、顺逆时针、输入层顺序或原始 `proxy_id` 改变，导致同一几何产生不同标签。
2. 按 `point_index` 伪造跨阶段顶点身份，产生假的 MOVE/INSERT/DELETE。
3. 同一个实体被不同的重叠 proxy 分解表示，导致外部条件相同但 edit 标签不同。
4. construction 数据混入 no-op、拆除和重构阶段。
5. 数据生成、训练和 forward 使用不同的规范化规则。

Canonicalizer 是数据合同的一部分。任何会改变 canonical 输出、lineage、edit 顺序或 hash 的规则变化，都必须升级 canonicalizer 版本并重新生成数据集。

## 2. 规范用语

- **MUST / 必须**：不满足时数据不得进入正式训练集。
- **SHOULD / 应当**：默认执行；偏离时必须记录原因和配置。
- **MAY / 可以**：可选行为，不得影响核心确定性。
- **raw layer**：上游 OBJ/YAML 产生的原始 proxy 层。
- **canonical layer**：由阶段实体的唯一空间分解产生的无交叠挤出体。
- **lineage**：跨阶段持续存在的层或点的逻辑身份。
- **canonical hash**：仅由规范化后的整数几何和版本配置计算的 SHA256。

## 3. v1 范围和明确限制

### 3.1 v1 包含

- 单栋建筑的多阶段 YAML/内存 layer 数据。
- 水平面为 XZ，竖直方向为 Y 的挤出多边形 proxy。
- 环规范化、实体 union、无交叠 slab 分解。
- 相邻阶段的确定性 layer matching 和 point matching。
- 任意合法阶段对的唯一 edit v3。
- canonical YAML/JSON、兼容 v2 的整数 ID adapter、hash 和审计报告。
- 训练数据生成和 forward pipeline 共用的 API。

### 3.2 v1 不包含

- 从 image/VGGT 生成条件点云。
- 大场景 tile 聚类和调度本身。
- 材质、纹理、屋顶语义或原始 mesh 拓扑的保持。
- 带洞 polygon 的建模。
- demolition supervision。

### 3.3 v1 决策

| 议题 | v1 决策 |
|---|---|
| 任务语义 | construction-only；目标实体必须包含源实体 |
| 洞 | 不支持，检测到即 hard fail |
| MultiPolygon | 拆成多个 canonical layer，禁止只保留最大块 |
| 原始层重叠 | 按 solid union 消除，记录 warning；canonical 输出必须无体积重叠 |
| 原始 `proxy_id` | 只保留在 provenance 中，不参与规范身份和匹配 |
| 原始 `point_ids` | 只作诊断 hint，不直接信任 |
| 坐标 | 进入 canonicalizer 前必须已经转换到统一建筑/世界坐标系 |
| 数值核心 | 先量化为整数网格，再进行规范化、比较和 hash |
| no-op 阶段对 | 默认不进入训练 pair，可单独进入显式 no-op bucket |

## 4. 核心不变量

对同一个 `canonicalizer_version + canonicalizer_config_hash`，实现必须满足：

1. **Idempotence**：`C(C(x)) == C(x)`，且序列化字节和 hash 相同。
2. **Permutation invariance**：改变 raw layer 顺序、文件枚举顺序、raw proxy id，不改变输出。
3. **Ring invariance**：环循环平移、顺逆时针翻转、首尾重复点，不改变输出。
4. **Quantization invariance**：落入同一量化格的微小坐标扰动不改变输出。
5. **Decomposition invariance**：只要 raw layers 的实体 union 相同，其 canonical stage 必须相同。
6. **Determinism**：进程数、worker 数、输入文件遍历顺序不改变输出。
7. **Round-trip**：对任意有效 pair，`canonicalize(apply(source, edit))` 的 stage hash 必须等于 target stage hash。
8. **不允许静默丢失**：禁止静默丢弃 MultiPolygon 小分量、超容量点、超容量层或无法匹配的数据。

## 5. 几何等价关系

### 5.1 Stage 实体定义

一个 raw layer 表示：

```text
extrude(footprint, [min_height, max_height))
```

一个 stage 的物理实体为所有有效 raw layer 挤出体的 union：

```text
Solid(stage) = Union(extrude(raw_layer_i))
```

两个 stage 等价，当且仅当它们在 v1 整数量化网格上的 `Solid(stage)` 完全相等。原始 layer 数量、顺序、ID 和重叠方式不属于实体语义。

### 5.2 等价输入必须得到相同结果

以下变化必须保持 stage hash 不变：

- `[p0,p1,p2,p3]` 改成 `[p2,p3,p0,p1]`。
- CCW 改成 CW。
- 增加重复闭合点 `p0`。
- 插入落在同一直线段上的冗余共线点。
- raw layer 重新排序或重新编号。
- 一个实体由一层表示，改为若干相互重叠/相邻但 union 相同的层表示。

以下变化不等价：

- 量化后实体边界不同。
- 高度量化后不同。
- 出现或消失一个实体分量。
- 出现洞；v1 会拒绝而不是尝试近似。

## 6. 配置合同

配置文件是一个 bundle，但必须分别计算 core、validation 和 condition 三个 hash。模型容量变化不能改变几何 stage hash。

建议初始配置：

```yaml
schema_version: canonicalizer_bundle_v1

canonicalizer:
  canonicalizer_version: canonicalizer_v1
  geometry_version: canonical_geometry_v1
  coordinate_frame: world_xzy
  grid_xz: 0.001
  grid_y: 0.001
  rounding: half_away_from_zero
  polygon:
    remove_collinear: true
    reject_self_intersection: true
    reject_holes: true
    multipolygon_policy: split
    raw_overlap_policy: union_with_warning
  layer_matching:
    require_positive_intersection: true
    min_iou_3d_q: 10000       # IoU * 1e6，即 0.01
    min_smaller_coverage_q: 500000  # intersection/min(volume) * 1e6，即 0.5
    optimal_tie_break: lexicographic
  point_matching:
    max_move_distance_ratio: 0.25
    min_move_distance_q: 5
    fallback: delete_then_insert

validation_profile:
  mode: construction_only
  removed_volume_tolerance_q3: 0
  drop_noop_pairs: true
  capacity:
    max_layers: 64
    max_points_per_layer: 32
    max_buildings_per_tile: 16
    overflow_policy: error

condition_sampling:
  surface_mode: addition_exterior
  point_count: 2048
  sampler: deterministic_stratified_fps
```

说明：

- 当前 YAML 坐标精度为约 `0.001`，所以 v1 默认按 1 mm 网格量化。若坐标单位不是米，必须在首批 100 栋 pilot 前修订该配置。
- `geometry_config_hash` 对 `canonicalizer` 中影响 stage 几何的字段计算：`geometry_version`、坐标系、网格、rounding、polygon 和 solid partition 规则。它进入 `stage_hash`。
- `canonicalizer_config_hash` 对完整 `canonicalizer` 子树计算，包括 geometry、layer matching、point matching 和 edit ordering。它进入 edit/sequence hash。
- `validation_profile_hash` 对任务语义、容量和错误策略计算，只决定数据是否被某个模型 profile 接受，不改变几何 stage hash。
- `condition_config_hash` 对 condition sampling 子树计算，进入 condition hash。
- 调整网格、匹配阈值或 tie-break 会改变 `canonicalizer_config_hash`；调整模型容量只改变 `validation_profile_hash`。
- 一个正式数据集必须固定并记录三个 hash，不同 hash 的产物不得混装。

## 7. 输入合同

### 7.1 核心输入

```yaml
schema_version: raw_building_sequence_v1
building_key: building_0001
coordinate_frame: world_xzy
stages:
  - stage_index: 0
    stage_key: stage_0
    layers: []
  - stage_index: 1
    stage_key: stage_1
    layers:
      - raw_proxy_id: 3
        min_height: 0.0
        max_height: 10.0
        footprint:
          - [0.0, 0.0]
          - [10.0, 0.0]
          - [10.0, 8.0]
          - [0.0, 8.0]
```

### 7.2 必填要求

- `building_key` 必须是数据源内稳定、唯一、Unicode NFC 规范化后的字符串。
- `stage_index` 必须唯一且严格递增；预期阶段集合由数据集 manifest 给出。
- 坐标和高度必须为有限数值，禁止 NaN/Inf。
- `max_height > min_height`。
- footprint 至少包含三个有效非共线点。
- 输入坐标系必须已经解析完成。`local_to_first_vertex` 等坐标推断属于 upstream adapter，不允许在 canonical core 内隐式执行。

## 8. 输出合同

### 8.1 Canonical sequence

```yaml
schema_version: canonical_building_sequence_v1
canonicalizer_version: canonicalizer_v1
geometry_version: canonical_geometry_v1
geometry_config_hash: <sha256>
canonicalizer_config_hash: <sha256>
building_key: building_0001
building_uid: <sha256(building_key) 前 16 字节十六进制>
sequence_hash: <sha256>
stages:
  - stage_index: 1
    stage_key: stage_1
    stage_hash: <sha256>
    layers:
      - canonical_layer_index: 0
        layer_lineage_id: 0
        geometry_hash: <sha256>
        min_height_q: 0
        max_height_q: 10000
        footprint_q:
          - [0, 0]
          - [10000, 0]
          - [10000, 8000]
          - [0, 8000]
        point_lineage_ids: [0, 1, 2, 3]
provenance:
  source_paths: []
  source_hashes: []
  geos_version: <version>
  generator_git_commit: <commit>
warnings: []
```

核心文件必须保存整数网格坐标。供可视化或旧代码使用的 float 坐标由 adapter 通过 `q * grid` 生成，不作为 hash 来源。

### 8.2 Canonical edit v3

```yaml
schema_version: canonical_edit_v3
canonicalizer_version: canonicalizer_v1
geometry_version: canonical_geometry_v1
geometry_config_hash: <sha256>
canonicalizer_config_hash: <sha256>
source_stage_hash: <sha256>
target_stage_hash: <sha256>
edit_hash: <sha256>
layer_edits:
  - action: MODIFY_LAYER
    source_layer_index: 0
    target_layer_index: 0
    layer_lineage_id: 0
    source_height_q: [0, 10000]
    target_height_q: [0, 12000]
    point_edits:
      - action: KEEP_POINT
        point_lineage_id: 0
        source_index: 0
        target_index: 0
      - action: INSERT_POINT
        point_lineage_id: 4
        source_index: null
        target_index: 4
        target_coord_q: [5000, 10000]
      - action: EOS
```

## 9. Canonicalization pipeline

```text
validate provenance/frame
  -> integer quantization
  -> raw ring cleanup
  -> per-stage solid union
  -> canonical slab decomposition
  -> canonical ring + layer ordering
  -> stage hashing / no-op detection
  -> adjacent-stage global layer matching
  -> cyclic order-preserving point matching
  -> lineage assignment
  -> canonical edit ordering
  -> round-trip verification
  -> serialization + collision audit
```

### 9.1 数值量化

所有几何计算前先转换为整数：

```text
quantize(v, grid) = sign(v) * floor(abs(v) / grid + 0.5)
```

要求：

- 禁止依赖 Python `round()` 的 banker rounding。
- X/Z 使用 `grid_xz`，Y 使用 `grid_y`。
- 量化后再删除重复点、判断共线和生成 hash。
- 几何库操作必须启用 fixed precision/snap rounding；操作结果必须重新量化。
- 面积使用整数坐标的两倍有向面积 `area2`，体积比较使用整数面积与整数高度，避免 float 作为相等判断依据。

### 9.2 Raw ring cleanup

对每个 raw footprint 严格执行：

1. 删除尾部与首点相同的闭合点。
2. 删除连续重复点。
3. 循环删除位于相邻边之间的严格共线中间点，直到稳定。
4. 若少于 3 点或 `area2 == 0`，报错。
5. 若自交，报错；v1 不使用 `buffer(0)` 静默修复训练数据。
6. 若有洞，报错。
7. 若有多个 polygon 分量，保留所有分量并分别进入后续分解。

输入清理只允许移除不改变量化实体的冗余表示。任何改变实体面积的 repair 都不属于 canonicalization，必须回到上游修复。

### 9.3 唯一 ring 表示

对一个简单外环：

1. 用整数 `area2` 统一为 CCW。
2. 计算所有循环平移后的顶点序列。
3. 选择完整序列字典序最小者作为唯一表示。
4. 实现 SHOULD 使用 Booth 最小表示算法达到 O(n)，但结果必须等价于枚举所有 rotation 后取最小值。

```text
canonical_ring(P) = min_lex(rotations(make_ccw(P)))
```

不能只选择最小 `(x,z)` 顶点，因为可能存在多个相同最小点或量化并列；必须比较完整循环序列。

### 9.4 唯一实体分解

Canonical layer 不直接继承 raw layer。必须从 stage solid 重新构造：

1. 收集所有 raw layer 的 `min_height_q` 和 `max_height_q`，排序去重得到高度事件 `H`。
2. 对每个非空区间 `[H_i, H_{i+1})`，选择覆盖其中点的 raw footprints。
3. 在整数 fixed-precision 网格上计算这些 footprints 的平面 union。
4. 将 union 的每个 Polygon 分量拆成独立 slab cell。
5. 若 union 产生洞，报错；若产生 MultiPolygon，全部保留并按 canonical ring 排序。
6. 对相邻高度区间中 canonical ring 完全相同的 cell 做竖直合并。
7. 删除完全重复的 cell，并记录 raw overlap/duplicate warning。
8. 最终验证 canonical cells 两两体积交集为 0，且其 union 与 raw stage solid 完全相等。

这一过程保证原始重叠方式、冗余高度切分和 layer 枚举顺序不会进入监督标签。

### 9.5 Canonical layer 排序与 stage hash

尚未分配 lineage 前，stage 中的 layer 按下列 key 排序：

```text
(
  min_height_q,
  max_height_q,
  footprint_area2,
  canonical_ring_serialized
)
```

`geometry_hash` 由 `(min_height_q, max_height_q, canonical_ring)` 计算。

`stage_hash` 由以下内容的 canonical JSON 字节计算：

```text
geometry_version
geometry_config_hash
ordered list of canonical layer integer geometry
```

`raw_proxy_id`、路径、时间戳、lineage id 和 float 坐标不得进入 `stage_hash`。

### 9.6 Construction 单调性和 no-op

对每个相邻阶段 `S_i -> S_j`：

```text
removed = Solid(S_i) - Solid(S_j)
added   = Solid(S_j) - Solid(S_i)
```

- v1 要求 `volume(removed) == 0`，否则报 `E_CONSTRUCTION_REMOVAL`。
- `added == 0` 且 stage hash 相同，标记为 no-op。
- no-op stage metadata 可以保留，但默认不生成普通训练 pair。
- 缺失预期阶段必须报错，不能把阶段缺失解释为合法跳跃。

### 9.7 Layer matching

Lineage 必须通过整个 building sequence 的相邻阶段依次分配，而不是为每个任意 pair 单独匹配。

#### 候选边

source layer 与 target layer 满足以下条件时才建立候选：

- 二者有正的 3D 交集；并且
- `IoU3D >= min_iou_3d_q / 1e6`，或者 `intersection / min(source_volume, target_volume) >= min_smaller_coverage_q / 1e6`。

第二个条件用于覆盖“小体积在后续阶段扩展成大体积”的 construction 情况，避免仅因整体 IoU 较低而错误地产生 DELETE + INSERT。

construction-only v1 不把完全移动到不相交位置的 layer 视为同一 lineage；这种变化表示为 DELETE + INSERT。

#### 匹配目标

建立允许 unmatched dummy node 的全局二分图匹配。匹配质量按以下优先级比较：

1. 最大化总 3D intersection volume。
2. 最大化总 IoU3D。
3. 最小化总 symmetric-difference volume。
4. 最小化高度边界差。
5. 最小化 centroid distance。
6. 在所有同分最优解中，选择 source canonical index 顺序下 target index 向量字典序最小的解；unmatched 排在所有真实 target 之后。

实现不得依赖 Hungarian/SciPy 对平分情况的未定义返回顺序。建议先求最优总成本，再逐个 source 固定能保持全局最优的最小 target，得到确定性 lexicographic optimum。

#### 拆分与合并

- 一对多 split：只有全局最优的一条 target 继承 source lineage，其余 target 分配新 lineage。
- 多对一 merge：只有全局最优的一条 source lineage 被 target 继承，其余 source 产生 DELETE。
- 选择完全由上述全局目标和 tie-break 决定，禁止使用 raw proxy id 决胜。

#### 层 Lineage 分配

- 第一个非空 stage 按 canonical layer 顺序分配 `layer_lineage_id = 0..N-1`。
- matched target 继承 source lineage。
- unmatched target 按 target canonical order 分配单调递增的新 lineage。
- 已删除 lineage 永不复用。

### 9.8 Point matching

Point lineage 只在 matched layer lineage 内传播。

#### 初始分配

lineage 第一次出现时，按 canonical ring 顺序分配 `point_lineage_id = 0..N-1`。

#### 循环对齐

对 source 和 target 的 CCW canonical ring：

1. source 顺序固定。
2. 枚举 target 的所有循环 offset；不再枚举反向，因为方向已统一为 CCW。
3. 对每个 offset 做保持循环顺序的动态规划 alignment。
4. 允许操作：KEEP、MOVE、DELETE、INSERT。

匹配规则：

- 坐标完全相同：KEEP，成本 0。
- 距离不超过 `max(min_move_distance_q, max_move_distance_ratio * bbox_diagonal_q)`：可选 MOVE。
- 超过最大移动距离：禁止匹配，只能 DELETE + INSERT。

DP 解按下列 tuple 取最小：

```text
(
  delete_count + insert_count,
  move_count,
  sum_squared_move_distance_q,
  serialized_correspondence
)
```

`serialized_correspondence` 使用固定 action priority：

```text
KEEP < MOVE < DELETE < INSERT
```

并依次比较 source index、target index。它只用于确定性 tie-break。

如果没有可靠匹配，必须使用确定性的“source 全 DELETE，target 全 INSERT”，并记录 `W_POINT_ALIGNMENT_FALLBACK`，不能退化为按数组下标匹配。

#### 点 Lineage 分配

- KEEP/MOVE target 继承 source point lineage。
- INSERT target 按 target canonical index 分配该 layer lineage 内单调递增的新 point lineage。
- DELETE lineage 永不复用。

### 9.9 任意阶段 pair 的一致性

Lineage 只通过相邻阶段建立一次。对任意合法 `stage_i -> stage_j`：

- 两个阶段中存在相同 layer lineage：生成 KEEP/MODIFY。
- 只存在于 source：生成 DELETE。
- 只存在于 target：生成 INSERT。
- point action 同样依据已经传播的 point lineage 直接生成。
- 禁止为每个抽样 pair 重新运行 layer/point matcher；否则同一个 layer 在不同 pair 中可能得到不同身份。
- 禁止简单拼接相邻 edit 作为训练标签；必须直接比较 pair 两端的 canonical stage 和全序列 lineage，以得到唯一、无中间动作的 direct edit。

这条规则优先保证整个 sequence 内身份传递一致。因中间 split/merge 已退休的 lineage 不得在远距离 pair 中重新匹配或复用。

### 9.10 唯一 edit 排序

#### 层编辑顺序

1. 所有 source-backed action（KEEP/MODIFY/DELETE）按 `source_layer_index` 升序。
2. 所有 INSERT_LAYER 按 `target_layer_index` 升序。
3. 顶层 EOS。

#### 点编辑顺序

每个 matched layer 内：

1. KEEP/MOVE/DELETE 按 `source_index` 升序。
2. INSERT 按 `target_index` 升序。
3. EOS。

每条 action 必须携带显式 source/target index。执行器按 `target_index` 重建目标环，不能依赖 action 出现顺序推断目标下标。

### 9.11 Hash 和序列化

- 使用 UTF-8、Unicode NFC、无 BOM 的 canonical JSON 计算 hash。
- key 按字典序排列；数组保持规范顺序；禁止 NaN/Inf。
- hash 输入只使用整数坐标和规范字段。
- YAML 仅供人读；YAML 文本本身不是 hash 源。
- `edit_hash` 包含 source hash、target hash、`canonicalizer_config_hash` 和完整 canonical edit。
- `sequence_hash` 包含 building key、所有 stage hash、lineage 表和版本。

## 10. ID 策略和 v2 兼容 adapter

Canonical core 使用局部、单调、可解释的 lineage id，不使用几何 hash 直接作为 lineage：几何移动后 hash 会变，但 lineage 可能保持。

模型兼容 adapter MAY 生成当前整数 ID：

```text
proxy_id = dense_building_id * 1_000_000_000 + layer_lineage_id
point_id = dense_building_id * 1_000_000_000
         + layer_lineage_id * 100_000
         + point_lineage_id
```

要求：

- `dense_building_id` 是 dataset/sample/tile adapter 分配的局部 ID，不进入 canonical hash。
- `layer_lineage_id < 10_000`，`point_lineage_id < 100_000`；越界报错，禁止取模。
- 新的训练生成器不得再使用 `point_index` 生成跨阶段身份。
- raw `proxy_id` 和 raw `point_ids` 仅写入 provenance 映射，不能影响 edit 标签。

## 11. Condition 的确定性接口

Canonicalizer core 输出 canonical source/target solid。条件点生成是相邻模块，但必须遵守同一唯一性合同。

### 11.1 Condition geometry

```text
addition_solid  = target - source
removal_solid   = source - target
```

- v1 construction-only 要求 removal 为空。
- 条件 surface primitives 必须由 canonical solid delta 生成，不得从 raw proxy 分解生成。
- 如果 delta 为空，普通训练 pair 必须跳过；禁止用单个零点伪装有效变化。

### 11.2 Deterministic point sampling

如果需要固定 N 个点：

1. 按量化 surface area 使用 largest-remainder method 分配各 primitive 点数。
2. 使用由 `(source_stage_hash, target_stage_hash, condition_config_hash)` 派生的固定 seed；stage hash 已包含 geometry config。
3. 使用确定性低差异采样或固定实现的 stratified sampling。
4. 最终点量化并按 `(x_q,y_q,z_q)` 排序。
5. 下采样使用确定性 FPS，初始点固定为字典序最小点；禁止直接取数组前 N 点。

`condition_hash` 由量化后的有序点和 `condition_config_hash` 计算。

数据集必须检查：

```text
(source_stage_hash, condition_hash) -> exactly one target_stage_hash/edit_hash
```

同一输入 key 对应多个目标或 edit 时，报 `E_SUPERVISION_COLLISION`。

## 12. Validator 和错误策略

### 12.1 Hard errors

| 错误码 | 条件 |
|---|---|
| `E_COORDINATE_FRAME` | 坐标系未解析或阶段间不一致 |
| `E_NONFINITE_VALUE` | NaN/Inf |
| `E_INVALID_HEIGHT` | `max <= min` 或量化后高度塌缩 |
| `E_TOO_FEW_POINTS` | cleanup 后少于 3 点 |
| `E_SELF_INTERSECTION` | 量化环自交 |
| `E_HOLE_UNSUPPORTED` | union 产生洞 |
| `E_SOLID_MISMATCH` | canonical cells union 不等于 raw solid |
| `E_CANONICAL_OVERLAP` | canonical cells 有正体积交叠 |
| `E_MISSING_STAGE` | manifest 预期阶段缺失 |
| `E_CONSTRUCTION_REMOVAL` | 后一阶段未包含前一阶段 |
| `E_LAYER_CAPACITY` | canonical layer 数超过模型容量 |
| `E_POINT_CAPACITY` | 任一 canonical ring 超过点容量 |
| `E_BUILDING_CAPACITY` | 单 tile 的 building 数超过 adapter 容量 |
| `E_ID_OVERFLOW` | compatibility ID 超出 stride 合同 |
| `E_BUILDING_UID_COLLISION` | 截断后的 building UID 在同一数据集内冲突 |
| `E_ROUNDTRIP_MISMATCH` | apply edit 后 hash 不等于 target |
| `E_SUPERVISION_COLLISION` | 同一 source+condition 对应不同监督 |
| `E_NONDETERMINISTIC_OUTPUT` | 重跑或不同 worker 数得到不同 hash |
| `E_CONDITION_EMPTY` | 非训练 no-op 未产生 addition surface |
| `E_CONDITION_SAMPLING` | 去重后 condition 候选不足或点数合同不一致 |
| `E_NORMALIZATION_PROFILE` | train-only normalization 无几何或 grid 合同不一致 |
| `E_INPUT_ADAPTER` | Pilot 的显式 stage 路径或 YAML schema 无法读取 |

### 12.2 Warnings

| 警告码 | 条件 |
|---|---|
| `W_RAW_OVERLAP_CANONICALIZED` | raw layers 有正体积重叠，已通过 union 消除 |
| `W_RAW_DUPLICATE_LAYER` | raw layer 完全重复 |
| `W_QUANTIZATION_COLLAPSE` | 点或高度因量化合并，但实体仍有效 |
| `W_NOOP_STAGE` | 相邻 stage hash 相同 |
| `W_POINT_ALIGNMENT_FALLBACK` | 点匹配退化为 delete-then-insert |
| `W_LAYER_SPLIT` | 一对多 layer split |
| `W_LAYER_MERGE` | 多对一 layer merge |

正式数据集允许 warning，但 manifest 必须记录每类数量和样本路径。任何 hard error 都使该 building 失败；不得只跳过坏 layer 后继续。

## 13. API 设计

Phase 2 实际模块位置：

```text
gendiff_data_process/canonicalization/
  __init__.py
  config.py
  types.py
  errors.py
  quantize.py
  polygon.py
  solid_partition.py
  layer_matching.py
  point_matching.py
  core.py
  edit_v3.py
  collision.py
  condition.py
  release_contracts.py
  history_adapter.py
  pilot.py
  serialize.py
  adapters/area_v2.py
  packed_contract.py
```

核心纯函数接口：

```python
def canonicalize_stage(
    raw_stage: RawStage,
    bundle: CanonicalizerBundle,
) -> CanonicalStage:
    ...

def canonicalize_building_sequence(
    sequence: RawBuildingSequence,
    bundle: CanonicalizerBundle,
) -> CanonicalBuildingSequence:
    ...

def build_canonical_edit(
    source: CanonicalStage,
    target: CanonicalStage,
    cfg: CanonicalizerConfig,
) -> CanonicalEdit:
    ...

def apply_canonical_edit(
    source: CanonicalStage,
    edit: CanonicalEdit,
    cfg: CanonicalizerConfig,
) -> CanonicalStage:
    ...
```

核心函数必须无全局随机状态、无文件系统枚举依赖。文件读取、坐标 adapter 和并行调度放在 core 之外。

## 14. 后续 CLI 目标（Phase 2 未实现）

以下是 pilot 前需要另行审阅和实现的目标形状，不是当前可调用入口：

```bash
python -m gendiff_data_process.canonicalization.cli canonicalize-building \
  --input <building_dir> \
  --output <versioned_output_dir> \
  --config configs/canonicalizer_v1.yaml

python -m gendiff_data_process.canonicalization.cli validate-dataset \
  --input <canonical_dataset_dir> \
  --manifest <dataset_manifest.yaml> \
  --workers 56

python -m gendiff_data_process.canonicalization.cli compare \
  --left <canonical_sequence_a.yaml> \
  --right <canonical_sequence_b.yaml>

python -m gendiff_data_process.canonicalization.cli determinism-check \
  --input <raw_dataset_dir> \
  --workers 1,8,56 \
  --runs 3
```

CLI 必须输出：

- canonical files 到新目录，禁止覆盖 raw 数据。
- `dataset_manifest.yaml`。
- `validation_report.json` 和摘要 YAML。
- error/warning 的逐 building 明细。
- stage/action/capacity 分布。
- 输入文件 SHA256、代码 commit、GEOS/Shapely/Python 版本。

## 15. 与当前代码的集成

### 15.1 数据入口

- CityEngine/OBJ 转换器只负责产生 raw layer 和明确坐标 frame。
- `/mnt/d/data/obj_to_language` 可以作为 raw adapter，但不得直接输出训练标签。
- 必须找到并版本化当前 CityEngine YAML 实际生成入口。

### 15.2 `generate_area_sequence_v2_dataset.py`

应替换以下行为：

- 删除“缺少 `point_ids` 时按 point index 造 ID”的 fallback。
- 删除当前贪心 `_match_layers` 作为监督来源。
- 读取 canonical sequence 和 canonical edit v3。
- stage/pair 合法性依据 hash 和 monotonicity，不依据目录下标 alone。
- 先按 building 划分 train/val/test，再在 split 内枚举 area state/pair。
- 禁止把同一个 `pair_records` 写入三个 split。

### 15.3 Packed dataset

- packer 只接受通过 validator 的 canonical dataset manifest。
- `geometry_config_hash`、`canonicalizer_config_hash`、`validation_profile_hash`、`condition_config_hash`、`sequence_hash`、`source/target/edit/condition hash` 必须进入 packed metadata。
- 任何 layer/point/building 容量超限在 pack 前报错，不得截断。

### 15.4 Forward pipeline

- source scene 先走同一 canonicalizer。
- 网络输出应用 edit 后再次 canonicalize，再参与 tile merge。
- merge 后对完整场景运行 overlap、ID、capacity 和 hash 验证。
- `build_structure_tensors()` 不得截断，必须接收已经通过容量验证的数据。

### 15.5 v2 迁移

第一阶段保留 `adapters_v2.py`，把 canonical v3 映射到当前模型张量和动作 enum。canonical core 不应为了兼容 v2 的偶然顺序而改变规范结果。

## 16. 测试规范

### 16.1 Unit tests

| 测试 | 预期 |
|---|---|
| 环循环平移 | canonical ring/hash 相同 |
| 环反向 | canonical ring/hash 相同 |
| 重复闭合点 | 相同 |
| 冗余共线点 | 相同 |
| layer 顺序打乱 | stage hash 相同 |
| raw proxy id 改写 | stage/edit hash 相同 |
| 两层重叠与单层 union 等价 | stage hash 相同，前者有 warning |
| MultiPolygon | 所有分量保留并稳定排序 |
| hole | `E_HOLE_UNSUPPORTED` |
| 自交 | `E_SELF_INTERSECTION` |
| 同几何不同起点的跨阶段 ring | 全部 KEEP，不得出现假 MOVE |
| shared vertex 换数组下标 | shared vertex lineage 保持 |
| layer split/merge | primary lineage 选择稳定 |
| 量化格内扰动 | hash 相同 |
| 跨量化格变化 | hash 不同 |
| 施工删除 | 强制失败 |
| no-op | 标记并从普通 pair 排除 |
| 33 点且容量 32 | hard fail，不截断 |

### 16.2 Golden tests

必须把审计中发现的真实案例加入固定 fixture：

- `building_0097`
- `building_0099`
- `building_0112`
- `building_0299`
- `building_1500`
- 至少一个 raw overlap 严重样本，如 `building_0006`
- 至少一个阶段体积回退样本，如 `building_0007`

每个 fixture 保存预期 canonical hash、lineage、edit 和错误/警告码。只有显式升级 spec/version 才能更新 golden 输出。

### 16.3 Property-based tests

对随机有效 polygon/stage 自动施加：

- rotation、reverse、layer permutation、ID permutation。
- 等价重叠分解。
- 量化格内 jitter。
- 不同 worker/order 执行。

断言 canonical 字节和 hash 完全一致，并对生成 edit 做 round-trip。

### 16.4 Dataset acceptance tests

100 栋 pilot 和最终 1501 栋 release 必须满足：

- hard error = 0。
- 三次重跑、workers=1/8/56 的 sequence hash 100% 一致。
- 所有 edit round-trip 100% 通过。
- canonical layer 正体积重叠 = 0。
- train/val/test 的 building 重叠 = 0。
- 静默丢 layer/point/building = 0。
- `(source_hash, condition_hash)` 监督冲突 = 0。
- action 分布中，任务声明需要的动作均有明确非零覆盖；否则训练配置必须禁用相应能力声明。

## 17. Dataset manifest

正式输出必须包含：

```yaml
schema_version: canonical_dataset_manifest_v1
dataset_version: history_area_edit_v3_001
canonicalizer_version: canonicalizer_v1
geometry_version: canonical_geometry_v1
geometry_config_hash: <sha256>
canonicalizer_config_hash: <sha256>
validation_profile_hash: <sha256>
condition_config_hash: <sha256>
source_root: <path>
source_file_hash: <aggregate sha256>
generator_git_commit: <commit>
runtime:
  python: <version>
  shapely: <version>
  geos: <version>
building_count: 0
stage_count: 0
canonical_layer_count: 0
pair_count: 0
split:
  unit: building
  seed: 0
  train_buildings: []
  val_buildings: []
  test_buildings: []
validation:
  error_count: 0
  warning_counts: {}
hashes:
  sequence_hash_aggregate: <sha256>
  edit_hash_aggregate: <sha256>
  condition_hash_aggregate: <sha256>
```

## 18. 实现里程碑

### M1：几何核心

- 实现 config、整数 quantization、ring cleanup/canonical rotation。
- 实现 fixed-precision union、slab decomposition、canonical layer 排序和 hash。
- 完成基础 unit/property tests。

验收：所有单 stage 等价变换 hash 一致；canonical union 无重叠且无实体损失。

### M2：血缘与编辑 v3

- 实现确定性全局 layer matching。
- 实现 cyclic point alignment、lineage 分配和唯一 action 排序。
- 实现 edit executor 和 round-trip 检查。
- 加入真实 golden fixtures。

验收：已知换起点/反向案例不再产生假 MOVE；所有 golden pair round-trip 通过。

### M3：验证器、CLI 与 v2 适配器

- 实现错误码、报告、manifest、determinism-check。
- 接入当前 v2 tensor/action adapter。
- 修改 packer，去除所有静默截断。

验收：相同输入在 workers=1/8/56 下字节级一致；坏数据全部得到明确错误。

### M4：100 栋试点

- 从 1501 栋中选固定 100 栋，生成不覆盖原目录的 v3 数据。
- 审查 warning 分布和 layer/action/capacity 分布。
- 建筑级切分，完成小规模 overfit 与 held-out 基线。

验收：满足第 16.4 节全部 gate，才能处理剩余 1501 栋。

### M5：全量接入与前向流程

- 生成全量 versioned canonical dataset。
- 修改 area pair generator 和 forward pipeline 使用同一 core。
- 对现有大场景 case 产出 final YAML/OBJ、pipeline summary 和 GT 几何指标。

验收：训练/验证/forward 的 canonicalizer version 和 config hash 完全一致，无任何隐式 fallback。

## 19. 参考伪代码

```python
def canonicalize_building_sequence(raw, cfg):
    validate_input_contract(raw, cfg)
    geometry_stages = []

    for raw_stage in sort_stages(raw.stages):
        quantized = quantize_raw_layers(raw_stage.layers, cfg)
        clean_rings = [canonicalize_raw_ring(x, cfg) for x in quantized]
        cells = build_union_slab_cells(clean_rings, cfg)
        cells = merge_identical_vertical_cells(cells)
        stage = sort_and_hash_stage(cells, cfg)
        verify_stage_union(stage, clean_rings, cfg)
        geometry_stages.append(stage)

    validate_expected_stages(geometry_stages)
    validate_construction_monotonicity(geometry_stages)

    lineage = LineageState()
    first_stage = lineage.initialize_first_stage(geometry_stages[0])
    canonical_stages = [first_stage]
    adjacent_edits = []

    for target_geometry in geometry_stages[1:]:
        source = canonical_stages[-1]
        correspondence = deterministic_layer_match(source, target_geometry, cfg)
        correspondence = deterministic_point_match(
            source, target_geometry, correspondence, cfg
        )
        target_with_ids = lineage.assign(source, target_geometry, correspondence)
        edit = build_canonical_edit(source, target_with_ids, correspondence, cfg)
        assert hash(canonicalize_stage(apply_edit(source, edit), cfg)) == target_geometry.stage_hash
        canonical_stages.append(target_with_ids)
        adjacent_edits.append(edit)

    direct_pair_edits = build_direct_pair_edits_from_sequence_lineage(
        canonical_stages, lineage, cfg
    )
    audit_supervision_collisions(direct_pair_edits)
    return serialize_and_hash(canonical_stages, direct_pair_edits, cfg)
```

## 20. 完成定义

Unified Canonicalizer v1 只有在以下条件同时满足时才算完成：

1. spec 中的输入、输出、算法、tie-break 和错误码均已实现，没有未记录 fallback。
2. audit 中已发现的真实非唯一案例全部成为 golden tests。
3. 100 栋 pilot 达到 0 hard error、100% determinism、100% edit round-trip。
4. 同一实体的不同 raw proxy 分解得到相同 stage hash。
5. train、val、test 在 building 层级完全隔离。
6. 当前训练、packer 和 forward 均记录并校验同一个 `canonicalizer_version + canonicalizer_config_hash`，并记录各自使用的 validation/condition profile hash。
7. 超容量、洞、自交、拆除和无法解析坐标等情况都显式失败，绝不静默丢数据。

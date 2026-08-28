# Unified Canonicalizer v1 test plan

状态：Phase 1 计划，尚未实现

依据：`docs/handoff/GenDiff_unified_canonicalizer_spec_v1.md`、
`docs/TRAINING_CONSUMER_AUDIT.md`、
`catalog/training_consumer_manifest.yaml`

## 目的与边界

本计划定义进入 canonicalizer 实现、pilot regeneration 和 bounded overfit 前的可执行
门槛。它同时测试三层合同：

```text
raw building stages
  -> Unified Canonicalizer v1 core (integer geometry, lineage, canonical_edit_v3)
  -> reviewed v3-to-area-v2 adapter
  -> area_v2_packed_v1 loader tensors / model decode / apply
```

Phase 1 不创建 canonicalizer、不修改 `/mnt/d/projects/GenDiff`、不复制真实大数据、
不生成 pilot，也不训练。本计划中的所有测试状态均为 `planned`。

## 固定语义

测试实现不得重新解释以下规则：

- 坐标系为 world XZY：footprint 是 XZ，height 是 Y。
- core 在 geometry 操作前按 `grid_xz=grid_y=0.001` 量化，rounding 是
  `half_away_from_zero`，禁止 Python banker `round()`。
- canonical ring 统一 CCW、删除等价冗余点，并在所有 cyclic rotations 中取完整序列
  lexicographic minimum。
- canonical stage 是 raw extruded solids union 的确定性无重叠 slab decomposition；
  raw proxy ID、raw point ID、输入顺序和路径不得进入 stage hash。
- construction-only 必须满足 source solid 是 target solid 的子集；hole、自交、删除体积、
  容量超限和 ambiguity 按规范 hard fail。
- layer matching 是 deterministic global optimum；point matching 是 cyclic
  order-preserving alignment；无法可靠匹配时固定 delete-then-insert 并给 warning。
- direct pair edit 从全 sequence lineage 比较两端生成，不拼接中间 edit。
- canonical JSON 使用 UTF-8 NFC、sorted keys、整数坐标、无 NaN/Inf；相同输入必须
  byte-identical。
- 当前模型 adapter 的 MOVE/INSERT value 必须是 area-normalized XZ 中的**绝对 target
  coordinate**，不能输出 delta 或 anchor-relative value。

任何上述语义变化都必须升级 canonicalizer/config version，并更新 reviewed golden；
不能只改测试期望值。

## 建议测试布局

规范建议实现位于 GenDiff：

```text
craftsman/data/canonicalization/
  config.py
  quantize.py
  polygon.py
  solid_partition.py
  layer_matching.py
  point_matching.py
  lineage.py
  edit_v3.py
  adapters_v2.py
  serialize.py
  validate.py
```

测试和 fixtures：

```text
tests/canonicalization/
  test_quantize.py
  test_ring.py
  test_solid_partition.py
  test_lineage.py
  test_edit_roundtrip.py
  test_determinism.py
  test_collision_audit.py
  test_area_v2_adapter.py
  test_packed_loader_smoke.py
  test_forward_contract.py
  test_bounded_overfit.py

tests/fixtures/canonicalizer/
  synthetic/
  golden/
  adapter/
  packed_smoke/
```

实际 implementation location 需要 review 批准。若 core 最终放在数据仓库，GenDiff 只能
通过 versioned package/import 消费，不能复制一份分叉实现；测试 ID 和断言保持不变。

## Fixture 合同

每个 fixture 必须是小型、tracked、无绝对路径的数据包，并包含：

```yaml
fixture_id: SYN_RING_RECT_001
schema_version: raw_building_sequence_v1
coordinate_frame: world_xzy
canonicalizer_bundle: configs/canonicalizer_v1.yaml
expected:
  status: pass
  warning_codes: []
  error_code: null
  stage_hashes: unknown_pending_reviewed_reference
  edit_hashes: unknown_pending_reviewed_reference
```

`unknown_pending_reviewed_reference` 不能由待测实现自动写回。首次 golden hash 必须由
独立 reference calculation、round-trip 和人工 review 同时确认后显式固定。

### Synthetic fixtures

| ID | 最小内容 |
|---|---|
| `SYN_RING_RECT_001` | 单层非对称矩形，4 点，便于 rotation/reverse/closure 变换 |
| `SYN_RING_SYM_001` | 正方形及多个 lexicographic tie candidate |
| `SYN_QUANT_001` | 正负半格、半格两侧和跨格坐标 |
| `SYN_LAYER_PERM_001` | 两个高度 slab、raw layer/proxy ID 可置换 |
| `SYN_OVERLAP_001` | 两个重叠 raw layers 与单一等价 union 表示 |
| `SYN_MULTIPOLY_001` | 两个不连通分量，验证全部保留和稳定排序 |
| `SYN_HOLE_001` | union 产生 hole，预期 `E_HOLE_UNSUPPORTED` |
| `SYN_SELF_X_001` | bow-tie 自交，预期 `E_SELF_INTERSECTION` |
| `SYN_SPLIT_MERGE_001` | 相邻阶段一对多、多对一 lineage tie |
| `SYN_AMBIG_001` | 完全对称、多个同分 layer/point correspondence |
| `SYN_CONSTRUCTION_001` | 合法增加、no-op、体积删除三个 transition |
| `SYN_CAPACITY_001` | 64/65 层、32/33 点、16/17 building 边界 |
| `SYN_COLLISION_001` | 同 source+condition 的相同 target duplicate 与冲突 target |
| `SYN_ADAPTER_001` | KEEP/MOVE/DELETE/INSERT、空 source、insert layer 全覆盖 |

### Real golden fixtures

从 `/mnt/d/projects/GenDiff/datasets/history_stages_all_new` 只复制每个 case 所需的最小
stage YAML 到 tracked fixtures；不 symlink、不引用绝对路径、不修改源数据。来源和
SHA256 在复制时写入 fixture manifest。

| Fixture | 已知风险 | 预期 |
|---|---|---|
| `building_0097` | cyclic start/index shift | shared geometry 不产生假 MOVE |
| `building_0099` | winding/start difference | ring/hash invariant |
| `building_0112` | shared vertex index shift | point lineage 保持 |
| `building_0299` | 33-point layer | profile max 32 时 hard fail，不截断 |
| `building_1500` | boundary real case | deterministic reviewed result |
| `building_0006` | raw overlap | union 等价并给 overlap warning |
| `building_0007` | construction volume rollback | `E_CONSTRUCTION_REMOVAL` |

这些路径来自
`docs/handoff/GenDiff_unified_canonicalizer_spec_v1.md:723` 和现有 server audit；Phase 1
没有复制或读取这些 case。实施任务开始时先确认源文件可访问；不可访问时 fixture
source/hash 写 `unknown`，不得用相似 case 冒充。

## Test matrix

### Quantization、ring 和 stage geometry

| Test ID | Fixture / mutation | Required assertion | Gate |
|---|---|---|---|
| `CAN-Q-001` | `SYN_QUANT_001`，`n+0.5` 正负边界 | half-away-from-zero 精确整数；正负对称 | blocking |
| `CAN-Q-002` | 同一格内 `±epsilon` jitter | canonical bytes/stage hash 相同 | blocking |
| `CAN-Q-003` | 跨一个格边界 | integer geometry 和 stage hash 不同 | blocking |
| `CAN-R-001` | ring 所有 cyclic starts | canonical ring、bytes、hash 全相同 | blocking |
| `CAN-R-002` | CW/CCW reverse | canonical ring、bytes、hash 全相同 | blocking |
| `CAN-R-003` | duplicate close、consecutive duplicate、collinear insert | 相同实体得到相同 ring/hash | blocking |
| `CAN-R-004` | raw layer order、file enumeration、proxy ID permutation | stage bytes/hash 相同 | blocking |
| `CAN-R-005` | `SYN_RING_SYM_001` | 完整序列 lexicographic tie-break 固定，不依赖 library order | blocking |
| `CAN-S-001` | `SYN_OVERLAP_001` 两种等价 decomposition | stage hash 相同；重叠输入有 `W_RAW_OVERLAP_CANONICALIZED` | blocking |
| `CAN-S-002` | `SYN_MULTIPOLY_001` | 所有分量保留、稳定排序、0 volume overlap | blocking |
| `CAN-S-003` | `SYN_HOLE_001` | 精确 `E_HOLE_UNSUPPORTED`，无 partial output | blocking |
| `CAN-S-004` | `SYN_SELF_X_001` | 精确 `E_SELF_INTERSECTION`，不调用 silent `buffer(0)` repair | blocking |
| `CAN-S-005` | canonicalize canonical output | canonical bytes 和 hash 不变 | blocking |

### Lineage、edit 和 fail-closed

| Test ID | Fixture / mutation | Required assertion | Gate |
|---|---|---|---|
| `LIN-L-001` | layer input permutation | global matching assignment 和 lineage ID 不变 | blocking |
| `LIN-L-002` | symmetric equal-cost candidates | lexicographically minimal global optimum，重复 100 次一致 | blocking |
| `LIN-L-003` | split/merge | 只有规范 primary 继承 lineage，warning 稳定 | blocking |
| `LIN-P-001` | same polygon with shifted start | 所有 shared point 为 KEEP，0 false MOVE/INSERT/DELETE | blocking |
| `LIN-P-002` | reversed source ring | canonicalization 后 lineage/edit 与原 case 相同 | blocking |
| `LIN-P-003` | reliable small MOVE | lineage 保持，MOVE value 为 quantized target coordinate | blocking |
| `LIN-P-004` | move 超阈值 | deterministic DELETE then INSERT + fallback warning | blocking |
| `LIN-P-005` | `SYN_AMBIG_001` 未满足可靠匹配 | fail closed 或规范 fallback，不按数组下标猜身份 | blocking |
| `LIN-ID-001` | raw proxy/point IDs 全量改写 | stage/edit hash 与 lineage 不变；provenance 映射变化允许 | blocking |
| `LIN-ID-002` | insert/delete 后再新增 | retired layer/point lineage 永不复用 | blocking |
| `EDIT-O-001` | action 混合 | source-backed actions 按 source index，insert 按 target index，EOS 唯一 | blocking |
| `EDIT-O-002` | direct stage 0→3 | 直接比较端点 lineage，不等于简单拼接中间 token | blocking |
| `EDIT-RT-001` | 所有 valid synthetic/golden pairs | `hash(canonicalize(apply(src, edit))) == target_stage_hash` 100% | blocking |
| `EDIT-RT-002` | malformed index/value/action | explicit error code，无 silent clamp/drop | blocking |

### Validation、determinism 和 collision

| Test ID | Fixture / mutation | Required assertion | Gate |
|---|---|---|---|
| `VAL-C-001` | legal construction increase | pass，removed volume exactly 0 | blocking |
| `VAL-C-002` | no-op | `W_NOOP_STAGE`，普通 pair manifest 不包含该 pair | blocking |
| `VAL-C-003` | source volume removed | `E_CONSTRUCTION_REMOVAL` | blocking |
| `VAL-CAP-001` | 64 vs 65 layers | 64 pass；65 `E_LAYER_CAPACITY`，无 slice | blocking |
| `VAL-CAP-002` | 32 vs 33 points | 32 pass；33 `E_POINT_CAPACITY`，无 slice | blocking |
| `VAL-CAP-003` | compatibility integer ID boundary | stride 上界前 pass；越界 `E_ID_OVERFLOW` | blocking |
| `DET-B-001` | same input 3 runs | canonical JSON/YAML-adjacent outputs byte-identical | blocking |
| `DET-B-002` | workers 1/2/8，reversed file order | all sequence/edit/condition hashes identical | blocking |
| `DET-B-003` | Python hash seed 0/1/9876 | bytes/hashes identical | blocking |
| `DET-CFG-001` | core geometry config change | corresponding config/stage hash changes | blocking |
| `DET-CFG-002` | validation capacity-only change | validation profile hash changes，stage hash 不变 | blocking |
| `COL-001` | exact duplicate canonical key and same target/edit | dedupe or report duplicate，不能产生两个训练 rows | blocking |
| `COL-002` | same source+condition, conflicting target | `E_SUPERVISION_COLLISION`，dataset build fails | blocking |
| `COL-003` | truncated building UID collision | `E_BUILDING_UID_COLLISION` | blocking |

## Area-v2 adapter contract tests

adapter 是当前训练 consumer 的唯一兼容边界；core 不得为适配旧 slot 顺序而改变 hash。

### Mapping contract

1. 输入必须包含 reviewed `canonicalizer_version`、`geometry_version`、四类 config hash、
   source/target stage hash、edit hash、condition hash 和 building-level split identity。
2. adapter 先按 grid 将 integer canonical geometry dequantize 到 world coordinates，再按
   frozen area/tile normalization stats 转为 float model coordinates。
3. layer slot 按 current loader 规则明确分配：source-backed edit 使用 source slot，insert
   使用 source layer count 后的新 slot；原 canonical target index 仍保存在 metadata。
4. point action 保留 canonical order。KEEP/MOVE/DELETE 写合法 source index；
   KEEP/MOVE/INSERT 写合法 target index；EOS 恰好一个。
5. MOVE/INSERT 的 `target_coord` 和 `value` 都是 normalized absolute target XZ；DELETE/KEEP
   value 是 `[0,0]`。禁止 anchor、delta 或 `_safe_index()` clamp 修复坏输入。
6. adapter 输出 `area_v2_absolute_target_coord_no_anchor`，packer 输出
   `area_v2_packed_v1`；错误 schema 必须 hard fail，不能只 warning。
7. packed metadata 必须携带 canonical hashes、producer commit/diff hash、command/config、
   runtime 与 validator report，loader smoke 必须校验这些字段。

| Test ID | Fixture | Required assertion | Gate |
|---|---|---|---|
| `ADP-S-001` | `SYN_ADAPTER_001` | exact loose edit schema/required fields/actions | blocking |
| `ADP-V-001` | MOVE/INSERT | values equal normalized absolute target coords, not delta/anchor | blocking |
| `ADP-I-001` | permuted layer/point input | slots/index targets deterministic and in range | blocking |
| `ADP-I-002` | index = 32 under max 32 | explicit capacity error before target builder; no clamp to 31 | blocking |
| `ADP-H-001` | canonical metadata | all version/config/source/target/edit/condition hashes survive pack | blocking |
| `ADP-P-001` | minimal loose dataset | pack → load preserves every action/index/value/mask exactly | blocking |
| `ADP-P-002` | missing/wrong edit schema | packer and loader both fail, no warning-only output | blocking |
| `ADP-RT-001` | adapter output through current applier | denormalize + canonicalize equals target hash | blocking |

## Loader smoke and split tests

`packed_smoke/` 应只含 3 buildings、每个 2 stages、最多 8 pairs、每 shard 最多 4
samples。train/val/test 按 building 分开，不能复用 shard。

| Test ID | Required assertion | Pass threshold |
|---|---|---|
| `LDR-001` | DataModule `setup(fit/test)` 和 `num_workers=0` 各取一 batch | no exception |
| `LDR-002` | 所有 batch key dtype/shape 与 audit 合同一致 | exact match |
| `LDR-003` | masks、index range、EOS、action/value masks | 100% valid |
| `LDR-004` | NaN/Inf、missing hash、wrong schema | explicit failure |
| `LDR-005` | 65 layer、33 point、17 building fixture | explicit capacity failure，0 truncation |
| `LDR-006` | split building IDs 和 shard paths | pairwise intersection = empty |
| `LDR-007` | repeated DataLoader run，workers 0/2 | ordered sample keys identical when shuffle disabled |
| `LDR-008` | canonical key uniqueness | conflict count = 0，duplicate rows = 0 |

loader smoke 只证明 ingestion/tensorization，不等于模型可学性。

## Forward contract tests

当前 untracked forward 使用 per-tile normalization 和 prefix truncation，必须先用 tests
固定改进后的行为：

| Test ID | Required assertion | Gate |
|---|---|---|
| `FWD-N-001` | training adapter 与 forward 对同一 tile 使用同一 normalization API/config hash | blocking |
| `FWD-N-002` | normalize → model identity edit → denormalize | world coordinates within one grid unit |
| `FWD-CAP-001` | 33-point/65-layer/17-building tile | fail or deterministic split；0 slice/drop | blocking |
| `FWD-COND-001` | condition permutation | canonical condition points/hash and model input identical | blocking |
| `FWD-M-001` | two adjacent tiles with shared context | owned building exactly once，context change not leaked | blocking |
| `FWD-M-002` | inserted layer/proxy ID conflict | deterministic unique ID and preserved lineage mapping | blocking |
| `FWD-RT-001` | GT edit through tile merge | final canonical scene hash equals GT | blocking |

Phase 2 不需要跑完整 1501-building case；这些测试使用 synthetic 2-tile fixture。

## Bounded overfit gate

只有前述 blocking tests 全绿后才允许运行。固定一个 tracked release：

```text
dataset: canonicalizer_overfit_v1
buildings: 1-5, building-level isolated
samples: 4-16 deterministic pairs
actions: KEEP, MODIFY, INSERT, MOVE, DELETE_POINT, EOS all non-zero
max capacity: <= 16 layers, <= 8 points/layer, <= 5 buildings
condition: canonical deterministic 256 or 2048 points, fixed config hash
seed: 0
training steps: max 2000
validation/test: not aliases; overfit reporting uses train only
```

Pass/fail threshold：

- teacher-forced layer action、AR action、source index、target index、length accuracy 都为
  1.000 on all samples；
- normalized point value MAE `<= 1e-4`，height error `<= 1e-4`；
- free-run decode/apply/canonicalize round-trip `16/16` 或实际 sample count 的 100%；
- invalid polygon、canonical overlap、silent drop、NaN/Inf、hash mismatch 均为 0；
- 同 seed 重跑两次，sample order、initial config、final decoded canonical hashes 完全一致。

loss 下降本身不算通过；必须以 free-run executable edit 和 target canonical hash 为准。

## Commands for the implementation task

在批准且环境固定后，建议按以下顺序执行。`<approved-python>` 当前为 `unknown`；
Phase 1 只确认 `/mnt/d/anaconda3/envs/gendiff/bin/python` 能导入 torch 2.4.0，不能把它
自动当作 release environment。

```bash
<approved-python> -m pytest -q tests/canonicalization/test_quantize.py tests/canonicalization/test_ring.py tests/canonicalization/test_solid_partition.py
<approved-python> -m pytest -q tests/canonicalization/test_lineage.py tests/canonicalization/test_edit_roundtrip.py
<approved-python> -m pytest -q tests/canonicalization/test_determinism.py tests/canonicalization/test_collision_audit.py
<approved-python> -m pytest -q tests/canonicalization/test_area_v2_adapter.py tests/canonicalization/test_packed_loader_smoke.py
<approved-python> -m pytest -q tests/canonicalization/test_forward_contract.py
```

小 fixture determinism CLI：

```bash
<approved-python> -m craftsman.data.canonicalization.cli determinism-check \
  --input tests/fixtures/canonicalizer/synthetic \
  --workers 1,2,8 \
  --runs 3
```

bounded overfit 必须使用独立 config，例如
`configs/canonicalizer_overfit_v1.yaml`，并记录 exact command、clean commit、config hash、
environment 和 output manifest。建议命令形状：

```bash
<approved-python> train.py \
  --config configs/canonicalizer_overfit_v1.yaml \
  --train \
  --gpu 0 \
  trainer.max_steps=2000
```

以上命令均未在 Phase 1 执行。

## Acceptance report

每次 test run 输出一个 tracked small report 或 CI artifact，至少包含：

```yaml
schema_version: canonicalizer_test_report_v1
canonicalizer_version: canonicalizer_v1
geometry_config_hash: <sha256>
canonicalizer_config_hash: <sha256>
validation_profile_hash: <sha256>
condition_config_hash: <sha256>
git_commit: <sha>
dirty: false
command: [<argv>]
runtime: {python: <version>, torch: <version>, shapely: <version>, geos: <version>}
tests: {passed: 0, failed: 0, skipped: 0}
determinism: {runs: 3, worker_counts: [1, 2, 8], byte_mismatch_count: 0}
round_trip: {checked: 0, failed: 0}
collisions: {duplicate_keys: 0, conflicting_targets: 0}
split_overlap: {train_val: 0, train_test: 0, val_test: 0}
silent_drop_count: 0
```

任何 `unknown` hash、dirty unrecorded diff、skipped blocking test、collision、round-trip
failure 或 silent drop 都使 gate 失败。

## Exact next gate

下一任务先由 reviewer 批准本测试语义和 adapter 边界，然后实现 fixtures 和 tests，让
现有 candidate 的已知不兼容点以明确失败呈现；之后才实现 canonical core 与 adapter。
必须先达到 unit/golden/property/adapter/loader smoke 全绿，才可生成一个 versioned small
pilot。pilot 验证通过后才可执行 bounded overfit；bounded overfit 通过后仍需再次 review，
才可讨论 bulk regeneration。Phase 1 在此停止。

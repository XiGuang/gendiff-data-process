# Canonicalizer Pilot 冻结合同

日期：2026-08-29

状态：历史 construction-only 合同。commit
`ca2a1ecb1e851f56506de9437d8e3598d9bc6efe` 的 100-building pilot 已执行并按合同标为
FAIL；本文不授权训练。

施工/拆除双向扩展见 `docs/CANONICALIZER_BIDIRECTIONAL_CONTRACT.md`。新合同使用独立
配置 `configs/canonicalizer_bidirectional_v1.yaml`，不得覆盖本合同的 config、hash 或
artifact。

## 1. Normalization

配置来源：`configs/canonicalizer_v1.yaml` 的 `normalization`。

- 方法：`train_bbox_uniform_v1`。
- 统计范围：仅 successful train buildings 的全部 canonical stages；val/test 和失败 building
  不得参与。
- 统计输入：整数 canonical XZ 顶点和 Y 高度 extrema，不从 float adapter 输出反推。
- 中心：各轴 `(min_q + max_q) / 2 * grid`。
- 统一尺度：`max(span_x_q, span_z_q, span_y_q, min_scale_q) * grid`，保证
  `scale_xz == scale_y`。
- Profile ID：method config hash、排序后的 train building UID、整数 extrema、grid 和 scale
  的 canonical SHA256 前 16 位。
- 完整 profile、train UID 清单和 hash 写入 run artifact；同一 profile 用于 train/val/test。

该算法与现有 GenDiff area-v2 的 bbox center/uniform scale 形式兼容，但消除了 per-target、
per-sequence 或 val/test 统计泄漏。

## 2. Building Split

配置来源：`configs/canonicalizer_v1.yaml` 的 `split`。

```text
building_uid = SHA256(NFC(building_key))[:32 hex]
bucket = uint64_be(SHA256("sha256_threshold_v1\0<seed>\0<building_uid>")[:8]) % 10000
train: bucket < 8000
val:   8000 <= bucket < 9000
test:  9000 <= bucket < 10000
```

Seed 固定为 `canonicalizer_split_20260829_v1`。Split 在读取 stage 内容前按 building
identity 分配；失败 building 仍保留原 split 和错误记录。任何 building UID 跨 split
出现都使 pilot 失败。

## 3. Condition Sampler

配置来源：`configs/canonicalizer_v1.yaml` 的 `condition_sampling`。

1. 从 canonical target solid 减 source solid 得到 addition solid；任何 removal hard fail。
2. 按 height events 构造 addition slices，并提取水平/垂直 surface primitives。
3. 按整数化 surface area 使用 largest-remainder 分配 `2 * 2048` 个候选。
4. 每个 primitive 使用由 source hash、target hash、condition config hash 和 primitive
   descriptor 派生 offset 的 Halton `(2,3)` 序列采样。
5. 候选量化、去重、字典序排序；以字典序最小点作为 FPS 起点，固定取 2048 点。
6. 最终点按 `(x_q, y_q, z_q)` 排序，连同 condition config hash 计算 `condition_hash`。
7. No-op pair 显式登记并跳过；禁止写单个零点。候选不足报 `E_CONDITION_SAMPLING`。

## 4. Package Pin

- Distribution：`gendiff-data-process==0.1.0`；import：`gendiff_data_process`。
- 从审阅后 clean Git commit 使用 `pip wheel --no-deps --no-build-isolation` 构建 wheel，
  不安装依赖。
- Run manifest 同时记录 repository URL、完整 commit、clean status、wheel 文件名和 SHA256。
- `config.yaml` 复制到 run artifact，记录源路径与副本 SHA256；validator 从副本加载，并核对
  源配置、七个子合同 hash 和 artifact 副本一致。
- GenDiff 后续只能固定 wheel SHA256 或同一 Git commit；禁止复制 canonicalizer 源码形成分叉。
- Pilot 不修改 `/mnt/d/projects/GenDiff`。

## 5. 黄金 Hash Gate

固定结果见 `tests/fixtures/canonicalizer/golden/reviewed_hashes.yaml`。审阅路径包括：

- 生产 core 输出；
- 不调用生产 `solid_partition/apply/serialize` 的独立 Shapely solid symmetric-difference、
  edit target 重建和标准库 JSON SHA256 oracle；
- source manifest 中 11 个原始文件 SHA256；
- 不同 worker/Python hash seed 的 fingerprint。

`0006/0097/0112` 固定 stage/edit/sequence hash；`0007/0099/0299/1500` 固定 hard error。
本轮是用户授权的 Codex 双路径审阅，未冒充外部人工审阅；manifest 中
`human_external_review` 明确为 `not_performed`。

## 6. 正式 Pilot 范围

- 输入：`/mnt/d/projects/GenDiff/datasets/history_stages_all_new`，只读。
- 选择：构造 `building_0001` 到 `building_0100`，不枚举或扫描其余 building。
- Stage：每栋只读取明确的 `stage_0` 到 `stage_3` YAML；每个文件记录 size 和 SHA256。
- 上限：100 buildings、400 个小 YAML；不读取 OBJ/JPG/log，不训练。
- 输出：`/mnt/d/artifacts/gendiff-data-process/runs/<run_id>`，目录必须事先不存在，拒绝覆盖。
- Packed consumer：`area_v2_packed_v1`，train/val/test 使用互斥 shard。

每个 pair 必须满足以下核算式：

```text
attempted = emitted + noop_skipped + duplicates_deduplicated + explicit_failures
silent_drop_count = 0
```

若某 building 在 canonicalize/condition 阶段 hard fail，该 building 的全部选定 transition
槽位计入 `failed_building_pair_slots`；冲突监督行计入 `collision_conflicting_rows`。二者均属于
`explicit_failures`，不得从 `attempted` 中移除。

任何 building hard error、determinism mismatch、collision、split overlap、hash mismatch、
loader failure、silent drop 或空 split 都使正式 pilot 标为 FAIL。即使 FAIL，也保留新 artifact
中的 manifest、明确失败清单和 partial diagnostic packed 数据；不得通过筛掉失败 building
把结果伪装成 PASS。

## 7. 命令合同

提交后构建 wheel：

```bash
/mnt/d/anaconda3/envs/gendiff/bin/python -m pip wheel . \
  --no-deps --no-build-isolation --wheel-dir <temporary-wheel-dir>
```

正式生成与验证：

```bash
/mnt/d/anaconda3/envs/gendiff/bin/python tools/build_canonicalizer_pilot.py \
  --dataset-root /mnt/d/projects/GenDiff/datasets/history_stages_all_new \
  --config configs/canonicalizer_v1.yaml \
  --building-start 1 --building-count 100 --stage-indices 0,1,2,3 \
  --determinism-workers 1,2,8 \
  --producer-commit <clean-commit> --package-wheel <wheel> --run-root <new-run-root>

/mnt/d/anaconda3/envs/gendiff/bin/python tools/validate_canonicalizer_pilot.py \
  --run-root <run-root> --gendiff-repo /mnt/d/projects/GenDiff \
  --hash-seed-workers 0:1,1:2,9876:8
```

Validator 必须遍历这个 bounded pilot 的全部 packed samples，并由真实 GenDiff loader 在
`num_workers=0` 下读取全部 train/val/test batch。它不调用模型 forward 或 trainer。

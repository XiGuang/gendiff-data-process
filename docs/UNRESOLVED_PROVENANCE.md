# 未解决数据血缘与风险登记

## P0：阻塞可信训练结论

- 已观测 GenDiff consumer 现已记录在
  `catalog/training_consumer_manifest.yaml`：已检查 checkout commit
  `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`、loader
  `/mnt/d/projects/GenDiff/craftsman/data/packed_area_edit_v2_data_module.py`,
  run command/config，以及所需 `area_v2_packed_v1` /
  `area_v2_absolute_target_coord_no_anchor` schema 均已知。历史 run commit、
  dirty diff hash、精确 Python environment、packed 数据生成 command 和 producer
  commit 仍为 `unknown`。
- 已观测 packed release 不是有效的 held-out release：train、validation 和 test
  alias 同一批 20,000 个样本。证据见 `docs/TRAINING_CONSUMER_AUDIT.md` 和
  manifest 证据 ID `E14`/`E15`。
- 本仓库的旧 canonical construction candidate 与已观测 loader 不直接兼容。Phase 2E
  已增加有版本的 `canonical_edit_v3 -> area_v2_absolute_target_coord_no_anchor`
  candidate adapter；双向正式 pilot 的 382 条 packed sample 已被真实 loader 全量读取。
  旧 producer 仍未迁移，GenDiff 也未固定消费该 wheel；原有 12 项历史审计 mismatch 仍阻塞
  “旧 producer 已兼容”的结论。
- Canonical edit 唯一性已在 synthetic、七个裁剪黄金案例和 construction-only
  100-building pilot 中做有界测量；该 pilot 的 122 emitted pair 均通过 round-trip、
  collision、determinism 和真实 loader 检查，但 generation 因 40 栋 hard failure 标为
  FAIL。全数据分布仍未测量。
- 对同一 100 栋做的只读 transition 分类得到 187 个纯新增、4 个纯删除、58 个 no-op、
  48 个 mixed，另有 `building_0032` 的 hole。原 39 个 removal failure building 全都包含
  mixed；不能通过把所有 removal 视为 demolition 关闭问题。证据和命令见
  `docs/CANONICALIZER_BIDIRECTIONAL_CONTRACT.md`。
- 双向 candidate 已实现纯 construction/demolition 的正反 pair、方向化 condition、
  `change_kind`/`pair_hash` 和 pair 级失败核算。正式 pilot 固定在 clean commit
  `f0de8c4de1cfe3f666d5f466de998c635ebdae0d`；382 条 sample 的 split、collision、
  determinism 和真实 loader 检查通过，但 generation 因 96 个有向 mixed failure 和
  `building_0032` 的 hole 标为 FAIL。48 个 mixed transition 的业务语义和修复来源仍为
  `unknown`，不得筛掉后提升 release。
- GenDiff 当前 loader 不把 `change_kind` 作为独立模型输入；XYZ + source 是否足以稳定
  区分施工和拆除仍为 `unknown`，需要后续训练实验，当前未授权。
- `/mnt/d/anaconda3/envs/gendiff/bin/python` 可运行 bounded canonicalizer、wheel build 和
  loader preflight，但它与历史训练环境的关系仍为 `unknown`，不能据此重建历史 run。

## P1：阻塞可靠再生成

- 几乎所有 legacy 数据集都缺少精确 command、config、seed 和 producer commit。
- 当前 CityEngine stage YAML producer 及其不可变 source/config 为 `unknown`；
  本地 `obj_to_language` schema 不能作为该关系的证据。
- `data/images` 和 `data/latents` 很可能跨仓库，其 producer 为 unknown。
- 两套 polygon proxy 实现尚未进行 schema 或几何等价测试。
- real-data test 使用被 ignore 的本地路径，没有小型 versioned fixture。
- 多个一次性脚本包含硬编码 `/mnt/d/...` 路径。
- 历史生成 YAML pair index 包含绝对 `/mnt/d/data/...` 路径；`data.yaml` 与
  `config/yuehai_with_remove.yaml` 字节完全相同。
- `requirements.txt`/`requirements_pip.txt` 是历史 environment export，
  `pyproject.toml`/`uv.lock` 则描述更窄的代码环境。

## P2：存储与仓库卫生

- legacy `.git` 目录约 18 GB，包含约 17.09 GiB 临时 pack。在完成备份和 fresh
  clone 验证前，它被有意保留而未删除。
- 历史输出混合了 candidate run、diagnostic 和 scratch 数据。它们已在
  `catalog/legacy_outputs.yaml` 中登记，但尚未分配不可变 run ID。candidate output
  consumer 已显式写为 `unknown`；已观测 packed GenDiff run 不能证明其消费这些目录。
- Dataset manifest 记录的是表观字节数和文件元数据，不是内容 hash；对全部 3.59 TB
  做 hash 属于另一项计划任务。

## 关闭问题所需证据

pipeline/dataset 关系要达到 high confidence，必须附带：

1. 不可变 producer commit 和 clean worktree 证明；
2. 精确 command/config/seed/environment；
3. 不可变输入 dataset ID 和内容 hash；
4. 输出数量、大小和 hash；
5. validation report，包括 canonical collision 和 round-trip 结果；
6. 下游 consumer commit/config 及成功的 smoke/overfit 结果。

未知字段必须保持 `unknown`；目录名和 mtime 只能作为辅助线索，不能作为证明。

## 已确认、不再未知的关系

- 已观测 run 到 command/config：high confidence，manifest evidence `E04/E08`。
- 已观测 run 到选定 consumer 文件字节：high confidence，`E05`。
- Packed dataset 到 source-stage 路径和已记录生成参数：high confidence，`E14`。
- Candidate producer 到已观测训练 run：仍为 `unknown`，`E26/E27`。

这些关闭结论仅适用于已检查的路径和字节，既不提升任何 pipeline，也不能重建缺失的
历史 run manifest。

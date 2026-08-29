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
  candidate adapter，但尚未完成旧 producer 迁移、GenDiff 集成或 pilot；原有
  normalization/schema 差异和 12 项审计 mismatch 仍阻塞正式 producer-to-consumer 结论。
- Canonical edit 唯一性已在 synthetic、七个裁剪黄金案例和 5-building preflight 中做
  有界测量；完整 100-building pilot 和全数据分布仍未测量。
- Canonical candidate 尚无全数据集 round-trip、collision、ambiguity 或
  duplicate-key 报告。
- Phase 2A 到 2E 及用户批准的 normalization/split/condition/package 合同已有实现和
  小型 preflight，但不能替代正式 100-building pilot。冻结算法见
  `docs/CANONICALIZER_PILOT_CONTRACT.md`。
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

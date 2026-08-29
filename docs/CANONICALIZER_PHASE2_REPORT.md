# Canonicalizer Phase 2A 到 2E 验收报告

日期：2026-08-29

状态：**candidate 范围通过；pilot 已获授权但尚未执行，release 和训练门槛未通过。**

## 结论

Unified Canonicalizer v1 的唯一 candidate 实现已放在
`gendiff_data_process/canonicalization/`，没有向 `/mnt/d/projects/GenDiff` 写入代码。
本轮完成了版本化配置、整数几何 core、sequence lineage、`canonical_edit_v3`、可执行
round-trip、area-v2 adapter、packed metadata 前置校验和真实 GenDiff loader 的小型冒烟
测试。

这只证明 2A 到 2E 的有界 candidate 合同成立，不证明真实 release 可生成、完整数据
唯一、模型可学或 forward 可用。没有 pipeline 或 dataset 被提升为 `current`。

机器可读结果见 `catalog/canonicalizer_phase2_test_report.yaml`。

## 分阶段验收

| 阶段 | 结果 | 已完成内容 | 证据 |
|---|---|---|---|
| 2A | PASS | 冻结代码归属、world XZY、1 mm half-away-from-zero、hash 分层、失败关闭策略 | `docs/CANONICALIZER_PHASE2_DECISIONS.md`；`configs/canonicalizer_v1.yaml` |
| 2B | PASS | 建立合成 fixture 和 7 个真实风险的最小裁剪 fixture；11 个来源文件 SHA256 复核一致 | `tests/fixtures/canonicalizer/`；`golden/source_manifest.yaml` |
| 2C | PASS | 实现 union/slab、ring 唯一化、layer/point matching、lineage、edit/apply、condition、collision audit | `gendiff_data_process/canonicalization/`；53 个定向测试通过 |
| 2D | PASS | 显式 normalization 输入、绝对 target value、容量/ID 失败关闭和 canonical metadata | `adapters/area_v2.py`；`packed_contract.py` |
| 2E | PASS | 临时目录构造 3-building、互斥 split packed fixture，并由实际 GenDiff loader 各读取一批 | `tests/canonicalization/test_packed_loader_smoke.py`；2 个测试通过 |

PASS 仅适用于上述 bounded candidate 范围。黄金 fixture 是
`selected_layers_only`，不冒充完整 building；`captured_at` 因无法证明精确复制时刻而保持
`unknown`，当前 SHA 复核时间单独记录为 `2026-08-29T00:55:46+08:00`。

## 实现合同

- Geometry hash 只包含版本化几何配置和整数 XZ/Y；raw ID、路径、输入排列和时间戳不参与。
- Canonical stage 来自 fixed-grid polygon union 和 height-event slab decomposition；hole、
  自交、实体不一致、正体积交叠和容量超限 hard fail。
- Layer matching 使用 Python 整数多目标 assignment 与显式 lexicographic tie-break；point
  matching 使用 cyclic order-preserving DP，无可靠匹配时 delete-then-insert 并 warning。
- Sequence 显式维护不复用的 layer/point lineage；每个相邻 edit 都先执行并比对 target
  stage hash。
- Adapter 不从 target 推断 normalization；MOVE/INSERT value 是 normalized absolute
  target XZ，不是 delta 或 anchor。64/32/16 容量边界和 ID stride 在 tensorization 前检查。
- GenDiff loader 保持只读。冒烟测试只在系统临时目录写入最多三个 building 的 PT 文件。

## 验证证据

Core、property、黄金和 adapter：

```bash
/mnt/d/anaconda3/envs/gendiff/bin/python -m unittest \
  tests.canonicalization.test_quantize_ring \
  tests.canonicalization.test_solid_partition \
  tests.canonicalization.test_lineage_edit \
  tests.canonicalization.test_golden_cases \
  tests.canonicalization.test_determinism_collision \
  tests.canonicalization.test_release_contracts \
  tests.canonicalization.test_condition \
  tests.canonicalization.test_reviewed_golden \
  tests.canonicalization.test_pilot \
  tests.canonicalization.test_area_v2_adapter
```

结果：53 passed，0 failed，0 skipped，1.693 秒。

真实 loader gate：

```bash
GENDIFF_REPO=/mnt/d/projects/GenDiff \
  /mnt/d/anaconda3/envs/gendiff/bin/python -m unittest \
  tests.canonicalization.test_packed_loader_smoke
```

结果：2 passed，0 failed，0 skipped，4.169 秒。读取的 loader 为
`/mnt/d/projects/GenDiff/craftsman/data/packed_area_edit_v2_data_module.py`，训练仓库 commit
为 `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`。其 dirty 路径与 Phase 1 manifest 完全
一致；本轮没有修改这些路径。

仓库级结构化校验结果：129 个 catalog YAML 和 16 个 config/fixture YAML 可解析；103 个
Python 文件与 `catalog/code_inventory.yaml` 一一对应并用 `tokenize.open()` 加 `compile()`
验证且不写 `pyc`；119 个 dataset manifest 与 index 的 5,672,789 个文件、
3,587,926,590,672 bytes 聚合一致；11 个黄金来源 SHA256 全部一致。原有 polygon proxy
合成测试 5/5 通过；21 个 package source 文件通过 mypy；Git 禁用路径计数为 0，
`git diff --check` 和 `uv lock --check` 通过。

本次测试解释器是 `/mnt/d/anaconda3/envs/gendiff/bin/python`，Python 3.11.13、PyYAML
6.0.1、Shapely 2.1.2、GEOS 3.13.1、SciPy 1.13.0、Torch 2.4.0+cu121。它与历史训练
run 的关系仍为 `unknown`，不能称为正式 release environment。

## 阻塞项

1. 当前 candidate 尚未形成 clean commit 和正式 wheel SHA256；因此已授权 pilot 还不能启动。
2. GenDiff 尚未固定本包的 wheel/Git commit，现有 loader 也不会原生强制 canonical metadata；
   pilot validator 只能在调用 loader 前执行合同检查。
3. 正式 100-building pilot、forward contract、bounded overfit、训练和 bulk regeneration 均未运行。
4. 历史 packed release 的 producer commit/command、dirty diff hash 和训练 Python/lock 仍为
   `unknown`；已观测 train/val/test alias 问题也未被本轮改写。

## 精确下一门槛

先完成当前工作树的最终检查、显式暂存、提交和推送，并从该 clean commit 构建带 SHA256
的 wheel；随后执行已授权的 100-building versioned pilot，输出 determinism、round-trip、
capacity、collision、split 和 silent-drop 报告。Pilot FAIL 时停在失败审阅；PASS 后仍需另行
授权 bounded overfit。在 bounded overfit PASS 前不得进行全量生成或训练。

## 后续授权更新

用户于 2026-08-29 批准了建议的 normalization、building split、condition sampler 和
package pin，并授权黄金 hash 审阅、提交推送和最多 100-building pilot。四项合同及黄金
双路径审阅现记录在 `docs/CANONICALIZER_PILOT_CONTRACT.md`；本报告前文仍保留 2A 到 2E
完成时的审计轨迹；上方“阻塞项”和“精确下一门槛”已按当前授权更新，不把尚未执行的
正式 pilot 写成已通过。

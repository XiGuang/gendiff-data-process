# Canonicalizer Phase 2 实现决策

日期：2026-08-29

状态：Phase 2A 冻结。本文只记录实现边界；算法语义仍以
`docs/handoff/GenDiff_unified_canonicalizer_spec_v1.md` 为准。

## 代码归属

Unified Canonicalizer 的唯一实现源位于本仓库的
`gendiff_data_process/canonicalization/`。`building_process/` 下的历史 candidate
不再作为 canonical v1 的语义来源。GenDiff 只能通过固定版本和 commit 消费该包，
不得复制一份独立实现。

Core 只依赖 Python 数据结构与 Shapely，不依赖 Torch，不读取目录，也不写数据。
GenDiff checkout 在本阶段保持只读；area-v2 loader smoke 只在临时目录生成不超过三个
building 的小型 packed fixture。

## 已冻结合同

1. 坐标系是 world XZY，footprint 为 XZ，height 为 Y。
2. Core 在所有几何操作前按 1 mm 网格执行 half-away-from-zero 整数量化。
3. Stage 和 edit hash 只使用版本化配置、规范字段和整数坐标。
4. Raw proxy/point ID、路径、时间戳和输入排列只进入 provenance，不参与身份与 hash。
5. Ring 统一 CCW，并选择完整顶点循环序列的字典序最小 rotation。
6. Stage 由挤出体 union 的无交叠 slab decomposition 产生；MultiPolygon 全部保留，
   hole、自交、无效高度和实体丢失 hard fail。
7. Layer matching 使用整数化多目标全局 assignment，并显式进行字典序 tie-break。
   不依赖第三方 solver 在平分情况下的返回顺序。
8. Point matching 使用 cyclic、order-preserving DP。无可靠匹配时固定
   delete-then-insert，并产生 warning；禁止按数组下标猜 identity。
9. 归一化不进入 canonical core。Area-v2 adapter 必须接收显式、带 profile ID 的
   normalization stats；不能从 target 隐式计算。
10. Adapter 的 MOVE/INSERT value 是 normalized absolute target XZ，容量超限在
    pack/tensorization 前 hard fail，禁止 clamp、slice 或 drop。

## 本阶段范围

Phase 2A 到 2E 只实现并验证：配置/hash、合成和有界真实 fixture、几何 core、
lineage/edit/apply、area-v2 adapter 和只读 loader smoke。不会生成 pilot、扫描全量数据、
运行训练、修改 `/mnt/d/projects/GenDiff` 或提升任何 pipeline/dataset 为 `current`。

## Pilot 授权后的冻结结果

- 正式 normalization 使用 successful train buildings 的 integer bbox uniform profile。
- Building-level train/val/test 使用固定 seed 的 SHA256 threshold 80/10/10。
- Condition 使用 canonical addition solid、largest-remainder、Halton 候选和字典序起点 FPS。
- 本包从 clean commit 构建 wheel，并按 Git commit 与 wheel SHA256 双重固定。

精确算法、命令和失败门槛见 `docs/CANONICALIZER_PILOT_CONTRACT.md`。这些合同不改变
core geometry/stage hash；condition、normalization、split 和 package 各自使用独立 config
hash。正式 pilot 尚未在本文更新时执行，bounded overfit 仍被阻塞。

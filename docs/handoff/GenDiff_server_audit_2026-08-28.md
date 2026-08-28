# GenDiff 项目与数据审计（2026-08-28）

## 1. 结论先行

项目当前的主要瓶颈不是继续调网络，而是**训练数据尚未形成唯一、无泄漏、能代表真实任务的监督合同**。

目前实际生效的配置仍是 5 栋楼的小规模过拟合集：5 栋楼各 4 个阶段，笛卡尔积得到 1024 个区域状态，从 98,976 个合法前向状态对中抽取 20,000 对。训练、验证和测试使用的是同一批样本。模型已经能记住这批数据，但该结果不能衡量跨建筑泛化，也没有验证 1501 栋新数据上的能力。

用户担心的“拆分结果没有唯一性”确实存在，而且已经能在真实数据中观察到：原始阶段 YAML 没有稳定 `point_ids`，构造程序退化为按顶点下标生成 ID；同一轮廓只要改变起点或顺逆时针，物理几何不变，监督序列就会改变，甚至产生假的 MOVE/INSERT/DELETE 标签。

建议暂停扩大训练，先完成 P0 数据修复和独立划分；否则增加训练量只会放大相互矛盾的标签。

## 2. 项目当前目标

输入：任意大小的已有 proxy 建筑场景和描述变化的条件点云。

输出：更新后的 proxy 场景，以及从旧场景到新场景的可执行 edit 过程。

当前设计把大场景按条件点云簇和完整建筑划分为 tile；每个 tile 由网络预测层级动作、高度和逐层多边形点编辑，最后合并回完整场景。

## 3. 最新周报与服务器实际状态对照

最新周报：`D:\user\Dropbox\DiffGen汇报\2026.06.05周报.pdf`。

| 2026-06-05 周报内容 | 2026-08-28 服务器实际状态 |
|---|---|
| 修改程序化数据，使 proxy 层不再重叠 | 1501 栋新数据仍检测到 9,020 对有体积/平面交叠的层；“不重叠”没有成为全量数据的强校验条件 |
| 把大量小文件打包为 PT，降低 I/O 瓶颈 | 已实现并实际使用，属于已经完成且有效的工程优化 |
| 用 5 栋楼、每栋 4 阶段做区域过拟合 | 当前配置仍停留在这套数据；元数据实际是 1024 个组合，不是约 8000 个 |
| 新的 20k 数据继续过拟合 | 训练在周报后继续到 2026-06-11，已有 `epoch=170-step=122000.ckpt`；但验证/测试复用训练集，只证明记忆能力 |
| forward pipeline 已完成，待大场景测试 | 已生成一个 1501 栋楼的大场景 case，但未找到完整推理输出或 pipeline summary；该计划尚未闭环 |
| 下一步构造训练数据，先用 GT 点云训练 | 1501 栋新数据尚未接入当前训练配置；当前仍是 5 栋楼 tiny-overfit 数据 |

服务器项目最新文件活动停在 2026-06-11 左右，之后未发现新的训练或推理闭环产物。

## 4. 当前真实代码和数据链路

```text
CityEngine 分阶段 OBJ/YAML
  -> datasets/history_stages_origin（当前 5 栋训练来源）
  -> tools/generate_area_sequence_v2_dataset.py
       - 枚举区域状态和前向状态对
       - 构造 exterior delta 条件点云
       - 构造层动作、高度和逐点 AR 编辑序列
  -> tools/build_area_edit_v2_packed_dataset.py
  -> datasets/history_area_edit_v2_packed
  -> configs/packed_area_edit_v2_tiny_overfit.yaml
  -> BuildingChangeConditionV2 + BackboneBuildingLayerEditV2
  -> outputs/packed_area_edit_v2/tiny_overfit_20k_fixed_1
```

网络输入和输出概要：

- source：最多 64 层，每层最多 32 个二维顶点，每个 tile 最多 16 栋楼的 embedding。
- condition：数据模块先固定为 2048 个三维点，条件编码器再取 256 个 token。
- structure encoder：编码 source 的点、层、高度和 building id。
- fusion：融合 source 结构与条件点云。
- hierarchical decoder：预测层动作和目标高度。
- layer-wise AR decoder：逐层预测 KEEP/MOVE/DELETE/INSERT/EOS 和坐标值。
- forward pipeline：聚类条件点、关联建筑、生成 tile、推理后合并场景。

`/mnt/d/data/obj_to_language` 不是当前 CityEngine 阶段 YAML 的直接上游：它输出 `mesh_id + height + 3D footprint`，而训练阶段数据是 `proxy_id + min/max_height + 2D footprint`。当前 CityEngine YAML 的确切生成器在服务器上没有找到，可能仍位于 Windows/CityEngine 工作流中。因此数据来源目前不可完全复现。

## 5. 数据审计结果

### 5.1 当前真正用于训练的 5 栋数据

| 指标 | 结果 |
|---|---:|
| 建筑 / 阶段 / 层 / 顶点 | 5 / 20 / 42 / 205 |
| 区域状态 | 1,024 |
| 合法前向状态对 | 98,976 |
| 抽样训练对 | 20,000，状态对本身无重复 |
| train / val / test | 三者完全相同，val/test 是 train alias |
| 缺少显式 `point_ids` 的层 | 42 / 42 |
| 顺时针轮廓 | 30 / 42 |
| 非规范起点轮廓 | 16 / 42 |
| 数据最大容量 | 21 层、每层 7 点，远低于配置上限 |

20,000 个训练样本的动作计数：

| 动作 | 数量 |
|---|---:|
| KEEP_POINT | 688,281 |
| INSERT_POINT | 654,221 |
| DELETE_POINT | 20,030 |
| MOVE_POINT | **0** |
| DELETE_LAYER | **0** |

这套数据无法训练或评估点移动能力，也没有独立建筑上的泛化信号。

### 5.2 准备用于大场景的 1501 栋新数据

目录：`/mnt/d/projects/GenDiff/datasets/history_stages_all_new`。

| 指标 | 结果 |
|---|---:|
| 建筑 / 阶段 YAML | 1,501 / 6,003 |
| 缺失阶段 | `building_1501/stage_3` |
| 层 / 顶点 | 16,013 / 75,171 |
| 缺少显式 `point_ids` 的层 | 16,013 / 16,013 |
| 顺时针轮廓 | 8,852 |
| 非规范起点轮廓 | 9,061 |
| 超过每层 32 点上限的层 | 3（`building_0299` 的 stage 1/2/3，各 33 点） |
| 同阶段中发生交叠的层对 | 9,020 |
| 重复几何阶段 | 920 个阶段记录；全部 9,003 个前向阶段对中有 1,379 个纯 no-op 对 |
| 相邻阶段体积非单调 | 342 次 |
| 全部前向阶段对中包含体积删除 | 1,438 / 9,003 |
| 同时发生增加和删除 | 1,334 / 9,003 |
| 外部实体不变但 proxy 编辑非空 | 1 对 |

“阶段编号更大”目前并不等价于“只进行施工增加”。生成器按阶段下标把这些样本当作 forward，但 `include_demolition: false`，会形成语义冲突。

### 5.3 已实证的非唯一监督

在 1501 栋数据中，对同一 `proxy_id` 的轮廓变化进行检查：

- 48 个变化 transition 中，45 个仍共享精确几何顶点。
- 其中 26 个 transition 的共享顶点换了数组下标，共涉及 75 个移位顶点。
- 当前“`point_id = proxy_id * stride + point_index`”的策略会把这些情况解释成约 137 个表观移动动作。
- 典型原因是同一环改变了起始顶点或顺逆时针，例如 `building_0097`、`building_0099`、`building_0112`。

同一个多边形环在循环平移和反向后仍是同一几何对象，但当前 AR 标签依赖数组顺序，所以相同物理问题可以得到不同监督答案。这正是用户所怀疑的唯一性问题。

## 6. 根因定位

### 6.1 顶点身份由下标伪造

当前 CityEngine YAML 没有 `point_ids`。`load_area_stage_layers()` 退化为用 `building_id + proxy_id + point_index` 生成 ID，隐含假设“所有阶段的同一下标都是同一个物理顶点”。真实数据不满足这个假设。

### 6.2 规范化不完整

部分代码只保证 CCW，但没有把环旋转到唯一的起始顶点；`buffer(0)` 产生 MultiPolygon 时还会只保留最大块。`obj_to_yaml_new.py` 同样没有稳定的分量排序、规范起点、洞处理和稳定 ID，并会丢掉 MultiPolygon 的较小部分。

### 6.3 proxy 分解本身可能不唯一

相同实体可以由多组重叠或切分不同的 proxy 层表达。条件使用的是实体外表面差分，通常看不到内部 proxy 分解；若标签却要求复现某一种内部切分，就可能出现同一 source+condition 对应多个 edit 序列。

### 6.4 阶段语义没有被验证

生成器以 stage 下标决定 forward，没有先验证 `solid(stage_i) ⊆ solid(stage_j)`，导致 construction 数据混入拆除和重构。

### 6.5 数据划分发生直接泄漏

生成器把同一 `pair_records` 同时写入 train、val 和 test。当前接近 1.0 的指标不是 held-out 结果。

## 7. 建议的数据唯一性合同

在重新构造训练集前，先固定以下规则并写成自动校验：

1. **唯一轮廓表示**
   - 删除重复闭合点和连续重复点，统一坐标精度。
   - 修复/拒绝自交；MultiPolygon 必须显式拆层或拒绝，不能静默只保留最大块。
   - 统一为 CCW。
   - 在所有循环平移中选择字典序最小的完整顶点序列；有并列时比较后续完整序列。
   - 层按 building、min/max height、规范 footprint hash、稳定 lineage 排序。

2. **唯一跨阶段匹配**
   - 层匹配使用确定性的全局二分图匹配，而不是贪心匹配；代价至少包含 footprint IoU、高度重叠、中心距离和 lineage。
   - 点匹配同时枚举循环平移和反向，选择总代价最小的对应；确定性 tie-break 后再产生 KEEP/MOVE/DELETE/INSERT。
   - 无法可靠对应时，使用固定的 delete-then-insert 规则，不使用原数组下标冒充身份。

3. **唯一实体分解**
   - 明确 proxy 是互不相交的体积单元。
   - 先按所有高度事件切成 slab，再对每个 slab 做平面 arrangement/union，得到无交叠 cell；相邻且 footprint 相同的 slab 再确定性合并。
   - 若业务需要保留洞或多个分量，必须扩充 schema，不得静默丢失。

4. **阶段合法性**
   - construction-only 数据必须满足实体单调包含；否则剔除或明确标为 demolition。
   - 若未来支持 demolition，条件中需要增加增/删语义（例如符号通道或双分支条件），不能只给无符号 XYZ。

5. **可复现性**
   - 同一 OBJ/YAML 连续运行两次，规范 YAML 和 edit 标签的 SHA256 必须完全一致。
   - 对规范化后的 `source + condition` 做哈希；若同一输入哈希对应不同目标或不同规范标签，构造阶段直接报错。

## 8. 当前 forward pipeline 的具体不足

- building 数量检查实际上无效：embedding 映射先截断到 `max_buildings`，再检查映射长度，永远不会超限；多余建筑会落到 ID 0。
- 没有检查单层点数，`build_structure_tensors()` 会静默截断；新数据已有 3 层为 33 点，而模型上限是 32。
- 条件点过多时直接取数组前缀，不保证空间覆盖；应使用 FPS/分层均匀采样。
- 超过 layer 容量的 tile 只会跳过或报错，没有递归细分/拆 job 策略。
- 聚类的链式连通可能把远距离变化连成一个大簇，进一步触发容量问题。
- 没有发现针对 tile 分配、容量边界、合并去重和 round-trip 的自动化测试。
- 目前只有大场景 case 输入，没有找到跑完后的最终场景、summary 和对 GT 的几何指标。

## 9. 推荐下一步任务（按顺序）

### P0：先修数据合同，暂不扩大训练

1. 把上述唯一轮廓、层排序、层/点匹配实现为一个公共 canonicalizer，供 OBJ/YAML 转换、训练集生成和 forward 共用。
2. 增加 dataset validator，输出逐建筑错误清单，并把以下问题设为 hard fail：缺失阶段、无效多边形、层交叠、非单调 construction、重复阶段、容量超限、输入哈希冲突、重复运行不一致。
3. 查明并纳入版本控制的 CityEngine YAML 生成入口；记录原始 OBJ、生成参数、代码提交和随机种子。
4. 清洗 1501 栋数据。不要在原目录就地覆盖，生成带版本号的新数据目录和审计报告。
5. 在**建筑级别**先切 train/val/test，再在各自集合内构造状态对；禁止同一建筑跨 split。
6. 统计并平衡动作、层数、点数、建筑数和变化复杂度；若任务需要 MOVE/DELETE，训练集必须实际覆盖这些动作。

### P1：建立可信的小规模基线

7. 从清洗后的数据选约 50-100 栋构造 v3 小数据，先做 1-5 栋过拟合，要求 edit round-trip 精确重建目标。
8. 再做建筑级 held-out 训练，核心指标改为：目标重建 IoU/Chamfer、高度 MAE、有效多边形率、层交叠率、edit 执行成功率；token accuracy 只作为辅助指标。
9. 做必要消融：有/无 canonicalization、有/无 building embedding、不同条件采样、不同动作平衡。20k 与 80k 必须固定数据来源、训练 step、batch 和初始化后再比较。

### P1：修复并闭环 forward

10. 修正 building/point 容量检查，所有截断改成显式报错或可追踪的拆分。
11. 条件点改为空间覆盖采样；为超容量 tile 实现递归细分，同时保持整栋建筑和相关点簇不被拆散。
12. 先用 GT edit/target 测 tile 切分与 merge，再接模型；对现有 1501 栋 case 产出最终 YAML/OBJ、pipeline summary 和逐 tile 指标。
13. 增加自动化测试：边界建筑、跨 tile 点簇、新建筑插入、空条件、超 16 栋、超 64 层、33 点轮廓、重复 proxy id、被跳过 tile。

### P2：最后再扩展条件来源

14. 只有在 GT 点云的独立验证集上通过后，再接 image/VGGT 等点云来源；同时模拟缺失点、噪声和遮挡进行鲁棒性训练。

## 10. 建议的两周执行节奏

**第 1 周：数据修复。** 完成 canonicalizer、validator、建筑级 split；先清洗 100 栋并生成审计报告。验收条件是重复运行哈希一致、0 split 泄漏、0 非法重叠、0 非单调 construction、0 静默截断。

**第 2 周：基线与 forward。** 在 v3 小数据上完成 overfit + held-out；修复 forward 容量问题，跑通一个有 GT 的大场景 case。验收条件是 edit round-trip 成功、几何指标可复现、所有 skipped tile 有明确原因且无静默丢数据。

## 11. 审计范围与边界

- 服务器检查为只读，没有修改 `/mnt/d/projects/GenDiff` 或 `/mnt/d/data`。
- 两个服务器目录都有未提交/未跟踪文件；后续修改前应先保存当前工作区状态。
- 本次统计基于服务器现有 YAML、packed PT、配置、训练 CSV/ckpt 和 forward case。
- CityEngine 阶段 YAML 的精确生成脚本未在服务器定位到，是当前可复现链路中唯一明显缺口。

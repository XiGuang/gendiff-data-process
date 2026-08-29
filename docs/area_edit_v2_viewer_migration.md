# Area Edit V2 查看器迁移与兼容说明

## 当前状态与证据

原查看器来自只读审计的 GenDiff worktree：

- 路径：`/mnt/d/projects/GenDiff/construction_edit_animation_viewer/`；
- branch/HEAD：`structured_proxy` / `c6bcd8fda184dfa4042c8158a8fd8c797fb57fbc`；
- tracked viewer 文件数：43；迁移前这些文件相对 HEAD 无修改；
- 其中 `viewer/public/data/default.viewer.json` 是旧流程生成的展示样例，不是查看器
  实现，未纳入当前代码仓库。该文件在上述 commit 中为 1,363,890 bytes，SHA-256 为
  `d0f0ae2ef1157b914d9b4c3fdc42b9688fc3c19589db6bbe2486b462a00adbba`；当前查看器不
  内置或自动加载数据，必须由用户显式选择只读数据集路径；
- 原本文档迁移前 SHA-256：
  `66916e23260ee8f46cb11e86a26976c4c1e77ef8128ba68335465e6e98f7d87b`；
  其中关于 `area_delta_exterior` 的 5 行新增、2 行删除已保留在下文条件点云章节；
- 根级旧脚本 `/mnt/d/projects/GenDiff/tools/export_edit_animation_viewer_data.py`
  与完整查看器中的 exporter 不同，未作为当前 viewer 运行时迁移，也未删除。

当前归属：

| 责任 | 当前路径 | 状态 |
| --- | --- | --- |
| React/Three.js 前端 | `viewer/` | support |
| 数据摘要、分页和 condition API | `tools/dataset_browser_api.py` | support |
| pair 查看格式导出 | `tools/export_edit_animation_viewer_data.py` | support |
| packed schema/index/shard 读取 | `gendiff_data_process/viewer_packed.py` | support |
| Python 兼容测试 | `tests/viewer/` | experiment |

迁移后的查看器同时读取 legacy raw area-v2 和 `area_v2_packed_v1`。对当前双向
canonical candidate，它显示 `change_kind`、`pair_hash` 和精确 shard locator，并把 packed
`edit_object` 转为播放操作；`DELETE_LAYER`/`DELETE_POINT` 与反向
`INSERT_LAYER`/`INSERT_POINT` 均已通过测试和浏览器检查。

验证证据（2026-08-29）：

- `PYTHONPATH=. /mnt/d/anaconda3/envs/gendiff/bin/python -m unittest -v tests.viewer.test_packed_viewer`：
  3 个双向 packed 测试与 1 个 raw 回归测试；
- 显式 `GENDIFF_REPO=/mnt/d/projects/GenDiff` 的真实 loader smoke：2/2；
- `cd viewer && npm test -- --run`：24/24；
- `cd viewer && npm run build`：通过；仅有 bundle 大于 500 kB 的非阻塞提示；
- Playwright + 系统 Chrome：1440x1000 与 390x844 均完成 pair 加载；construction、
  demolition、3D、condition overlay 和 2D edit playback 无 console/page error；
- canvas 像素检查：desktop 1080x1000，RGB 标准差 16.997、6747 种颜色；mobile
  390x389，RGB 标准差 19.951、2546 种颜色，均非空白。

以上基础结果先来自三条 synthetic packed sample。随后对正式双向 100-building artifact
完成了只读 API 与浏览器抽检：

- 数据路径：
  `/mnt/d/artifacts/gendiff-data-process/runs/canonicalizer_pilot_bidirectional_v1_f0de8c4de1cf_b0001_b0100/outputs/canonicalizer_pilot_bidirectional_v1`；
- `PYTHONPATH=. /mnt/d/anaconda3/envs/gendiff/bin/python tools/dataset_browser_api.py summary
  --dataset-dir <上述路径>` 返回 packed area、382 pairs、296 states、split 278/66/38 和
  `hasConditions: true`；
- construction 抽样 `building_0002_stage_0_to_stage_1` 位于 `packed:train:0:0`，
  `pair_hash=e530c7249fa0317e42602d39bba673ca29724943b99aad9ac62c26df000e6a84`；
- demolition 抽样 `building_0002_stage_1_to_stage_0` 位于 `packed:train:0:1`，
  `pair_hash=545882e60aea5b134d248abd747591c95923c606b61a89b45b882ab374f965a6`；
- 两个方向都由 API 返回 `validationOk: true` 和 2048 个 condition 点；demolition condition
  的实际 shard locator 为 `shards/train/train_00000.pt#samples[1].condition`；
- Playwright 使用系统 Chrome 打开 `http://127.0.0.1:5174/`，分别检查 demolition 3D、
  demolition condition、demolition playback 和 construction playback，画面均非空白，
  console error 与 page error 均为 0；6 条 warning 仅为 `THREE.Clock` deprecation 和 headless
  WebGL `ReadPixels` 性能提示；
- 四张运行时截图 SHA-256 依次为
  `47b5c89c9e9bb818cab2a23b7d5916ea24bfec7c9d0290bb953179dcc4b6f987`、
  `5b83912ffb0d3dabbbb0ad3720d25745a9f85a0dd50aa09f730071a0980a4238`、
  `f71ebf76f0ee7302c1e6c145a24e39db54837dc655503376e889415edb9a7e0c`、
  `37f47722682d996743498cf96c1a35020d75638b84363a4265fc16ca579446da`；截图位于
  `/tmp/.playwright-cli/`，属于易失运行时证据，不替代 artifact 内的稳定 hash。

正式 artifact 的 viewer 兼容性因此为 PASS，但 generation 仍因 mixed transition 和 hole
标为 FAIL；视觉通过不得解除 release、bounded overfit 或训练 gate。

本文档面向原有 edit viewer 的改造。目标是让 viewer 可以正确读取当前改版后的区域级数据集，并可视化从 `t0` 到 `t1` 的编辑过程。

原文记录的 legacy area-v2 数据由 GenDiff 下列脚本生成；这些路径不是当前 canonical
producer，也未随查看器迁移：

- `tools/generate_area_sequence_v2_dataset.py`
- `tools/pack_area_edit_v2_dataset.py`
- `tools/build_area_edit_v2_packed_dataset.py`

当前 canonical packed candidate 由本仓库
`gendiff_data_process/canonicalization/pilot.py` 调用
`gendiff_data_process/canonicalization/adapters/area_v2.py` 写出。两条 producer lineage 不得
根据相同 schema 名合并。

核心变化是：数据不再是单栋建筑 edit sequence，而是一整片区域的 area-level edit sequence；坐标已经统一归一化；`edit_sequences_v2` 使用绝对目标坐标，不再使用 anchor 或相对坐标；输入 YAML 如果是局部 footprint，生成脚本会通过 OBJ 推断并转换到区域坐标后再归一化。

## 查看器需要先改的假设

原 viewer 如果按旧数据工作，通常会有这些隐含假设。新版数据下需要逐项修改。

| 旧假设 | 新事实 | 查看器修改 |
| --- | --- | --- |
| 一个样本只属于一栋建筑 | 一个样本是一片区域，包含多栋建筑 | UI 和渲染数据结构需要支持 `building_id` 分组 |
| `t1/t2` 是单栋建筑 stage | `t1/t2` 是区域状态目录 `area_state_xxxxxx` | state 读取逻辑改为读取 `states/<area_state>/bs_<area_state>_r0.yaml` |
| 坐标可能需要 viewer 再归一化 | 输出已经统一归一化 | 禁止 viewer 再按单样本或单建筑归一化 |
| edit sequence 是旧 `edit_sequences/` | 当前只写 `edit_sequences_v2/` | 改为读取 `v2_edit_sequence` |
| MOVE/INSERT 是相对坐标或依赖 anchor | MOVE/INSERT 是目标绝对坐标 | 删除 anchor 解析逻辑，直接使用 `target_coord` |
| 点 token 可以按 source 顺序播放 | v2 点 token 对目标 polygon 友好排序 | KEEP/MOVE/INSERT 按 `target_point_index` 放置，DELETE 单独处理 |
| layer id 在一个建筑内部唯一 | layer id 是区域级 id | 用 `proxy_id` 或 `building_id + building_layer_index` 区分层 |
| condition 点云可能是局部/未归一化 | condition 点云已按 area stats 归一化 | 直接按 `[x, y, z]` 渲染，不要再缩放 |

## 推荐读取入口

### Raw 数据集

raw 输出目录结构：

```text
<dataset_root>/
  dataset_meta.yaml
  train.yaml
  val.yaml
  test.yaml
  states/
  edit_objects/
  edit_sequences_v2/
  conditions/
  pair_meta/
  validation_reports/
  preview_objs/
```

viewer 最简单的入口是 split 文件：

```yaml
- t1: /abs/path/to/states/area_state_000067
  t2: /abs/path/to/states/area_state_000979
  condition: /abs/path/to/conditions/pair_..._r0.pt
  edit_object: /abs/path/to/edit_objects/pair_....yaml
  v2_edit_sequence: /abs/path/to/edit_sequences_v2/pair_....yaml
  edit_schema_version: area_v2_absolute_target_coord_no_anchor
  v2_preview_obj: /abs/path/to/preview_objs/pair_..._from_v2.obj
```

读取 state YAML：

```python
from pathlib import Path
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []

def resolve_state_yaml(state_dir):
    state_dir = Path(state_dir)
    return state_dir / f"bs_{state_dir.name}_r0.yaml"

combo = split_items[index]
history_layers = load_yaml(resolve_state_yaml(combo["t1"]))
target_layers = load_yaml(resolve_state_yaml(combo["t2"]))
edit_objects = load_yaml(combo["edit_object"])
v2_tokens = load_yaml(combo["v2_edit_sequence"])
```

### Packed 数据集

如果 viewer 需要快速浏览大量样本，建议支持 packed 输出：

```text
packed_dataset/
  dataset_meta.pt
  dataset_meta.yaml
  states.pt
  train_index.pt
  val_index.pt
  test_index.pt
  shards/
```

packed viewer 读取方式：

```python
import torch

meta = torch.load(root / "dataset_meta.pt", weights_only=False)
states = torch.load(meta["states_path"], weights_only=False)
index = torch.load(root / "train_index.pt", weights_only=False)
shard = torch.load(index["shards"][0]["path"], weights_only=False)

sample = shard["samples"][0]
history_state = states["states"][sample["source_state_index"]]
target_state = states["states"][sample["target_state_index"]]

history_layers = history_state["layers"]
target_layers = target_state["layers"]
condition = sample["condition"]
edit_objects = sample["edit_object"]
v2_tokens = sample.get("v2_edit_sequence")
```

当前 canonical adapter 的 packed sample 不写 `v2_edit_sequence`。查看器必须从
`edit_object[].action` 和 `point_edits[].action` 构造 layer/point 播放操作；只有历史
sample 确实携带 `v2_edit_sequence` 时才可直接读取该字段。

packed 的 `states.pt` 可能包含预计算 tensor：

```python
state["tensors"] = {
    "point_coords": Tensor[max_layers, max_points_per_layer, 2],
    "point_mask": Tensor[max_layers, max_points_per_layer],
    "height_values": Tensor[max_layers, 2],
    "layer_mask": Tensor[max_layers],
    "building_ids": Tensor[max_layers],
    "proxy_ids": Tensor[max_layers],
    "building_layer_indices": Tensor[max_layers],
}
```

这些 tensor 适合快速渲染 state；当前 canonical packed 的回放使用 list 形式的 `layers`
和 `edit_object`。

## Schema 判断

viewer 初始化时应检查：

```yaml
edit_schema_version: area_v2_absolute_target_coord_no_anchor
coordinate_normalized: true
normalization_scope: area
point_value_semantics: absolute_target_coord
anchor_supervision: false
legacy_edit_sequences_written: false
```

推荐校验逻辑：

```python
assert dataset_meta["edit_schema_version"] == "area_v2_absolute_target_coord_no_anchor"
assert dataset_meta["coordinate_normalized"] is True
assert dataset_meta["normalization_scope"] == "area"
assert dataset_meta["point_value_semantics"] == "absolute_target_coord"
assert dataset_meta["anchor_supervision"] is False
```

如果 viewer 同时支持旧版数据，建议按 `edit_schema_version` 分发到不同 parser：

```python
if schema == "area_v2_absolute_target_coord_no_anchor":
    parser = AreaEditV2Parser()
elif schema == "v2_absolute_target_coord_no_anchor":
    parser = SingleBuildingV2Parser()
else:
    parser = LegacyEditParser()
```

## 坐标和归一化

新版 raw/packed 输出里的几何都已经在同一个 area-level 归一化坐标系内：

- `states/*/bs_*_r0.yaml`
- `edit_objects/*.yaml`
- `edit_sequences_v2/*.yaml`
- `conditions/*.pt`
- `preview_objs/*.obj`
- packed `states.pt`
- packed shard `condition`

viewer 不应再按单栋建筑、单 state、单 pair 重新归一化。否则会再次拉伸，导致 preview 比例不对。

坐标含义：

- layer footprint: `[x, z]`
- layer height: `min_height`, `max_height`
- condition point cloud: `[x, y, z]`

反归一化仅用于显示世界坐标标尺或和原始 OBJ 对齐：

```python
stats = dataset_meta["normalization_stats"]

def denorm_point(x_norm, y_norm, z_norm):
    x = x_norm * stats["scale_xz"] + stats["center_x"]
    z = z_norm * stats["scale_xz"] + stats["center_z"]
    y = y_norm * stats["scale_y"] + stats["center_y"]
    return x, y, z
```

脚本当前使用等比归一化：

```text
scale_xz == scale_y == uniform_scale
```

但 viewer 不应硬编码这个假设，仍应使用 `scale_xz` 和 `scale_y` 两个字段。

## 输入 YAML 局部坐标转换

新版 generator 增加了：

```bash
--yaml-coordinate-mode auto
--yaml-coordinate-mode world
--yaml-coordinate-mode local-to-first-vertex
```

默认 `auto`。当输入 YAML 的 footprint 是 `docs/history_proxy_yaml_format.md` 中描述的 `local to first vertex` 格式时，脚本会从同 stage 目录下的 OBJ 推断世界平移：

```text
stage_x/stage_x.yaml  # 局部 footprint
stage_x/stage_x_0.obj # 世界坐标 OBJ
```

转换在生成 raw dataset 前完成。viewer 看到的 state、edit sequence、condition 都已经是转换后再归一化的坐标。

元数据会记录每个 stage 的转换：

```yaml
coordinate_transform:
  requested_mode: auto
  resolved_mode: local_to_first_vertex
  offset_x: -178.632
  offset_z: 25.494
  stage_obj: /path/to/stage_3_0.obj
  raw_match_count: 0
  shifted_match_count: 36
  footprint_point_count: 36
```

这些字段主要用于 viewer 的 debug panel，不应用于再次变换已输出的几何。

建议在 viewer 的 state info 面板显示：

- `yaml_coordinate_mode`
- `coordinate_transform.resolved_mode`
- `offset_x`, `offset_z`
- `raw_match_count`, `shifted_match_count`

这样可以快速发现输入是否仍是局部坐标，或者某个 stage 是否 fallback 到 world。

## State 层格式

区域 state 是 layer list：

```yaml
- proxy_id: 1000000003
  source_proxy_id: 1000000003
  local_proxy_id: 3
  building_id: 1
  building_name: building_0002
  building_stage_name: stage_3
  building_layer_index: 2
  level_index: 2
  min_height: -0.23
  max_height: 0.37
  footprint:
    - [-0.12, 0.08]
    - [-0.09, 0.10]
  point_ids:
    - 1000000300
    - 1000000301
  source_point_ids:
    - 1000000300
    - 1000000301
  point_roles:
    - original
    - original
```

viewer 中建议建立两个索引：

```python
layer_by_global_index = list(enumerate(layers))
layers_by_building = defaultdict(list)

for layer_index, layer in enumerate(layers):
    layers_by_building[layer["building_id"]].append((layer_index, layer))
```

常用显示字段：

- `building_id`: 区域内建筑编号
- `building_name`: 原始建筑文件夹名
- `building_stage_name`: 该 state 中此建筑的 stage
- `building_layer_index`: 当前区域 state 内，该建筑自己的 layer 序号
- `proxy_id`: 区域级 layer id，跨建筑不冲突
- `local_proxy_id`: 原 YAML 内的 proxy id

注意：`source_layer_index` 和 `target_layer_index` 是整个区域 layer list 的 index，不是单栋建筑内部 index。要显示“第几栋楼第几层”，应使用 `building_id + building_layer_index`。

## 区域 State 元数据

每个 state 目录包含：

```text
states/area_state_000067/
  area_state_meta.yaml
  bs_area_state_000067_r0.yaml
```

`area_state_meta.yaml` 示例：

```yaml
state_index: 67
state_name: area_state_000067
state_tuple: [0, 3, 1, 2, 0]
selections:
  - building_id: 0
    building_name: building_0001
    stage_position: 0
    stage_index: 0
    stage_name: stage_0
    stage_yaml: /path/to/building_0001/stage_0/stage_0.yaml
    coordinate_transform: ...
```

viewer 应显示 `source_state_tuple -> target_state_tuple`。这比只显示 layer edit 更容易理解“哪栋楼发生了变化”。

建议 UI：

```text
building_0001: stage_0 -> stage_1
building_0002: stage_3 -> stage_3
building_0003: stage_1 -> stage_2
```

如果某栋楼 `stage_position` 没变，但仍出现 edit，需要检查：

- 是否 source/target state 实际 layer 顺序或坐标不同
- `coordinate_transform` 是否对同一 stage 不一致
- viewer 是否把全局 layer index 误当成单栋 layer index
- pair 是否读取错了 t0/t1 文件

## Pair 元数据

`pair_meta/*.yaml` 是 viewer 的最佳 debug 入口：

```yaml
pair_name: pair_017840_area_state_000067_to_area_state_000979
source_state: area_state_000067
target_state: area_state_000979
source_state_index: 67
target_state_index: 979
source_state_tuple: [0, 3, 1, 2, 0]
target_state_tuple: [1, 3, 2, 3, 0]
include_demolition: false
is_demolition_pair: false
fallback_alignment_used: false
reconstructed_layer_count_match: true
reconstructed_point_count_match: true
max_coord_error: 0.0
max_height_error: 0.0
max_ar_tokens_required: 57
```

viewer 建议在样本列表中显示：

- `pair_name`
- `source_state -> target_state`
- `source_state_tuple -> target_state_tuple`
- `is_demolition_pair`
- `fallback_alignment_used`
- validation 四个字段

如果 validation 失败，viewer 应给出明显提示，而不是继续默认认为 v2 可重建 t1。

## 编辑对象格式

`edit_objects/*.yaml` 是完整调试格式，适合 viewer 的 tooltip、列表面板、筛选和点击定位。

Layer edit 示例：

```yaml
- action: MODIFY
  source_layer_index: 4
  target_layer_index: 5
  source_proxy_id: 1000000001
  target_proxy_id: 1000000001
  source_building_id: 1
  target_building_id: 1
  source_building_name: building_0002
  target_building_name: building_0002
  source_building_layer_index: 1
  target_building_layer_index: 2
  source_point_count: 4
  target_point_count: 5
  height_edit:
    source_min_height: -0.5
    source_max_height: -0.2
    target_min_height: -0.5
    target_max_height: 0.1
  point_edits:
    - action: MOVE
      source_point_id: 100000000001
      target_point_id: 100000000001
      source_point_index: 1
      target_point_index: 1
      source_coord: [-0.05, -0.2]
      target_coord: [-0.03, -0.18]
      source_point_role: original
      target_point_role: original
      value: [-0.03, -0.18]
```

Layer action：

- `KEEP`: 层不变
- `MODIFY`: 层存在于 t0/t1，但高度或 footprint 变化
- `INSERT`: t1 新增层
- `DELETE`: t0 删除层

Point action：

- `KEEP`: 点不变
- `MOVE`: 点移动，`target_coord` 是目标绝对坐标
- `INSERT`: 新增点，`target_coord` 是目标绝对坐标
- `DELETE`: 删除点

`edit_object` 比 v2 sequence 多出以下调试信息：

- `source_coord`
- `target_coord`
- `source_point_role`
- `target_point_role`
- source/target building name
- source/target point count

因此 viewer 做 hover 或逐点调试时应优先用 `edit_object`。v2 token 用于模拟模型输出或 token 回放。

## V2 编辑序列格式

`edit_sequences_v2/*.yaml` 是最小 token 流。

Layer token：

```yaml
- type: MODIFY_LAYER
  value:
    source_layer_index: 4
    target_layer_index: 5
    source_proxy_id: 1000000001
    target_proxy_id: 1000000001
    source_building_id: 1
    target_building_id: 1
    source_building_layer_index: 1
    target_building_layer_index: 2
```

Height token：

```yaml
- type: MODIFY_HEIGHT
  value:
    source_min_height: -0.5
    source_max_height: -0.2
    target_min_height: -0.5
    target_max_height: 0.1
```

Point token：

```yaml
- type: KEEP_POINT
  value:
    source_point_id: 100000000000
    source_point_index: 0
    target_point_id: 100000000000
    target_point_index: 0

- type: MOVE_POINT
  value:
    source_point_id: 100000000001
    source_point_index: 1
    target_point_id: 100000000001
    target_point_index: 1
    target_coord: [-0.03, -0.18]

- type: INSERT_POINT
  value:
    target_point_id: 100000000005
    target_point_index: 4
    target_coord: [-0.12, -0.19]

- type: DELETE_POINT
  value:
    source_point_id: 100000000004
    source_point_index: 3
```

Viewer parser 关键点：

- 不再读取 `anchor`。
- 不再把 `value` 当相对位移。
- `MOVE_POINT.target_coord` 是目标 `[x, z]`。
- `INSERT_POINT.target_coord` 是目标 `[x, z]`。
- `KEEP_POINT` 没有 `target_coord`，需要从 source layer 的 `source_point_index` 复制。
- `DELETE_POINT` 不进入目标 polygon，只用于动画或标记删除。
- layer token 和 height token 总是成组出现，然后跟 point token。

## V2 token 回放算法

推荐 viewer 用 v2 token 构造可播放的中间状态。

基本流程：

```python
from copy import deepcopy

def replay_v2(source_layers, tokens):
    output_layers = []
    current = None
    current_action = None
    current_source_index = None
    current_points = {}

    def finish():
        nonlocal current, current_action, current_points
        if current is None or current_action == "DELETE":
            current = None
            current_points = {}
            return
        current["footprint"] = [
            point for _, point in sorted(current_points.items(), key=lambda item: item[0])
        ]
        if len(current["footprint"]) >= 3:
            output_layers.append(current)
        current = None
        current_points = {}

    for token in tokens:
        token_type = token["type"]
        value = token.get("value") or {}

        if token_type.endswith("_LAYER"):
            finish()
            current_action = token_type[:-len("_LAYER")]
            if current_action == "DELETE":
                current = None
                current_source_index = value.get("source_layer_index")
                continue

            current_source_index = value.get("source_layer_index")
            source = source_layers[current_source_index] if current_source_index is not None else None
            current = deepcopy(source) if source is not None else {
                "footprint": [],
                "min_height": 0.0,
                "max_height": 0.0,
            }
            current["proxy_id"] = value.get("target_proxy_id", current.get("proxy_id"))
            current["building_id"] = value.get("target_building_id", current.get("building_id"))
            current_points = {}
            continue

        if current is None:
            continue

        if token_type in {"KEEP_HEIGHT", "MODIFY_HEIGHT", "ADD_HEIGHT"}:
            current["min_height"] = value.get("target_min_height", value.get("source_min_height", current["min_height"]))
            current["max_height"] = value.get("target_max_height", value.get("source_max_height", current["max_height"]))
            continue

        if token_type == "KEEP_POINT":
            source_index = value["source_point_index"]
            target_index = value["target_point_index"]
            current_points[target_index] = source_layers[current_source_index]["footprint"][source_index]
            continue

        if token_type in {"MOVE_POINT", "INSERT_POINT"}:
            target_index = value["target_point_index"]
            current_points[target_index] = value["target_coord"]
            continue

        if token_type == "DELETE_POINT":
            continue

    finish()
    return output_layers
```

实际实现中还应保留：

- token index
- layer edit index
- point edit index
- source/target ids
- animation phase
- validation error state

## 渲染几何

每个 layer 是一个垂直 prism：

- 底面高度 `min_height`
- 顶面高度 `max_height`
- 平面 polygon `footprint: [[x, z], ...]`

viewer 生成 mesh 时应使用：

```python
bottom = [(x, min_height, z) for x, z in footprint]
top = [(x, max_height, z) for x, z in footprint]
```

不要把 footprint 的第二维当作 y。它是 z。

建议渲染层级：

1. `history_layers` 灰色半透明
2. `target_layers` 淡色轮廓或可切换显示
3. 当前正在回放的 layer edit 高亮
4. condition point cloud 独立颜色
5. point edit handles/labels

## 条件点云

raw condition 是 `.pt`：

```python
points = torch.load(condition_path, weights_only=False).float()
# shape: [N, 3], order: x, y, z
```

packed condition 在 shard sample 内：

```python
points = sample["condition"]
```

condition 语义：

- 默认 `condition_surface_mode: area_delta_exterior`。
- 建楼或 stage 增加：采样 source/target 2.5D 几何差异中新增加体量的最终外部表面。高度增加时只包含新增高度段；层与层重叠或相邻产生的内部面不会被采样。
- 拆楼或 stage 减小：采样删除后暴露出的水平平面。
- 历史允许 mixed 的 area-v2 pair：两部分合并后裁剪/补齐到固定点数。当前
  `bidirectional_monotonic_v1` 对 mixed transition 失败关闭，不会静默生成该类 sample。

viewer 显示拆楼 pair 时，不要把 condition 误认为“被删除体量表面”。它表示拆除后多出来的平面条件。

旧数据或显式使用 `--condition-surface-mode changed_layer_prism` 生成的数据会采 changed target layer 的完整 prism 表面，可能包含内部重叠面。viewer 可以读取 `dataset_meta.condition_surface_mode` 或 `pair_meta.condition_surface_mode` 后在 debug 面板中提示。

## 拆除 pair

当生成使用 `--include-demolition` 时，pair 可能包含 stage 下降。

判断：

```python
is_demolition = pair_meta["is_demolition_pair"]
delta = [
    b - a
    for a, b in zip(pair_meta["source_state_tuple"], pair_meta["target_state_tuple"])
]
```

含义：

- `delta[i] > 0`: 第 i 栋楼建造进度增加
- `delta[i] == 0`: 第 i 栋楼 stage 不变
- `delta[i] < 0`: 第 i 栋楼拆除或回退

viewer 建议：

- `INSERT_LAYER`: 绿色
- `DELETE_LAYER`: 红色
- `MODIFY_LAYER`: 蓝色或黄色
- 拆楼 condition 平面：橙色点云
- 在 building list 中显示 stage delta

## 原查看器具体改造清单

### 1. 数据入口

需要支持两种入口：

- raw dataset root + split name
- packed dataset root + split name

raw 模式读取 `train.yaml/val.yaml/test.yaml`。packed 模式读取 `dataset_meta.pt`、`states.pt` 和 shard。

### 2. State 解析器

旧 resolver 可能直接找 `bs_*.yaml` 或单栋 stage 文件。新版应改为：

```python
state_name = basename(state_dir)
state_yaml = state_dir / f"bs_{state_name}_r0.yaml"
state_meta = state_dir / "area_state_meta.yaml"
```

### 3. 坐标处理

删除或关闭：

- per-building normalization
- per-pair normalization
- per-state normalization
- viewer 内部自适应拉伸到单位盒子的逻辑

保留：

- 相机 framing 可以根据当前 normalized bbox 自动设置
- 世界坐标显示可以通过 `normalization_stats` 反归一化

### 4. 层标识

旧 viewer 如果用 `proxy_id` 直接显示“第几层”，现在容易混淆。新版推荐：

```text
building_name / building_id
building_layer_index
proxy_id
local_proxy_id
```

UI label 示例：

```text
building_0005 L3 proxy=4000000006 local=6
```

### 5. 编辑序列解析器

必须新增 `area_v2_absolute_target_coord_no_anchor` parser。

删除旧逻辑：

- anchor lookup
- relative move vector accumulation
- `value + source_coord` 得到目标点
- 依赖旧 `edit_sequences/`

新增逻辑：

- 从 `combo["v2_edit_sequence"]` 读取 token
- `MOVE_POINT` 直接用 `target_coord`
- `INSERT_POINT` 直接用 `target_coord`
- `KEEP_POINT` 从 source footprint 复制
- `DELETE_POINT` 只用于删除动画，不写入目标 footprint

### 6. token 时间线

建议 timeline 分三层：

```text
Layer token:  KEEP_LAYER / MODIFY_LAYER / INSERT_LAYER / DELETE_LAYER
Height token: KEEP_HEIGHT / MODIFY_HEIGHT / ADD_HEIGHT
Point tokens: KEEP_POINT / MOVE_POINT / INSERT_POINT / DELETE_POINT
```

可视化时可以按 layer edit 折叠：

```text
[MODIFY_LAYER building_0002 L1]
  MODIFY_HEIGHT
  KEEP_POINT 0 -> 0
  MOVE_POINT 1 -> 1
  INSERT_POINT -> 4
```

### 7. 调试面板

建议新增这些字段：

- pair name
- schema version
- source/target state tuple
- per-building stage transition
- normalization stats
- yaml coordinate mode
- per-stage coordinate transform
- validation report
- fallback alignment flag
- max AR tokens required

### 8. 验证结果显示

viewer 应读取：

```yaml
validation_reports/<pair_name>.yaml
```

或 `pair_meta` 中的同名字段。

如果出现：

```python
not reconstructed_layer_count_match
not reconstructed_point_count_match
max_coord_error > 1e-5
max_height_error > 1e-5
```

UI 应提示“v2 sequence cannot exactly reconstruct target state”。

### 9. 预览 OBJ

`preview_objs/*_from_v2.obj` 是从 v2 sequence 重建出的 t1 preview。它已经归一化。

viewer 可以提供三种对比模式：

- source state
- target state
- reconstructed-from-v2 preview

如果 `v2_preview_obj` 为空，说明生成时用了 `--no-v2-preview-obj`。

### 10. 条件点云

原 viewer 如果只支持 `.ply`，需要支持 `.pt`：

```python
torch.load(condition_path, weights_only=False)
```

如果生成时加了 `--save-condition-ply`，才会有 `.ply`。

## 常见错误和定位

### 预览比例被拉伸

常见原因：

- viewer 又做了一次 normalization
- viewer 分别缩放 x/z/y 到不同范围
- 把 footprint `[x, z]` 误当作 `[x, y]`
- condition 点云和 layer 使用了不同坐标系

检查：

```yaml
dataset_meta.coordinate_normalized: true
dataset_meta.normalization_scope: area
dataset_meta.normalization_stats.uniform_scale
```

### 某栋楼 stage 没变但显示有 edit

先检查 `area_state_meta.yaml` 中对应 building 的 stage 是否真的相同。若相同，再检查：

- source/target layer 是否来自同一个 state 文件
- viewer 是否按全局 layer index 读取
- viewer 是否用 `building_layer_index` 错当全局 index
- 是否二次归一化导致坐标不同
- validation report 是否失败

### MOVE_POINT 很少

这通常是数据生成问题，不是 viewer 问题。原因可能是输入没有稳定 point id，或者 point 拓扑发生了重排。

viewer 可以显示：

- `fallback_alignment_used`
- point id
- point role
- source/target point index

用于辅助排查。

### condition 点云位置不对

检查：

- 是否直接加载 normalized `.pt`
- 是否把 `[x, y, z]` 当成 `[x, z, y]`
- 是否把 condition 反归一化但 layer 没有反归一化
- 是否只保存了 `.pt`，viewer 却加载了旧 `.ply`

## 最小查看器适配顺序

建议按这个顺序改，风险最低：

1. 支持读取 raw split item 的 `t1/t2/condition/edit_object/v2_edit_sequence`。
2. 支持 area state YAML：`bs_area_state_xxxxxx_r0.yaml`。
3. 禁用 viewer 内二次归一化。
4. 按 `building_id` 给 layer 分组和着色。
5. 新增 v2 parser：绝对 `target_coord`，无 anchor。
6. 支持 condition `.pt` 点云。
7. 增加 pair/state/debug 面板。
8. 支持 packed dataset 快速读取。
9. 增加 validation 和 reconstructed preview 对比。

完成前 6 步后，viewer 应该已经能正确显示新版数据的主要几何和编辑过程。后 3 步用于批量浏览和排查数据问题。

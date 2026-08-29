# 区域编辑数据查看器

本目录提供区域编辑数据的浏览器查看器。前端使用 React 和 Three.js，后端接口由
Vite 本地中间件调用当前仓库的 Python helper。查看器不修改数据集。
仓库不内置生成数据或默认展示样例；启动后需要显式选择只读数据集目录。

## 支持范围

- raw `area_v2`：读取 `states/`、`edit_sequences_v2/`、`edit_objects/`、
  `pair_meta/` 和 `conditions/`。
- packed `area_v2_packed_v1`：校验 `dataset_meta`，按 split index 定位 shard，
  从 `states.pt` 和 sample 的 `edit_object`、`condition` 生成查看数据。
- 双向 canonical pair：显示 `construction`、`demolition`、`pair_hash` 和 shard
  locator；支持 `INSERT_LAYER`、`DELETE_LAYER` 及对应点操作。
- 旧 packed sample：仍可读取不含 `change_kind`/`pair_hash` 的历史 metadata，方向显示为
  unknown，不根据名称猜测。

packed 模式依赖包含 PyTorch 和 PyYAML 的 Python 环境。前端依赖由
`package-lock.json` 冻结；正常开发环境可使用 `npm ci` 准备依赖，本次迁移验证复用了
服务器既有依赖缓存，没有执行安装。

## 启动

在仓库根目录运行：

```bash
cd viewer
PYTHON=/mnt/d/anaconda3/envs/gendiff/bin/python npm run dev
```

默认地址为 `http://127.0.0.1:5173`。远程服务器可通过 SSH 端口转发访问：

```bash
ssh -L 5173:localhost:5173 <server>
```

页面中选择数据集目录后，先加载摘要，再选择 split、pair 并加载。packed pair 列表返回
精确 shard locator，因此加载已选样本不需要重新遍历全部 shard。模糊查询最多检查
5,000 个 pair；达到上限时页面会明确提示缩小查询范围。

## 命令行导出

raw 与 packed 数据使用同一个入口：

```bash
PYTHON=/mnt/d/anaconda3/envs/gendiff/bin/python \
  python tools/export_edit_animation_viewer_data.py \
  --dataset-dir /path/to/dataset \
  --pair-id <pair_name> \
  --output /tmp/pair.viewer.json
```

packed 数据从 pair 列表 API 获得 locator 后，可附加：

```text
--pair-locator packed:<split>:<shard_index>:<sample_index>
```

## 验证

```bash
cd viewer
npm test -- --run
npm run build

cd ..
PYTHONPATH=. /mnt/d/anaconda3/envs/gendiff/bin/python \
  -m unittest -v tests.viewer.test_packed_viewer
```

当前迁移证据、格式差异和已知边界见
`docs/area_edit_v2_viewer_migration.md`。

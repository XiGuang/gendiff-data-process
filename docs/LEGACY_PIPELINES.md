# 历史处理链路

历史代码予以保留，因为它解释现有 3.59 TB 已观测数据树。它不与较新的
polygon/construction candidate 路径混合。

## 区块与条件数据家族

历史上的大致关系如下：

```text
data/origin + data/component
  -> cut/merge/process_change scripts
  -> data/block
  -> process_combination and rotation scripts
  -> data/condition
  -> external render/encoder steps
  -> data/images and data/latents
  -> gen_yaml scripts
  -> data/yaml
```

这是 low-confidence 的家族级关系。精确 command、config、commit 和输入版本均未保留，
因此单个 manifest 中的对应字段保持 `unknown`。

## 点云处理链路输出

`pipeline/select_t1_and_t2.py` 和
`pipeline/segment_and_normalize_point_cloud.py` 与
`pipeline/output/{blocks,global,metrics.json}` 相关。这些输出按表观文件大小约占
5.96 GB，并继续被 Git ignore。

## OBJ/YAML 语言实验

`obj_to_language/` 包含 component footprint schema 与当前 GenDiff CityEngine
stage schema 不同的转换器。它们仍可用于检查和 round-trip 实验，但并非当前
`history_stages_*` 或 `area` 训练数据的已确认 producer。

## 保留规则

`tmp`、`test`、`new`、`fixed`、`fast` 或 `exact` 等名称不能作为删除标准。
只有在 consumer 状态、内容 hash、保留需求和替代项均已记录后，才能归档 legacy 目录。

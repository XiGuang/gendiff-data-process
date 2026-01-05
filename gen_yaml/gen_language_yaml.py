import os
import yaml
import random

dataset_folder="/mnt/d/data/data/block/language/yingrenshi_langauge_rotate"
condition_folder="/mnt/d/data/data/condition/yingrenshi_vggt_rotate"
yaml_folder="/mnt/d/data/data/yaml/yingrenshi_building"

# 可通过环境变量控制验证集比例与随机种子
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.01"))  # 0~1 之间的小数，默认 10%
SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))

combinations_list = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder,f))]

# combinations_list: ["a_b_c", ...]
# 1) 按 (a,b) 分组，记录每组第三位 c 的最大值
prefix_to_max_third = {}
for entry in combinations_list:
    parts = entry.split("_")
    if len(parts) != 3:
        continue
    try:
        a, b, c = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        continue
    key = (a, b)
    current_max = prefix_to_max_third.get(key, -1)
    if c > current_max:
        prefix_to_max_third[key] = c

# 2) 以 (a,b) 为单位划分 train/val，确保同组不混合
group_keys = list(prefix_to_max_third.keys())
rng = random.Random(SPLIT_SEED)
rng.shuffle(group_keys)

# 计算验证集分组数量
val_group_count = int(len(group_keys) * VAL_RATIO)
if VAL_RATIO > 0 and val_group_count == 0 and len(group_keys) > 0:
    val_group_count = 1

val_keys = set(group_keys[:val_group_count])
train_keys = set(group_keys[val_group_count:])

def generate_pairs_for_group(a, b, max_c):
    pairs = []
    upper = max_c + 1
    for i in range(upper):
        for j in range(upper):
            if i == j:
                continue

            with open(f"{yaml_folder}/{a}_{b}_{i}/data.yaml",'r') as f:
                t1_data=yaml.load(f,Loader=yaml.FullLoader)
                t1_building=set(t1_data['building_indices'])
                t1_ground=set(t1_data['ground_indices'])
            with open(f"{yaml_folder}/{a}_{b}_{j}/data.yaml",'r') as f:
                t2_data=yaml.load(f,Loader=yaml.FullLoader)
                t2_building=set(t2_data['building_indices'])
                t2_ground=set(t2_data['ground_indices'])

            if (len(t2_building - t1_building) > 0 and len(t1_building - t2_building) == 0) or (len(t2_ground - t1_ground) > 0 and len(t1_ground - t2_ground) == 0):
                pairs.append({
                    "t1": f"{dataset_folder}/{a}_{b}_{i}",
                    "t2": f"{dataset_folder}/{a}_{b}_{j}",
                    "condition": f"{condition_folder}/{a}_{b}_{i}_{a}_{b}_{j}.pt"
                })
    return pairs

train_combinations = []
val_combinations = []

for (a, b), max_c in prefix_to_max_third.items():
    if (a, b) in val_keys:
        val_combinations.extend(generate_pairs_for_group(a, b, max_c))
    else:
        train_combinations.extend(generate_pairs_for_group(a, b, max_c))

with open("train.yaml", "w") as f:
    yaml.dump(train_combinations, f)

with open("val.yaml", "w") as f:
    yaml.dump(val_combinations, f)
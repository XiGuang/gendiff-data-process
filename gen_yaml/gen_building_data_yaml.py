import os
import yaml
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

dataset_folder="/mnt/d/data/data/block/lihu_building_fast"
condition_folder="/mnt/d/data/data/condition/lihu_fast_building_ground_vggt_rotate"
# 可通过环境变量控制验证集比例与随机种子
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.0"))  # 0~1 之间的小数，默认 10%
SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "60"))  # 进程数，可通过环境变量控制
# 直接从dataset_folder读取split文件
# split_file = os.path.join(dataset_folder, "test.lst")
# with open(split_file, "r") as f:
#     combinations_list = f.read().splitlines()

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

# 共享缓存（通过 multiprocessing.Manager 提供跨进程共享内存能力）
BUILDING_CACHE = None
CACHE_LOCK = None
LOCAL_CACHE = {}


def _init_worker(cache, lock):
    """初始化子进程的共享缓存引用。"""
    global BUILDING_CACHE, CACHE_LOCK
    BUILDING_CACHE = cache
    CACHE_LOCK = lock


def _load_building_indices(folder_path):
    """读取并缓存 building_indices，优先使用共享内存，减少重复 IO。"""
    global BUILDING_CACHE, CACHE_LOCK

    # 优先使用共享缓存
    if BUILDING_CACHE is not None:
        cached = BUILDING_CACHE.get(folder_path)
        if cached is not None:
            return cached
        with CACHE_LOCK:
            cached = BUILDING_CACHE.get(folder_path)
            if cached is not None:
                return cached
            data_path = os.path.join(folder_path, "data.yaml")
            with open(data_path, "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            building = frozenset(data["building_indices"])
            BUILDING_CACHE[folder_path] = building
            return building

    # 回退到本地缓存（单进程运行时减少重复 IO）
    cached = LOCAL_CACHE.get(folder_path)
    if cached is not None:
        return cached
    data_path = os.path.join(folder_path, "data.yaml")
    with open(data_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    building = frozenset(data["building_indices"])
    LOCAL_CACHE[folder_path] = building
    return building

def generate_pairs_for_group(a, b, max_c):
    pairs = []
    upper = max_c + 1
    for i in range(upper):
        for j in range(upper):
            if i == j:
                continue
            t1_folder = f"{dataset_folder}/{a}_{b}_{i}"
            t2_folder = f"{dataset_folder}/{a}_{b}_{j}"
            t1_building = _load_building_indices(t1_folder)
            t2_building = _load_building_indices(t2_folder)

            # if len(t1_building) > 0 and not os.path.exists(f"{dataset_folder}/{a}_{b}_{i}/kl_embed/{a}_{b}_{i}_r0.pt"): # building_normalized
            #     continue
            # if len(t2_building) > 0 and not os.path.exists(f"{dataset_folder}/{a}_{b}_{j}/kl_embed/{a}_{b}_{j}_r0.pt"):
            #     continue

            if len(t2_building - t1_building) > 0 and len(t1_building - t2_building) == 0:
                pairs.append({
                    "t1": f"{dataset_folder}/{a}_{b}_{i}",
                    "t2": f"{dataset_folder}/{a}_{b}_{j}",
                    "condition": f"{condition_folder}/{a}_{b}_{i}_{a}_{b}_{j}.pt"
                })
    return pairs

def _process_group(item):
    (a, b), max_c = item
    is_val = (a, b) in val_keys
    pairs = generate_pairs_for_group(a, b, max_c)
    return is_val, pairs


def main():
    train_combinations = []
    val_combinations = []

    items = list(prefix_to_max_third.items())
    workers = max(1, min(MAX_WORKERS, len(items)))

    # if not items:
    #     with open("longhua_building1.yaml", "w") as f:
    #         yaml.dump(train_combinations, f)
    #     with open("val.yaml", "w") as f:
    #         yaml.dump(val_combinations, f)
    #     return

    with Manager() as manager:
        cache = manager.dict()
        lock = manager.Lock()
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(cache, lock)
        ) as executor:
            future_to_item = {executor.submit(_process_group, item): item for item in items}
            for future in as_completed(future_to_item):
                is_val, pairs = future.result()
                if is_val:
                    val_combinations.extend(pairs)
                else:
                    train_combinations.extend(pairs)

    with open("lihu_building_train.yaml", "w") as f:
        yaml.dump(train_combinations, f)

    with open("lihu_building_val.yaml", "w") as f:
        yaml.dump(val_combinations, f)


if __name__ == "__main__":
    main()
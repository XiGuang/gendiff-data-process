import os

def count_unique_prefixes(root_dir: str) -> int:
    prefixes = set()

    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue

        parts = name.split("_")
        if len(parts) < 2:
            continue  # 不符合 a_b_c 格式，跳过

        prefix = f"{parts[0]}_{parts[1]}"
        prefixes.add(prefix)

    return len(prefixes), prefixes


if __name__ == "__main__":
    root_dir = "/mnt/d/data/data/block/yuehai_building_block_exact"  # 改成你的路径
    count, prefixes = count_unique_prefixes(root_dir)

    print(f"不同的前两个数字组合数量: {count}")
    # print("具体组合:")
    # for p in sorted(prefixes):
    #     print(p)

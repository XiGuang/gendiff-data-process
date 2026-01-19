# 根据指定的新建的building，选出对应的t1 t2对。
# 挑出所有楼的建好的作为t2数据
import os
import yaml

dataset_folder = "/mnt/d/data/data/block/yingrenshi_building_simple"
building_names=['building1','building5','building14','building20','building22']
building_indices = [0,4,13,19,21]
output_file = "pipeline/pairs_t1_t2.yaml"

def get_all_t2_names(dataset_folder):
    subfolders=os.listdir(dataset_folder)

    # 1) 按 (a,b) 分组，记录每组第三位 c 的最大值
    prefix_to_max_third = {}
    for sub in subfolders:
        if not os.path.isdir(os.path.join(dataset_folder,sub)):
            continue
        parts = sub.split("_")
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
    # 生成所有t2的路径
    t2_names = []
    for (a,b),c in prefix_to_max_third.items():
        folder_name = f"{a}_{b}_{c}"
        t2_names.append(folder_name)
    return t2_names

def is_exsiting_any_buildings(data,building_names=None,building_indices=None):
    existing_building_names = data.get("building_names",[])
    existing_building_indices = data.get("building_indices",[])
    if building_indices is not None:
        for idx in building_indices:
            if idx in existing_building_indices:
                return True
        return False
    if building_names is not None:
        for name in building_names:
            if name in existing_building_names:
                return True
        return False
    raise ValueError("Either building_names or building_indices must be provided.")

def is_exsiting_all_buildings(data,building_names=None,building_indices=None):
    existing_building_names = data.get("building_names",[])
    existing_building_indices = data.get("building_indices",[])
    if building_indices is not None:
        for idx in building_indices:
            if idx not in existing_building_indices:
                return False
        return True
    if building_names is not None:
        for name in building_names:
            if name not in existing_building_names:
                return False
        return True
    raise ValueError("Either building_names or building_indices must be provided.")

def is_exsiting_other_buildings(data,building_names=None,building_indices=None):
    existing_building_names = data.get("building_names",[])
    existing_building_indices = data.get("building_indices",[])
    if building_indices is not None:
        for idx in existing_building_indices:
            if int(idx) not in building_indices:
                return True
        return False
    if building_names is not None:
        for name in existing_building_names:
            if name not in building_names:
                return True
        return False
    raise ValueError("Either building_names or building_indices must be provided.")

def is_exsiting_any_grounds(data,ground_names=None,ground_indices=None):
    existing_ground_names = data.get("ground_names",[])
    existing_ground_indices = data.get("ground_indices",[])
    if ground_indices is not None:
        for idx in ground_indices:
            if idx in existing_ground_indices:
                return True
        return False
    if ground_names is not None:
        for name in ground_names:
            if name in existing_ground_names:
                return True
        return False
    raise ValueError("Either ground_names or ground_indices must be provided.")

def is_exsiting_other_grounds(data,ground_names=None,ground_indices=None):
    existing_ground_names = data.get("ground_names",[])
    existing_ground_indices = data.get("ground_indices",[])
    if ground_indices is not None:
        for idx in existing_ground_indices:
            if int(idx) not in ground_indices:
                return True
        return False
    if ground_names is not None:
        for name in existing_ground_names:
            if name not in ground_names:
                return True
        return False
    raise ValueError("Either ground_names or ground_indices must be provided.")

def get_t1_names_from_building_index(dataset_folder,t2_names,building_names=None,building_indices=None):
    t1_names=[]
    for t2 in t2_names:
        data_path = os.path.join(dataset_folder,t2,"data.yaml")
        with open(data_path, "r") as f:
            t2_data = yaml.load(f, Loader=yaml.FullLoader)
        if not is_exsiting_any_buildings(t2_data,building_names,building_indices):
            t1_names.append(t2)
        else:
            parts = t2.split("_")
            c=int(parts[2])
            for i in range(c):
                folder_name = f"{parts[0]}_{parts[1]}_{i}"
                data_path = os.path.join(dataset_folder,folder_name,"data.yaml")
                with open(data_path, "r") as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                # 拆除对应的楼，变成地面
                if is_exsiting_any_grounds(data,building_names,building_indices) and not is_exsiting_other_grounds(data,building_names,building_indices):
                    t1_names.append(folder_name)
                    print(f"Selected t1: {folder_name} for t2: {t2}")
                    break
    return t1_names

if __name__ == "__main__":
    t2_names = get_all_t2_names(dataset_folder)
    t1_names = get_t1_names_from_building_index(dataset_folder,t2_names,building_indices=building_indices)
    assert len(t1_names) == len(t2_names), f"t1_names and t2_names should have the same length, but got {len(t1_names)} and {len(t2_names)}"
    pairs = [{'t1': os.path.join(dataset_folder, t1), 't2': os.path.join(dataset_folder, t2), 'position': f"{t1.split('_')[0]}_{t1.split('_')[1]}" } for t1, t2 in zip(t1_names, t2_names)]
    
    # 保存到yaml文件
    with open(output_file, "w") as f:
        yaml.dump(pairs, f)
    print(f"Saved {len(pairs)} pairs to {output_file}")
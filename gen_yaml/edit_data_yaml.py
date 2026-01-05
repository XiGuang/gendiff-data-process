import os
import yaml
import re

path='/mnt/d/data/data/block/yuehai_building_block_exact'

ground_folder='/mnt/d/data/data/component/yuehai/ground_voxel'

ground_names=[f.split('.')[0] for f in os.listdir(ground_folder)]
ground_names.sort(key=lambda x: tuple(map(int, re.findall(r"\d+", x))))

folders=[os.path.join(path,f) for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

for folder in folders:
    file=os.path.join(folder,'data.yaml')
    if os.path.exists(file):
        with open(file,'r') as f:
            data=yaml.safe_load(f)
        data['ground_names'] = [ground_names[idx] for idx in data['ground_indices']]
        print(f"更新 {file} ，ground_names {data['ground_names']}")
        with open(file,'w') as f:
            yaml.dump(data,f)
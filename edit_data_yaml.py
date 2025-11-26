import os
import yaml

path='/mnt/d/data/data/block/yuehai_building_block_exact'

folders=[os.path.join(path,f) for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

for folder in folders:
    file=os.path.join(folder,'data.yaml')
    if os.path.exists(file):
        with open(file,'r') as f:
            data=yaml.safe_load(f)
        data['']
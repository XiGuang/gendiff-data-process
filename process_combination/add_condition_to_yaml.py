import yaml
import os

yaml_path='/mnt/d/projects/GenDiff/data_lists/yingrenshi_building_simple_noise_0.1/test.yaml'
condition_folder='/mnt/d/data/data/condition/yingrenshi_building_simple_combinations_rotate'

with open(yaml_path,'r') as f:
    data=yaml.safe_load(f)

for item in data:
    t1=item['t1'].split('/')[-1]
    t2=item['t2'].split('/')[-1]
    condition_path=os.path.join(condition_folder,f'{t1}_{t2}.pt')
    item['condition']=condition_path
with open(yaml_path,'w') as f:
    yaml.safe_dump(data,f)
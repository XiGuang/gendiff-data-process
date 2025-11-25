import yaml
import os
import numpy as np
import fpsample
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
import trimesh

def load_change_point_clouds(t1_dir, t2_dir, num_points=65536):
    with open(os.path.join(t1_dir, "data.yaml"), "r") as f:
        t1_data = yaml.safe_load(f)
    with open(os.path.join(t2_dir, "data.yaml"), "r") as f:
        t2_data = yaml.safe_load(f)
    new_building=set(t2_data['building_names']) - set(t1_data['building_names'])
    # new_ground=set(t2_data['ground_names']) - set(t1_data['ground_names'])

    mesh=None
    for name in new_building:
        mesh_path=os.path.join(t2_dir,'building',f'{name}.obj')
        if os.path.exists(mesh_path):
            # 使用内存映射以减少峰值内存
            data = trimesh.load(mesh_path, force='mesh')
            if mesh is None:
                mesh=data
            else:
                mesh=trimesh.util.concatenate([mesh,data])
    

    if mesh is None:
        return None
    
    point_clouds=mesh.sample(num_points)
    # idx = fpsample.bucket_fps_kdline_sampling(point_clouds, num_points, h=5)
    point_clouds = point_clouds[idx]
    return point_clouds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="YAML文件的路径")
    parser.add_argument("--sample_points_num", type=int, default=65536, help="每个模型采样的点数")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="并发线程数")
    parser.add_argument("--out_yaml", type=str, default="./data.yaml", help="保存已生成组合列表的YAML文件")

    args = parser.parse_args()
    with open(args.yaml, "r") as f:
        data = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    # worker: 在子线程内完成加载与保存，返回是否成功以及索引
    def _process_one(idx: int, d: dict) -> Tuple[int, bool, Optional[str]]:
        try:
            point_cloud = load_change_point_clouds(d['t1'], d['t2'], args.sample_points_num)
            save_path = os.path.join(args.save_dir, f"{os.path.basename(d['t1'])}_{os.path.basename(d['t2'])}.pt")
            if point_cloud is None:
                return idx, False, f"未发现新增模型，跳过保存: {save_path}"
            # 确保为float32并落盘
            tensor = torch.from_numpy(point_cloud.astype(np.float32, copy=False))
            torch.save(tensor, save_path)
            return idx, True, f"已保存: {save_path}"
        except Exception as e:
            return idx, False, f"处理失败(idx={idx}): {e}"

    results_flags = {}

    # 并发执行
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = {ex.submit(_process_one, i, d): i for i, d in enumerate(data)}
        done_cnt = 0
        total = len(futures)
        for fut in as_completed(futures):
            idx, ok, msg = fut.result()
            results_flags[idx] = ok
            done_cnt += 1
            if msg:
                print(f"[{done_cnt}/{total}] {msg}")

    # 按原顺序收集成功的条目
    pc_data = [d for i, d in enumerate(data) if results_flags.get(i, False)]

    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(pc_data, f)
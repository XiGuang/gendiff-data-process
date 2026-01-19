import os
import yaml
import trimesh
import numpy as np
import torch

def crop_and_normalize_pointcloud(points, cube_center, cube_size, bounds=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]):
    """
    切割出在cube中的点云并按cube进行归一化。
    
    参数:
        points (np.ndarray): 形状为 (N, 3) 的原始点云数据
        cube_center (list or np.ndarray): 形状为 (3,) 的中心点坐标 [x, y, z]
        cube_size (float): cube的边长
        
    返回:
        normalized_points (np.ndarray): 形状为 (M, 3) 的归一化后点云
                                        范围在 [-1, 1] 之间
    """
    # 确保输入是 numpy 数组
    points = np.asarray(points)
    cube_center = np.asarray(cube_center)
    
    normalized_points = (points - cube_center) / cube_size * 2

    min_bound = np.array(bounds[0:3], dtype=np.float32)
    max_bound = np.array(bounds[3:6], dtype=np.float32)
    
    if len(normalized_points) > 0:
        mask = np.all((normalized_points >= min_bound) & (normalized_points <= max_bound), axis=1)
        cropped_points = normalized_points[mask]
    else:
        # 如果没有点在cube内，返回空数组
        cropped_points = np.empty((0, 3))
        
    return cropped_points

if __name__ == "__main__":
    chunk_file='pipeline/pairs_t1_t2.yaml'
    point_cloud_file='pipeline/yingrenshi_0.5_changed.ply'
    output_dir='pipeline/normalized_pointclouds'
    os.makedirs(output_dir, exist_ok=True)

    with open(chunk_file, 'r') as f:
        chunks = yaml.safe_load(f)
    mesh = trimesh.load(point_cloud_file)
    points = np.array(mesh.vertices)

    for chunk in chunks:
        with open(os.path.join(chunk['t1'],'data.yaml'), 'r') as f:
            data = yaml.safe_load(f)
        position = data['position']
        size = data['size']
        normalized_points = crop_and_normalize_pointcloud(points, position, size, [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
        if normalized_points.shape[0] > 10:
            parts=chunk['t1'].split('/')[-1].split('_')
            number=f'{parts[0]}_{parts[1]}'
            torch.save(torch.from_numpy(normalized_points.astype(np.float32)),
                       os.path.join(output_dir, f"{number}.pt"))
            print(f"Saved normalized point cloud for chunk {number} with {normalized_points.shape[0]} points.")
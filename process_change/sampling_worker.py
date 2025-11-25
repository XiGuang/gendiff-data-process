import os
import numpy as np
import trimesh
import fpsample


def process_obj_to_points(task):
    """
    Process a single OBJ to sample points and save outputs.

    Parameters
    ----------
    task : tuple
        (obj_path, out_dir, name, num_points)

    Returns
    -------
    tuple
        (name, True/False, message) indicating success and optional info
    """
    obj_path, out_dir, name, num_points = task
    try:
        os.makedirs(out_dir, exist_ok=True)

        mesh = trimesh.load(obj_path, force='mesh')
        # Large initial sample to ensure good coverage. Keep consistent with caller expectations.
        raw_points = mesh.sample(200000)
        # Farthest point sampling with kdline heuristic
        idx = fpsample.bucket_fps_kdline_sampling(raw_points, num_points, h=5)
        points = raw_points[idx]

        # Save PLY
        ply_path = os.path.join(out_dir, f"{name}_points.ply")
        trimesh.points.PointCloud(points).export(ply_path)

        # Save NPZ
        npz_path = os.path.join(out_dir, f"{name}.npz")
        np.savez(npz_path, fps_points=points.astype(np.float32))

        return name, True, f"Saved: {ply_path}, {npz_path}"
    except Exception as e:
        return name, False, str(e)

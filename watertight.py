import trimesh

mesh = trimesh.load("/mnt/d/data/data/component/qingdao/building_simple/12_building_7.obj")

# 体素化
vox = mesh.voxelized(pitch=0.01)

# 重建水密网格
solid = vox.marching_cubes
if solid.is_watertight:
    print("网格是水密的")
else:
    print("网格不是水密的")
solid.export("watertight.obj")

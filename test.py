import torch

pc=torch.load("/mnt/d/projects/vggt/output/yingrenshi_fixed_condition/9_4_5_9_4_12.pt", map_location="cpu")

print(f"pc shape: {pc.shape}")
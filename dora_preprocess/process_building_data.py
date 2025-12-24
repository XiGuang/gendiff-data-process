import os
import shutil
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_folder", type=str, default="/mnt/d/projects/Dora/sharp_edge_sampling/yingrenshi_building/sample")
    parser.add_argument("--output_folder", type=str, default="/mnt/d/data/yingrenshi_building_simple")
    args = parser.parse_args()

    sample_folder = args.sample_folder
    output_folder = args.output_folder

    for folder in os.listdir(sample_folder):
        if os.path.isdir(os.path.join(sample_folder, folder)):
            for file in os.listdir(os.path.join(sample_folder, folder)):
                if file.endswith(".npz"):
                    # if os.path.exists(os.path.join(output_folder, folder,f"{folder}.npz")):
                    #     os.rename(os.path.join(output_folder, folder,f"{folder}.npz"), os.path.join(output_folder, folder,f"{folder}_origin.npz"))
                    shutil.copy(os.path.join(sample_folder, folder, file), os.path.join(output_folder, folder, os.path.basename(file)))
                    print(f"file {file} copied to {output_folder}/{folder}")



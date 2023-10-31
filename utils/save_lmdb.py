import os
import lmdb
import rasterio
import numpy as np
import pickle
from tqdm import tqdm
from rasterio.windows import Window
import glob

def extract_percentile_range(data, lo, hi):
    plo = np.percentile(data, lo, axis=(1, 2), keepdims=True).astype(np.float16)
    phi = np.percentile(data, hi, axis=(1, 2), keepdims=True).astype(np.float16)

    # data = np.clip(data, plo, phi).astype(np.float16)
    with np.errstate(divide='ignore', invalid='ignore'):
        data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
    return data

# Parameters
root_dir = "/vol/research/RobotFarming/Projects/hyperkon/database/train"
patch_size = 160
stride = 160
channels = 224
lmdb_path = os.path.join(root_dir, "train.lmdb")
map_size = 135_555_627_776  # Maximum storage size of the database

# Find L1C and L2A image files
l1c_folder = os.path.join(root_dir, "L1C")
l2a_folder = os.path.join(root_dir, "L2A")
l1c_files = sorted(glob.glob(f"{l1c_folder}/*.TIF"))
l2a_files = sorted(glob.glob(f"{l2a_folder}/*.TIF"))

assert len(l1c_files) == len(l2a_files), "Number of L1C and L2A images should be the same."

# Open the LMDB file
env = lmdb.open(lmdb_path, map_size=map_size)

# Read each image pair, extract patches
with env.begin(write=True) as txn:
    total_patches_saved = 0
    for l1c_file, l2a_file in tqdm(zip(l1c_files, l2a_files), total=len(l1c_files)):
        with rasterio.open(l1c_file) as l1c_src, rasterio.open(l2a_file) as l2a_src:
            assert l1c_src.meta['width'] == l2a_src.meta['width'] and l1c_src.meta['height'] == l2a_src.meta['height'], "Corresponding images should have the same dimensions."

            width, height = l1c_src.meta['width'], l1c_src.meta['height']

            # Slide a window over the image
            for top in range(0, height - patch_size + 1, stride):
                for left in range(0, width - patch_size + 1, stride):
                    l1c_patch = l1c_src.read(window=Window(left, top, patch_size, patch_size))
                    l2a_patch = l2a_src.read(window=Window(left, top, patch_size, patch_size))

                    # Check for non-zero pixels
                    bpc = np.count_nonzero(l1c_patch)
                    tpc = l1c_patch.size
                    bpr = bpc / tpc

                    if bpr > 0.99:
                        l1c_patch = extract_percentile_range(l1c_patch, 1, 99)
                        l2a_patch = extract_percentile_range(l2a_patch, 1, 99)

                        # Write the patches to the database
                        txn.put(f"anchor_{total_patches_saved}".encode(), pickle.dumps(l1c_patch[:channels]))
                        txn.put(f"positive_{total_patches_saved}".encode(), pickle.dumps(l2a_patch[:channels]))
                        total_patches_saved += 1

    # Write the 'num_samples' key to the database
    txn.put('num_samples'.encode(), str(total_patches_saved).encode())

print(f"Total patches saved: {total_patches_saved}")
print("Done.")

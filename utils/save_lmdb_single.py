import os
import lmdb
import rasterio
import numpy as np
import pickle
from tqdm import tqdm
from rasterio.windows import Window
import glob

def extract_percentile_range(data, lo, hi):
    plo = np.percentile(data, lo, axis=(1, 2), keepdims=True).astype(np.float32)
    phi = np.percentile(data, hi, axis=(1, 2), keepdims=True).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
    return data

def generate_patches(img_files, patch_size, stride, channels):
    for img_file in tqdm(img_files, total=len(img_files)):
        with rasterio.open(img_file) as img_src:
            width, height = img_src.meta['width'], img_src.meta['height']

            # Slide a window over the image
            for top in range(0, height - patch_size + 1, stride):
                for left in range(0, width - patch_size + 1, stride):
                    img_patch = img_src.read(window=Window(left, top, patch_size, patch_size))

                    # Check for non-zero pixels
                    bpc = np.count_nonzero(img_patch)
                    tpc = img_patch.size
                    bpr = bpc / tpc

                    if bpr > 0.99:
                        img_patch = extract_percentile_range(img_patch, 1, 99)
                        yield img_patch[:channels]

# Parameters
# root_dir = r"/vol/research/RobotFarming/Projects/hyperkon/database/quickset"
# root_dir = "/scratch/Projects/hyperkon/database/"
root_dir = "/vol/research/ucdatasets/enhyperset/hyperkon/database/full"
# root_dir = r"/mnt/fast/datasets/ucdatasets/enhyperset/hyperkon/database/full"
lmdb_dir = "/vol/research/RobotFarming/Projects/hyperkon/database/full"
folders = [
        # 'test', 
        'val', 
        'train'
    ]
patch_size = 32
stride = 28
channels = 224
batch_size = 100 #20 #500 # Adjust this as needed
map_size = 1_099_511_627_776
un = "_n" # un -> unnormalized or n -> normalized

for folder in folders:
    num_imgs = 2
    start = 550
    total_patches_saved = 0
    if folder == 'test' or folder == 'val':
        start = 100
        num_imgs = 1
    img_folder = os.path.join(root_dir, folder)
    img_files = sorted(glob.glob(f"{img_folder}/*.TIF"))[start:num_imgs+start]
    lmdb_path = os.path.join(lmdb_dir, folder, f"{folder}_{patch_size}_{stride}_{len(img_files)}{un}.lmdb")

    # Open the LMDB file
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    patches_batch = []
    patch_generator = generate_patches(img_files, patch_size, stride, channels)
    for img_patch in patch_generator:
        patches_batch.append(img_patch)

        # If the batch is full, write it and clear the list
        if len(patches_batch) == batch_size:
            with env.begin(write=True) as txn:
                for patch in patches_batch:
                    txn.put(f"anchor_{total_patches_saved}".encode(), pickle.dumps(patch))
                    total_patches_saved += 1
            patches_batch = []

    # Save any remaining patches
    if patches_batch:
        with env.begin(write=True) as txn:
            for patch in patches_batch:
                txn.put(f"anchor_{total_patches_saved}".encode(), pickle.dumps(patch))
                total_patches_saved += 1

    # Write the 'num_samples' key to the database
    with env.begin(write=True) as txn:
        txn.put('num_samples'.encode(), str(total_patches_saved).encode())

    print(f"Total patches saved in {folder}: {total_patches_saved}")

print("Done.")








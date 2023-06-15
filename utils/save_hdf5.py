import os
import h5py
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.windows import Window
import glob

def extract_percentile_range(data, lo, hi):
    plo = np.percentile(data, lo, axis=(1, 2), keepdims=True)
    phi = np.percentile(data, hi, axis=(1, 2), keepdims=True)
    data = np.clip(data, plo, phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
    return data

# Parameters
root_dir = "/vol/research/RobotFarming/Projects/hyperkon/database/test_val"
patch_size = 160
stride = 160
channels = 224
hdf5_filename = os.path.join(root_dir, "val.hdf5") 
batch_size = 100 # Define the batch size

# Find L1C and L2A image files
l1c_folder = os.path.join(root_dir, "L1C")
l2a_folder = os.path.join(root_dir, "L2A")
l1c_files = sorted(glob.glob(f"{l1c_folder}/*.TIF"))
l2a_files = sorted(glob.glob(f"{l2a_folder}/*.TIF"))

assert len(l1c_files) == len(l2a_files), "Number of L1C and L2A images should be the same."

# Open or create the HDF5 file
if os.path.isfile(hdf5_filename):
    print(f"Opening {hdf5_filename}... in update mode")
    file = h5py.File(hdf5_filename, "r+")
    dset_anchor = file["anchor_patches"]
    dset_positive = file["positive_patches"]
else:
    print(f"Creating {hdf5_filename}... in write mode")
    file = h5py.File(hdf5_filename, "w")
    dset_anchor = file.create_dataset("anchor_patches", (0, channels, patch_size, patch_size), maxshape=(None, channels, patch_size, patch_size), dtype='f')
    dset_positive = file.create_dataset("positive_patches", (0, channels, patch_size, patch_size), maxshape=(None, channels, patch_size, patch_size), dtype='f')


# Create an empty list for storing patches
anchor_patches = []
positive_patches = []
total_patches_saved = 0
# Read each image pair, extract patches
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
                    # Append the patch to the list
                    anchor_patches.append(l1c_patch[:channels])
                    positive_patches.append(l2a_patch[:channels])
                    
                    # If enough patches have been collected, write them to the HDF5 file
                    if len(anchor_patches) >= batch_size:
                        start_idx = len(dset_anchor)
                        dset_anchor.resize(start_idx + batch_size, axis=0)
                        dset_positive.resize(start_idx + batch_size, axis=0)
                        dset_anchor[start_idx:] = np.stack(anchor_patches)
                        dset_positive[start_idx:] = np.stack(positive_patches)
                        total_patches_saved += batch_size

                        # Empty the list for the next batch
                        anchor_patches.clear()
                        positive_patches.clear()
                # else:
                #     print(f"Non-zero pixel ratio for patch from {l1c_file} and {l2a_file} at position ({top}, {left}) is less than 0.99: {bpr}")
                    
# After all patches have been collected, write any remaining patches to the HDF5 file
if anchor_patches:
    start_idx = len(dset_anchor)
    batch_remainder = len(anchor_patches)
    dset_anchor.resize(start_idx + batch_remainder, axis=0)
    dset_positive.resize(start_idx + batch_remainder, axis=0)
    dset_anchor[start_idx:] = np.stack(anchor_patches)
    dset_positive[start_idx:] = np.stack(positive_patches)
    total_patches_saved += batch_remainder
    
file.close()
print(f"Total patches saved: {total_patches_saved}")
print("Done.")
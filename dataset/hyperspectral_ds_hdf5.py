import h5py
import torch
from torch.utils.data import Dataset
import rasterio
import os
import argparse
import time
import glob
import torch
import numpy as np
import pandas as pd
import random
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from kornia import augmentation as K
from rasterio.windows import Window

class HyperspectralPatchHDF5Dataset(Dataset):
    def __init__(self, hdf5_filename, transform=None):
        self.hdf5_filename = hdf5_filename
        self.file = h5py.File(hdf5_filename, "r")
        self.anchor_patches = self.file['anchor_patches']
        self.positive_patches = self.file['positive_patches']
        self.transform = transform

    def __len__(self):
        assert len(self.anchor_patches) == len(self.positive_patches), "Number of anchor and positive patches should be the same."
        return len(self.anchor_patches)

    def __getitem__(self, idx):
        anchor_patch = torch.from_numpy(self.anchor_patches[idx])
        positive_patch = torch.from_numpy(self.positive_patches[idx])

        if self.transform:
            positive_patch = self.transform(positive_patch)

        return anchor_patch, positive_patch

    def close(self):
        self.file.close()

if __name__ == '__main__':
    # Parse the arguments
    print("Parsing arguments...")
    if 1:
        config_path = r'/vol/research/RobotFarming/Projects/hyperkon/config/config_i_8.json'
    else:
        config_path = None
    parser = argparse.ArgumentParser(description='HyperKon Training')
    parser.add_argument('-c', '--config', default=config_path,type=str,
                            help='Path to the config file')
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    
    def extract_rgb(data, r_range:tuple, g_range:tuple, b_range:tuple) -> np.ndarray:
        r_mean = np.mean(data[r_range[0] : r_range[-1], :, :], axis=0)
        g_mean = np.mean(data[g_range[0] : g_range[-1], :, :], axis=0)
        b_mean = np.mean(data[b_range[0] : b_range[-1], :, :], axis=0)

        rgb_img = np.zeros((3, data.shape[1], data.shape[2]))

        rgb_img[0, :, :] = r_mean
        rgb_img[1, :, :] = g_mean
        rgb_img[2, :, :] = b_mean
        
        return rgb_img

    def display_image(image, image2=None, save_image=False, path=None, fname='rgb_color') -> None:
        fig, axes = plt.subplots(
                nrows=1, 
                ncols=2 if image2 is not None else 1, 
                #figsize=(15, 6)
            )
        
        if image2 is None:
            axes.imshow(image)
            axes.axis('off')
        else:
            axes[0].imshow(image)
            axes[0].axis('off')
            axes[0].set_title('L1C - Anchor Patch')

            axes[1].imshow(image2)
            axes[1].axis('off')
            axes[1].set_title('L2A - Positive Patch')

        plt.show()
        
        if save_image:
            if path:
                fname = os.path.join(path, fname)
            fig.savefig(f'{fname}.png')


    def move_axis(data, channel_last:bool=False):
            if channel_last:
                data = np.moveaxis(data, 0, -1)
            else:
                data = np.moveaxis(data, (1, 0), (2, 1))
            
            return data

    hdf5_file = config["hdf5_file"]

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    dataset = HyperspectralPatchHDF5Dataset(hdf5_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

    for i, (anchor, positive) in enumerate(dataloader):
        print(f'Batch {i+1} anchor shape: {anchor.shape}')
        print(f'Batch {i+1} positive shape: {positive.shape}')

        # a_img = extract_rgb(data=anchor[0].squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
        # p_img = extract_rgb(data=positive[0].squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
       
        # a_img = move_axis(a_img, True)
        # p_img = move_axis(p_img, True)
        # display_image(a_img, p_img)

        if i == 0:
            break
        # Deallocate GPU memory for the current batch
        del anchor
        del positive
        torch.cuda.empty_cache()
    end = time.time()
    dataset.close()
    print(f'Elapsed time: {end - start}')
    
    # Use this dataset class for HyperKon if not using LMDB... gets all possible patches per image

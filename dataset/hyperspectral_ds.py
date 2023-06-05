from osgeo import gdal
import os
import time
import glob
import torch
import numpy as np
import pandas as pd
import argparse
import random
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from kornia import augmentation as K


class HyperspectralPatchDataset(Dataset):
    def __init__(self, l1c_paths, l2a_paths, patch_size, 
                 channels=224, device=torch.device('cpu'),
                 normalize=False, transform=False):
        self.l1c_paths = l1c_paths
        self.l2a_paths = l2a_paths
        self.patch_size = patch_size
        self.channels = channels
        self.transform = self._kornia_augmentation() if transform else None
        self.device = device
        self.normalize = normalize
        
    def __len__(self):
        return min(len(self.l1c_paths), len(self.l2a_paths))

    def __getitem__(self, idx):
        l1c_path = self.l1c_paths[idx]
        l2a_path = self.l2a_paths[idx]

        l1c_src = gdal.Open(l1c_path)
        l2a_src = gdal.Open(l2a_path)

        l1c_image = l1c_src.ReadAsArray()
        l2a_image = l2a_src.ReadAsArray()

        _, height, width = l1c_image.shape
        
        # Randomly select the top-left corner for the patch
        while True:
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)

            # Extract the patches from Level 1C and Level 2A images
            anchor_patch = l1c_image[:, top:top + self.patch_size, left:left + self.patch_size]
            positive_patch = l2a_image[:, top:top + self.patch_size, left:left + self.patch_size]

            # If the majority of the pixels are non-zero, then we break the loop
            bpc = np.count_nonzero(anchor_patch)
            tpc = anchor_patch.size
            bpr = bpc / tpc

            if bpr > 0.99:
                break

        anchor_patch = self._extract_percentile_range(anchor_patch, 1, 99)
        positive_patch = self._extract_percentile_range(positive_patch, 1, 99)

        # Convert to torch tensors
        anchor_patch = torch.from_numpy(anchor_patch)
        positive_patch = torch.from_numpy(positive_patch)

        if self.normalize:
            normalize_transform = transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)
            anchor_patch = normalize_transform(anchor_patch)
            positive_patch = normalize_transform(positive_patch)

        if self.transform:
            anchor_patch = self.transform(anchor_patch).to(self.device)
            positive_patch = self.transform(positive_patch).to(self.device)
        
        return anchor_patch[:self.channels, :, :], positive_patch.squeeze()[:self.channels, :, :]
    
    def _kornia_augmentation(self):
        aug_list = K.AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.8),
            K.RandomGaussianBlur(kernel_size=(9,9), sigma=(0.1, 2.0), p=0.8),
            same_on_batch=True,
        )
        return aug_list

    def _extract_percentile_range(self, data, lo, hi):
        plo = np.percentile(data, lo, axis=(1, 2), keepdims=True)
        phi = np.percentile(data, hi, axis=(1, 2), keepdims=True)
        data = np.clip(data, plo, phi)
        # Handle potential divisions by zero or invalid operations
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
        
        return data


if __name__ == '__main__':
    # Parse the arguments
    if 1:
        config_path = r'config/config_i_1.json'
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

    log_dir = config["log_dir"]
    channels = config["in_channels"]
    normalize = config["normalize"]
    patch_size = config["patch_size"]
    l1c_folder = config["l1c_folder"]
    l2a_folder = config["l2a_folder"]
    normalize = config["normalize"]
    transform = config["transform"]

    l1c_paths = sorted(glob.glob(f"{l1c_folder}/*.TIF"))
    l2a_paths = sorted(glob.glob(f"{l2a_folder}/*.TIF"))

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dataset = HyperspectralPatchDataset(l1c_paths, l2a_paths, patch_size, 
                                            channels, device, normalize, transform)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    for i, (anchor, positive) in enumerate(dataloader):
        # print(f'Batch {i+1} anchor shape: {anchor.shape}')
        # print(f'Batch {i+1} positive shape: {positive.shape}')

        a_img = extract_rgb(data=anchor.squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
        p_img = extract_rgb(data=positive.squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
       
        a_img = move_axis(a_img, True)
        p_img = move_axis(p_img, True)
        display_image(a_img, p_img)

        if i == 1:
            break
        # Deallocate GPU memory for the current batch
        del anchor
        del positive
        torch.cuda.empty_cache()
    end = time.time()

    print(f'Elapsed time: {end - start}')
    
    # Use this dataset class for HyperKon if not using LMDB

# this has the ability to load only L1 images and then apply the transforms to use them as positive patches
import lmdb
import torch
from torch.utils.data import Dataset
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
import pickle

class HyperspectralPatchLMDBDataset(Dataset):
    def __init__(self, root_dir, is_train=False, channels=224, transform=None, 
                 normalize=False, load_only_L1=True, is_topk_test=False,
                 patch_size=32, stride=24, device=None, weka_mnt=None):
        self.root_dir = root_dir
        # self.file_folder = os.path.join(self.root_dir, 'train' if is_train else 'val')
        # self.is_topk_test = is_topk_test

        # if self.is_topk_test:
        #     self.file_folder = os.path.join(self.root_dir, 'test')

        # self.lmdb_path = os.path.join(self.file_folder, f'train_{patch_size}_{stride}.lmdb' if is_train else f'val_{patch_size}_{stride}.lmdb')

        # if self.is_topk_test:
        #     self.lmdb_path = os.path.join(self.file_folder, f'test_{patch_size}_{stride}.lmdb')

        self.env = lmdb.open(self.root_dir, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            num_samples_str = int((txn.get('num_samples'.encode())).decode())
            if num_samples_str is None:
                raise ValueError("Key 'num_samples' does not exist in the LMDB database.")
            self.length = int(num_samples_str)
        
        self.channels = channels
        self.transform = transform if transform else None
        self.normalize = normalize
        self.load_only_L1 = load_only_L1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            anchor_patch = pickle.loads(txn.get(f'anchor_{idx}'.encode())).astype(np.float32)
            if self.load_only_L1:
                positive_patch = anchor_patch.copy()
            else:
                positive_patch = pickle.loads(txn.get(f'positive_{idx}'.encode()))
                
        if self.normalize:
            anchor_patch = self._extract_percentile_range(anchor_patch, 1, 99)
            positive_patch = self._extract_percentile_range(positive_patch, 1, 99)

        anchor_patch = torch.from_numpy(anchor_patch)
        positive_patch = torch.from_numpy(positive_patch)

        if self.transform:
            positive_patch = self.transform(positive_patch)
            # convert to back to float16
            # positive_patch = positive_patch.squeeze()#.half()

        # if random.random() > 0.5:
        #     return positive_patch.squeeze()[:self.channels, :, :], anchor_patch[:self.channels, :, :]
            
        return anchor_patch[:self.channels, :, :], positive_patch.squeeze()[:self.channels, :, :]

    def _add_gaussian_noise(self, image, mean=0., std=0.05, p=0.5):
        if random.random() < p:
            noise = torch.randn_like(image) * std
            noise += mean
            return image + noise
        return image

    def _kornia_augmentation(self):
        # No longer need to define an augmentation pipeline since we're just using one function
        pass

    
    def _extract_percentile_range(self, data, lo, hi):
        plo = np.percentile(data, lo, axis=(1, 2), keepdims=True)#.astype(np.float32)
        phi = np.percentile(data, hi, axis=(1, 2), keepdims=True)#.astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
        return data

class HyperspectralPatchLMDBDataset2(Dataset):
    def __init__(self, root_dir, is_train=False, channels=224, transform=None, 
                 normalize=False):
        self.root_dir = root_dir
        self.file_folder = os.path.join(self.root_dir, 'test_train' if is_train else 'test_val')
        
        self.lmdb_path = os.path.join(self.file_folder, 'train.lmdb' if is_train else 'val.lmdb')
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        self.channels = channels
        self.length = self._get_length()
        self.transform = self._kornia_augmentation() if transform else None
        self.normalize = normalize
        self.cache = {}  # Cache for loaded patches

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx in self.cache:
            anchor_patch, positive_patch = self.cache[idx]
        else:
            with self.env.begin(write=False) as txn:
                # print(f'anchor_{idx}')
                anchor_patch = pickle.loads(txn.get(f'anchor_{54}'.encode()))
                positive_patch = pickle.loads(txn.get(f'positive_{idx}'.encode()))
                self.cache[idx] = (anchor_patch, positive_patch)

        anchor_patch = torch.from_numpy(anchor_patch)
        positive_patch = torch.from_numpy(positive_patch)

        if self.transform:
            positive_patch = self.transform(positive_patch)

        return anchor_patch, positive_patch

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            num_samples_str = pickle.loads(txn.get('num_samples'.encode()))
            if num_samples_str is None:
                raise ValueError("Key 'num_samples' does not exist in the LMDB database.")
            return int(num_samples_str)


    def _kornia_augmentation(self):
        aug_list = K.AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            K.RandomGaussianBlur(kernel_size=(9,9), sigma=(0.1, 2.0), p=0.5),
            same_on_batch=True,
        )
        return aug_list


if __name__ == '__main__':
    # Parse the arguments
    from torchvision import transforms
    root_dir = r'/vol/research/RobotFarming/Projects/hyperkon/database/full/train/train_160_160_2_n.lmdb'

    def extract_rgb(data, r_range:tuple, g_range:tuple, b_range:tuple) -> np.ndarray:
        r_mean = np.mean(data[r_range[0] : r_range[-1], :, :], axis=0)
        g_mean = np.mean(data[g_range[0] : g_range[-1], :, :], axis=0)
        b_mean = np.mean(data[b_range[0] : b_range[-1], :, :], axis=0)

        rgb_img = np.zeros((3, data.shape[1], data.shape[2]))

        rgb_img[0, :, :] = r_mean
        rgb_img[1, :, :] = g_mean
        rgb_img[2, :, :] = b_mean
        
        return rgb_img

    def display_image(image, image2=None, save_image=True, path=None, fname='rgb_color') -> None:
        print('preparing image for display...')
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


    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    transform = True
    

    if transform:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(160, 160)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        ])

    dataset = HyperspectralPatchLMDBDataset(root_dir, transform=transform)
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
    end = time.time() 
    print(f'Elapsed time: {end - start}')

    # Further code here...

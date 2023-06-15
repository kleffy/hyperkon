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

class HyperspectralPatchDataset(Dataset):
    def __init__(self, root_dir, is_train, patch_size, stride, channels=224, 
                device=torch.device('cpu'), normalize=False, transform=False, csv_file="patch_data.csv", weka_mnt=None):
        self.root_dir = root_dir
        self.weka_mnt = weka_mnt  
        self.file_folder = os.path.join(self.root_dir, 'test_train' if is_train else 'test_val')
        self.csv_file = os.path.join(self.file_folder, 'train.csv' if is_train else 'val.csv')

        self.l1c_paths = sorted(glob.glob(f"{self.file_folder}/L1C/*.TIF"))
        self.l2a_paths = sorted(glob.glob(f"{self.file_folder}/L2A/*.TIF"))

        self.patch_size = patch_size
        self.stride = stride
        self.channels = channels
        self.transform = self._kornia_augmentation() if transform else None
        self.device = device
        self.normalize = normalize
        # print(self.csv_file)
        if os.path.isfile(self.csv_file):
            self.patch_info = pd.read_csv(self.csv_file)
        else:
            self.patch_info = self._generate_patch_info()
            self.patch_info.to_csv(self.csv_file, index=False)

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        info = self.patch_info.iloc[idx]
        l1c_path = info['l1c_path']
        l2a_path = info['l2a_path']
        top, left = info['top'], info['left']
        
        if self.weka_mnt:
            l1c_path = os.path.join(self.weka_mnt, l1c_path[1:])
            l2a_path = os.path.join(self.weka_mnt, l2a_path[1:])

        with rasterio.open(l1c_path) as l1c_src:
            anchor_patch = l1c_src.read(window=Window(left, top, self.patch_size, self.patch_size))

        with rasterio.open(l2a_path) as l2a_src:
            positive_patch = l2a_src.read(window=Window(left, top, self.patch_size, self.patch_size))

        anchor_patch = self._extract_percentile_range(anchor_patch, 1, 99)
        positive_patch = self._extract_percentile_range(positive_patch, 1, 99)

        anchor_patch = torch.from_numpy(anchor_patch).float()#.to(self.device)
        positive_patch = torch.from_numpy(positive_patch).float()#.to(self.device)

        if self.normalize:
            normalize_transform = transforms.Normalize(mean=[0.5] * self.channels, std=[0.5] * self.channels)
            anchor_patch = normalize_transform(anchor_patch)
            positive_patch = normalize_transform(positive_patch)

        if self.transform:
            positive_patch = self.transform(positive_patch)

        if random.random() > 0.5:
            return positive_patch.squeeze()[:self.channels, :, :], anchor_patch[:self.channels, :, :]
        
        return anchor_patch[:self.channels, :, :], positive_patch.squeeze()[:self.channels, :, :]

    def _generate_patch_info(self):
        patch_info = []

        num_images = min(len(self.l1c_paths), len(self.l2a_paths))
        for idx in tqdm(range(num_images), desc='Generating patch info'):
            l1c_path = self.l1c_paths[idx]
            l2a_path = self.l2a_paths[idx]

            with rasterio.open(l1c_path) as l1c_src:
                l1c_image = l1c_src.read()
                l1c_meta = l1c_src.meta

            height, width = l1c_meta['height'], l1c_meta['width']

            for top in range(0, height - self.patch_size + 1, self.stride):
                for left in range(0, width - self.patch_size + 1, self.stride):

                    anchor_patch = l1c_image[:, top:top+self.patch_size, left:left+self.patch_size]
                    bpc = np.count_nonzero(anchor_patch)
                    tpc = anchor_patch.size
                    bpr = bpc / tpc

                    if bpr > 0.99:
                        patch_info.append({'l1c_path': l1c_path, 'l2a_path': l2a_path, 'top': top, 'left': left})

        return pd.DataFrame(patch_info)


    def _kornia_augmentation(self):
        aug_list = K.AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            K.RandomGaussianBlur(kernel_size=(9,9), sigma=(0.1, 2.0), p=0.5),
            same_on_batch=True,
        )
        return aug_list

    def _extract_percentile_range(self, data, lo, hi):
        plo = np.percentile(data, lo, axis=(1, 2), keepdims=True)
        phi = np.percentile(data, hi, axis=(1, 2), keepdims=True)
        data = np.clip(data, plo, phi)
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.where(phi - plo == 0, 0, (data - plo) / (phi - plo))
        return data



if __name__ == '__main__':
    # Parse the arguments
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

    log_dir = config["log_dir"]
    channels = config["in_channels"]
    normalize = config["normalize"]
    patch_size = config["patch_size"]
    root_dir = config["root_dir"]
    transform = config["transform"]
    stride = config["stride"]
    weka_mnt = config["weka_mnt"]
    
    if weka_mnt:
        log_dir = os.path.join(weka_mnt, log_dir[1:])
        root_dir = os.path.join(weka_mnt, root_dir[1:])

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dataset = HyperspectralPatchDataset(root_dir=root_dir, is_train=False, patch_size=patch_size, 
                                        stride=stride, channels=224, 
                                    device=torch.device('cpu'), normalize=False, 
                                    transform=False, weka_mnt=weka_mnt)

    
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

    print(f'Elapsed time: {end - start}')
    
    # Use this dataset class for HyperKon if not using LMDB... gets all possible patches per image

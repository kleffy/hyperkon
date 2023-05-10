import os
import time
import lmdb
import torch
import numpy as np
import pandas as pd
import argparse
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential

class HyperspectralPatchLmdbDataset(Dataset):
    def __init__(self, keys, lmdb_save_dir, lmdb_file_name, 
                 channels=224, device=torch.device('cpu'),
                 normalize=False, mean_file=None, std_file=None):
        self.keys = keys
        self.env = lmdb.open(os.path.join(lmdb_save_dir, lmdb_file_name), readonly=True, lock=False, max_readers=1024, map_size=int(1e12))
        self.channels = channels
        self.transform = self._kornia_augmentation()
        self.device = device
        self.lmdb_save_dir = lmdb_save_dir
        self.lmdb_file_name = lmdb_file_name
        self.normalize = normalize
        self.mean = mean_file
        self.std = std_file
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.env.begin() as txn:
            patch = txn.get(key.encode())
        
        key_split = key.split("_")
        patch_size = (int(key_split[-2]), int(key_split[-1]))
        channels = int(key_split[-4])

        # dtype = np.float16 if key_split[-9] == 'uint16' else np.float32

        patch = np.frombuffer(patch, dtype=np.float32).reshape((-1, patch_size[0], patch_size[1]))
        anchor = torch.from_numpy(np.copy(patch)).to(self.device)

        if self.normalize:
            normalize_transform = transforms.Normalize(mean=self.mean, std=self.std + 1e-8)
            anchor = normalize_transform(anchor)
        
        anchor = self._normalize_hyperspectral_image(anchor)
        positive = self.transform(anchor).to(self.device)
        
        return anchor[:self.channels, :, :], positive.squeeze()[:self.channels, :, :]
    
    def _kornia_augmentation(self):
        aug_list = K.AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.8),
            K.RandomGaussianBlur(kernel_size=(9,9), sigma=(0.1, 2.0), p=0.8),
            same_on_batch=True,
        )
        return aug_list

    def _normalize_hyperspectral_image(self, image):
        min_vals = image.view(image.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        max_vals = image.view(image.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        eps = 1e-8
        normalized_image = (image - min_vals) / (max_vals - min_vals + eps)
        return normalized_image


if __name__ == '__main__':
    # Parse the arguments
    if 1:
        config_path = r'config/config_1.json'
    else:
        config_path = None
    parser = argparse.ArgumentParser(description='HyperKon Training')
    parser.add_argument('-c', '--config', default=config_path,type=str,
                            help='Path to the config file')
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    

    def extract_rgb(data, r_range:tuple, g_range:tuple, b_range:tuple) -> np.ndarray:
        # print(f'extract_rgb - data shape:: {data.shape}')
        # data = data.cpu().numpy().squeeze()
        r_mean = np.mean(data[r_range[0] : r_range[-1], :, :], axis=0)
        g_mean = np.mean(data[g_range[0] : g_range[-1], :, :], axis=0)
        b_mean = np.mean(data[b_range[0] : b_range[-1], :, :], axis=0)

        rgb_img = np.zeros((3, data.shape[1], data.shape[2]))

        rgb_img[0, :, :] = r_mean
        rgb_img[1, :, :] = g_mean
        rgb_img[2, :, :] = b_mean
        
        # rgb_img = (rgb_img - np.min(rgb_img))/np.ptp(rgb_img)
        # print(f'After: {np.max(rgb_img)}')
        return rgb_img

    def extract_percentile_range(data, lo, hi):
        plo = np.percentile(data, lo)
        phi = np.percentile(data, hi)
        data[data[:,:,:] < plo] = plo
        data[data[:,:,:] >= phi] = phi
        data = (data - plo) / (phi - plo) #np.percentile(data, hi)
        return data

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
            axes[0].set_title('Anchor')

            axes[1].imshow(image2)
            axes[1].axis('off')
            axes[1].set_title('Positive')

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
    
    def read_csv_keys(csv_file, csv_file_col_name):
        df = pd.read_csv(csv_file)
        keys = df[csv_file_col_name].tolist()
        return keys
    
    lmdb_save_dir = config["lmdb_save_dir"]
    log_dir = config["log_dir"]
    lmdb_file_name = config["lmdb_file_name"] 
    columns = config["columns"]
    csv_file_name = config["csv_file_name"]
    channels = config["in_channels"]
    normalize = config["normalize"]
    mean_file = None
    std_file = None
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    keys = read_csv_keys(os.path.join(lmdb_save_dir, csv_file_name), columns[0])
    
    dataset = HyperspectralPatchLmdbDataset(keys, lmdb_save_dir, lmdb_file_name, 
                                            channels, device, normalize, mean_file=mean_file, std_file=std_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    for i, (anchor, positive) in enumerate(dataloader, 520):
        # print(f'Batch {i+1} anchor shape: {anchor.shape}')
        # print(f'Batch {i+1} positive shape: {positive.shape}')

        a_img = extract_rgb(data=anchor.squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
        p_img = extract_rgb(data=positive.squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
       
        a_img = move_axis(a_img, True)
        p_img = move_axis(p_img, True)
        display_image(a_img, p_img)

        if i == 530:
            break
        # Deallocate GPU memory for the current batch
        del anchor
        del positive
        torch.cuda.empty_cache()
    end = time.time()

    print(f'Elapsed time: {end - start}')
    
    # Use this dataset class for HyperKon

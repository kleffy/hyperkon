import os
import time
import lmdb
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
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
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.env.begin() as txn:
            patch = txn.get(key.encode())
        
        key_split = key.split("_")
        patch_size = (int(key_split[-2]), int(key_split[-1]))
        channels = int(key_split[-4])
        patch = np.frombuffer(patch, dtype=np.float32).reshape((channels, patch_size[0], patch_size[1]))
        
        anchor = torch.from_numpy(patch).to(self.device)

        if self.normalize:
            normalize_transform = transforms.Normalize(mean=self.mean, std=self.std + 1e-8)
            anchor = normalize_transform(anchor)
        
        anchor = self._normalize_hyperspectral_image(anchor)
        positive = self.transform(anchor).to(self.device)
        
        return anchor[:self.channels, :, :], positive.squeeze()[:self.channels, :, :]
    
    def _kornia_augmentation(self):
        aug_list = AugmentationSequential(
            K.RandomGaussianBlur(kernel_size=(9,9), sigma=(0.1, 2.0), p=0.5),
            K.RandomHorizontalFlip(),
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
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    import torchvision
    from datetime import datetime

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
        # test.append([plo,phi])
        # print(np.mean(np.array(test)))
        return data

    def display_image(image, save_image=False, path=None, fname='rgb_color') -> None:
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            plt.imshow(image)
            plt.show()
            if save_image:
                if path:
                    fname = os.path.join(path, fname)
                plt.savefig(f'{fname}.png')

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
    
    lmdb_save_dir = r'C:\Project\Surrey\Code\hyperkon\database'
    log_dir = r'C:\Project\Surrey\Code\hyperkon\logs'
    mean_file = None
    std_file = None
    
    lmdb_file_name = 'EP_PS224_S224_O00_N5_L1_CHW_V1.lmdb' 
    columns = ['enmap_patches_keys']
    csv_file_name = 'EP_PS224_S224_O00_N5_L1_CHW_V1.csv'
    channels = 224
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = False

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer_name_tag = f'test_tb_images_{timestamp}'
    writer = SummaryWriter(os.path.join(log_dir, writer_name_tag))
    
    keys = read_csv_keys(os.path.join(lmdb_save_dir, csv_file_name), columns[0])
    
    dataset = HyperspectralPatchLmdbDataset(keys, lmdb_save_dir, lmdb_file_name, 
                                            channels, device, normalize, mean_file=mean_file, std_file=std_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    for i, (anchor, positive) in enumerate(dataloader):
        print(f'Batch {i+1} anchor shape: {anchor.shape}')
        print(f'Batch {i+1} positive shape: {positive.shape}')

        print(anchor.shape)
        #anchor = move_axis(anchor.squeeze().cpu().numpy(), True)
        #print(anchor.shape)
        rgb_img = extract_rgb(data=anchor.squeeze().cpu().numpy(), r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
        # rgb_img = (rgb_img - np.min(rgb_img))/np.ptp(rgb_img)
        # rgb_img = extract_percentile_range(rgb_img, 2, 98)
        print(rgb_img.dtype)
        rgb_img = move_axis(rgb_img, True)
        display_image(rgb_img)

        if i == 10:
            break
        # Deallocate GPU memory for the current batch
        del anchor
        del positive
        torch.cuda.empty_cache()
    end = time.time()

    writer.close()
    print(f'Elapsed time: {end - start}')
    
    # Use this dataset class for HyperKon

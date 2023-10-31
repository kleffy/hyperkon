import torch
import json
import os
import numpy as np
import torchvision
# from torchvision.transforms import v2
from torchvision import transforms #as v2

from torch.utils.data import DataLoader

def extract_rgb(data, r_range: tuple, g_range: tuple, b_range: tuple) -> np.ndarray:
    # print(f'extract_rgb - data shape:: {data.shape}')
    r_mean = np.mean(data[r_range[0] : r_range[-1], :, :], axis=0)
    g_mean = np.mean(data[g_range[0] : g_range[-1], :, :], axis=0)
    b_mean = np.mean(data[b_range[0] : b_range[-1], :, :], axis=0)

    rgb_img = np.zeros((3, data.shape[1], data.shape[2]))

    rgb_img[0, :, :] = r_mean
    rgb_img[1, :, :] = g_mean
    rgb_img[2, :, :] = b_mean
    return rgb_img


# def extract_percentile_range(data, lo, hi):
#     plo = np.percentile(data, lo)
#     phi = np.percentile(data, hi)
#     data[data[:, :, :] < plo] = plo
#     data[data[:, :, :] >= phi] = phi
#     data = (data - plo) / (phi - plo)
#     return data


def log_tripplet_images(logger, anchor, positive, log_externally=0, step_num=0):
    anchor = anchor.cpu().detach().numpy()
    positive = positive.cpu().detach().numpy()

    a = extract_rgb(data=anchor[0], r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
    p = extract_rgb(
        data=positive[0], r_range=(46, 48), g_range=(23, 25), b_range=(8, 10)
    )
    n = extract_rgb(
        data=positive[2], r_range=(46, 48), g_range=(23, 25), b_range=(8, 10)
    )

    images = torch.stack(
        [
            torch.from_numpy(a),
            torch.from_numpy(p),
            torch.from_numpy(n),
        ]
    )

    grid = torchvision.utils.make_grid(images, nrow=3)

    if log_externally == 1: # log to wandb
        logger.log({"tripplet images": logger.Image(grid)})
        
    if log_externally == 2: # log to tensorboard
        logger.add_image("tripplet images", grid, step_num)

def save_best_model(epoch, model, train_loss, top_k_accuracy_train, val_loss, top_k_accuracy_val, model_path, model_dir):
    """
    Function to save the best model and metrics.
    """
    torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, model_path)

    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'top_k_accuracy_train': top_k_accuracy_train,
        'val_loss': val_loss,
        'top_k_accuracy_val': top_k_accuracy_val
    }

    with open(os.path.join(model_dir, "best_metrics.json"), "w+") as outfile:
        json.dump(metrics, outfile, indent=4)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y)


def evaluate_topk_performance(afeatures, pfeatures, query_index, k=5):
    query = pfeatures[query_index].unsqueeze(0)
    
    return torch.topk(cosine_similarity(query, afeatures), k, dim=-1).indices.squeeze()



def compute_top_k_accuracy(dataloader, fextractor, step_num, logger, topk_tag, 
                           k=1, log_externally=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    afeatures, pfeatures = [], []
    with torch.no_grad():
        for anchor, positive in dataloader:
            # anchor = anchor.half().to(device)
            # positive = positive.half().to(device)

            afeatures.append(fextractor(anchor.to(device, dtype=torch.float32)).detach().cpu())
            pfeatures.append(fextractor(positive.to(device, dtype=torch.float32)).detach().cpu())

            # afeatures.append(fextractor(anchor.to(device, dtype=torch.float32).unsqueeze(2)).detach().cpu())
            # pfeatures.append(fextractor(positive.to(device, dtype=torch.float32).unsqueeze(2)).detach().cpu())

    afeatures = torch.cat(afeatures, dim=0)
    pfeatures = torch.cat(pfeatures, dim=0)

    total_correct_equal = 0
    for i in range(len(pfeatures)):
        topk_indices = evaluate_topk_performance(
            afeatures=afeatures, pfeatures=pfeatures, query_index=i, k=k
        )
        total_correct_equal += torch.sum(topk_indices == i).item()

    top_k_accuracy_equal = total_correct_equal / len(afeatures)
    if log_externally == 1: # log to wandb
        logger.log({topk_tag: top_k_accuracy_equal, "step": step_num})
        
    if log_externally == 2: # log to tensorboard
        logger.add_scalar(topk_tag, top_k_accuracy_equal, step_num)

    return top_k_accuracy_equal


def init_gdal_datasets_and_dataloaders(dataset_obj, config, pin_memory=True):
    
    channels = config["in_channels"]
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    patch_size = config["patch_size"]
    root_dir = config["root_dir"]
    normalize = config["normalize"]
    transform = config["transform"]
    stride = config["stride"]
    weka_mnt = config["weka_mnt"]
    load_only_L1 = config.get("load_only_L1", False)
    is_topk_test = config.get("is_topk_test")
    test_batch_size = config.get("test_batch_size")
    val_stride = config.get("val_stride")
    num_workers = config.get("num_workers", 2)
    train_data_path = config.get("train_data_path", None)
    val_data_path = config.get("val_data_path", None)
    test_data_path = config.get("test_data_path", None)
    
    if transform:
        transform = transforms.Compose([
            # RandomBandShuffling(),
            # RandomBandDropping(drop_prob=0.1),
            # SpectralJittering(jitter_strength=0.05),
            # transforms.RandomResizedCrop(size=(patch_size, patch_size), antialias=True),
            # transforms.RandomHorizontalFlip(p=0.2),
            # transforms.RandomVerticalFlip(p=0.2),
            # transforms.RandomRotation(degrees=15),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        ])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = dataset_obj(
        root_dir=train_data_path,
        is_train=True,
        channels=channels,
        transform=transform,
        normalize=normalize,
        load_only_L1 = load_only_L1,
        is_topk_test = False,
        patch_size=patch_size,
        stride=stride,
        device=device,
        weka_mnt = weka_mnt,
    )

    val_dataset = dataset_obj(
        root_dir=val_data_path,
        is_train=False,
        channels=channels,
        transform=transform,
        normalize=normalize,
        load_only_L1 = load_only_L1,
        is_topk_test = False,
        patch_size=patch_size,
        stride=stride,
        device=device,
        weka_mnt = weka_mnt,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    if is_topk_test:
        test_dataset = dataset_obj(
            root_dir=test_data_path,
            is_train=False,
            channels=channels,
            transform=transform,
            normalize=normalize,
            load_only_L1 = load_only_L1,
            is_topk_test = is_topk_test,
            patch_size=patch_size,
            stride=stride,
            device=device,
            weka_mnt = weka_mnt,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            persistent_workers=True
        )
    else:
        test_dataloader = None

    return train_dataloader,val_dataloader,test_dataloader

class RandomBandShuffling:
    def __call__(self, img):
        # Assuming img is a PyTorch tensor with shape [bands, height, width]
        shuffled_indices = torch.randperm(img.shape[0])
        return img[shuffled_indices]

class RandomBandDropping:
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob
        
    def __call__(self, img):
        mask = torch.rand(img.shape[0]) > self.drop_prob
        return img * mask[:, None, None]

class SpectralJittering:
    def __init__(self, jitter_strength=0.05):
        self.jitter_strength = jitter_strength
        
    def __call__(self, img):
        jitter = (torch.rand_like(img) - 0.5) * 2 * self.jitter_strength
        return img + jitter

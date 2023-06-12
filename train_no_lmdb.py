
from dataset.hyperspectral_ds import HyperspectralPatchDataset
import os
import glob
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
from models.resnext_3D import resnext50, resnext101, resnext152
from models.hyperkon_2D_3D import HyperKon_2D_3D
from models.squeeze_excitation_v3 import SqueezeExcitation

from info_nce import InfoNCE
from loss_functions.kon_losses import NTXentLoss

def train(
    dataloader,
    model,
    criterion,
    optimizer,
    epoch,
    writer,
    k=1,
    add_tb_images=False,
    compute_top_k=True,
    dataset_obj=None,
):
    model.train()
    running_loss = 0.0
    top_k_accuracy_train = 0.0
    do_logging = True
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (anchor, positive) in enumerate(tepoch):
            tepoch.set_description(f"Training: Epoch {epoch + 1}")

            optimizer.zero_grad()

            a_output = model(anchor.unsqueeze(2))
            p_output = model(positive.unsqueeze(2))

            a_output = F.normalize(a_output)
            p_output = F.normalize(p_output)

            loss = criterion(a_output, p_output)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if ((epoch + 1) % config["val_frequency"] == 0) and do_logging:
                if add_tb_images:
                    tb_add_images(anchor, positive, epoch+1, writer, dataset_obj)

                if compute_top_k and ((epoch + 1) % config["val_frequency"] == 0):
                    top_k_accuracy_train = compute_top_k_accuracy(
                        dataloader=dataloader,
                        fextractor=model,
                        step_num=epoch+1,
                        writer=writer,
                        topk_tag="Top-k Accuracy/Train",
                        k=k
                    )

                do_logging = False

            tepoch.set_postfix(loss=loss.item())

    return running_loss / (i + 1), top_k_accuracy_train


def validate(dataloader, model, criterion, epoch, tbwriter, k=1):
    # model.eval()
    val_loss = 0.0
    top_k_accuracy_val = 0.0
    do_logging = True
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, (vanchor, vpositive) in enumerate(tepoch):
                tepoch.set_description(f"Validation: Epoch {epoch + 1}")

                va_output = model(vanchor.unsqueeze(2))
                vp_output = model(vpositive.unsqueeze(2))

                # normalize => l2 norm
                va_output = F.normalize(va_output)
                vp_output = F.normalize(vp_output)

                vloss = criterion(va_output, vp_output)

                # Print statistics
                val_loss += vloss.item()

                if ((epoch + 1) % config["val_frequency"] == 0) and do_logging:
                    top_k_accuracy_val = compute_top_k_accuracy(
                        dataloader=dataloader,
                        fextractor=model,
                        step_num=epoch+1,
                        writer=tbwriter,
                        topk_tag="Top-k Accuracy/Validation",
                        k=k
                    )

                    do_logging = False

                tepoch.set_postfix(loss=val_loss)

    return val_loss / (i + 1), top_k_accuracy_val


def move_axis(data, channel_last: bool = False):
    if channel_last:
        data = np.moveaxis(data, 0, -1)
    else:
        data = np.moveaxis(data, (1, 0), (2, 1))

    return data


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


def extract_percentile_range(data, lo, hi):
    plo = np.percentile(data, lo)
    phi = np.percentile(data, hi)
    data[data[:, :, :] < plo] = plo
    data[data[:, :, :] >= phi] = phi
    data = (data - plo) / (phi - plo)
    return data


def display_image(image, save_image=False, path=None, fname="rgb_color") -> None:
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(image)
    plt.show()
    if save_image:
        if path:
            fname = os.path.join(path, fname)
        plt.savefig(f"{fname}.png")


def tb_add_images(anchor, positives, step_num, writer, dataset_obj):
    anchor = anchor.cpu().numpy()
    positives = positives.cpu().numpy()

    a = extract_rgb(data=anchor[0], r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))
    p = extract_rgb(
        data=positives[0], r_range=(46, 48), g_range=(23, 25), b_range=(8, 10)
    )
    n = extract_rgb(
        data=positives[2], r_range=(46, 48), g_range=(23, 25), b_range=(8, 10)
    )

    images = torch.stack(
        [
            torch.from_numpy(a.astype(np.float32)),
            torch.from_numpy(p.astype(np.float32)),
            torch.from_numpy(n.astype(np.float32)),
        ]
    )

    grid = torchvision.utils.make_grid(images, nrow=3)
    # Add the grid of images to TensorBoard
    writer.add_image("tripplet images", grid, step_num)

    # for item in (("1. Anchor", a), ("2. Positive", p), ("3. Negative", n)):
    #     writer.add_histogram(f"{item[0]} Histogram", item[1], step_num)


def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y)


def evaluate(afeatures, pfeatures, query_index, k=5):
    query = pfeatures[query_index].unsqueeze(0)
    distances = cosine_similarity(query, afeatures)
    topk_indices = torch.topk(distances, k, dim=-1).indices.squeeze()
    return topk_indices


def compute_top_k_accuracy(dataloader, fextractor, step_num, writer, topk_tag, k=1):
    afeatures, pfeatures = [], []
    with torch.no_grad():
        for anchor, positive in dataloader:
            afeatures.append(fextractor(anchor.unsqueeze(2)).detach().cpu())
            pfeatures.append(fextractor(positive.unsqueeze(2)).detach().cpu())

    afeatures = torch.cat(afeatures, dim=0)
    pfeatures = torch.cat(pfeatures, dim=0)

    total_correct_equal = 0
    for i in range(len(pfeatures)):
        topk_indices = evaluate(
            afeatures=afeatures, pfeatures=pfeatures, query_index=i, k=k
        )
        total_correct_equal += torch.sum(topk_indices == i).item()

    top_k_accuracy_equal = total_correct_equal / len(afeatures)
    
    writer.add_scalar(topk_tag, top_k_accuracy_equal, step_num)

    return top_k_accuracy_equal


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    # Parse the arguments
    if 1:
        config_path = r'/vol/research/RobotFarming/Projects/hyperkon/config/config_i_3.json'
    else:
        config_path = None
    parser = argparse.ArgumentParser(description='HyperKon Training')
    parser.add_argument('-c', '--config', default=config_path,type=str,
                            help='Path to the config file')
    args = parser.parse_args()
    
    config = json.load(open(args.config))

    tag = config["tag"]
    log_dir = config["log_dir"]
    experiment_dir = config["experiment_dir"]
    experiment_name = config["experiment_name"]
    in_channels = config["in_channels"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    val_batch_size = config["val_batch_size"]
    out_features = config["out_features"]
    patch_size = config["patch_size"]
    l1c_folder = config["l1c_folder"]
    l1c_folder_val = config["l1c_folder_val"]
    l2a_folder = config["l2a_folder"]
    l2a_folder_val = config["l2a_folder_val"]
    normalize = config["normalize"]
    transform = config["transform"]
    k = config["k"]
    learning_rate = config["learning_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l1c_paths = sorted(glob.glob(f"{l1c_folder}/*.TIF"))
    l2a_paths = sorted(glob.glob(f"{l2a_folder}/*.TIF"))
    l1c_paths_val = sorted(glob.glob(f"{l1c_folder_val}/*.TIF"))
    l2a_paths_val = sorted(glob.glob(f"{l2a_folder_val}/*.TIF"))

    # Instantiate the dataset and the dataloaders
    train_dataset = HyperspectralPatchDataset(
        l1c_paths,
        l2a_paths,
        patch_size,
        in_channels,
        device,
        normalize=normalize,
        transform=transform,
    )

    val_dataset = HyperspectralPatchDataset(
        l1c_paths_val,
        l2a_paths_val,
        patch_size,
        in_channels,
        device,
        normalize=normalize,
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=0, 
        drop_last=True
    ) 
    
    
    torch.cuda.empty_cache()
    overall_vloss = 1_000_000.0
    msg =''
    
    writer_name_tag = f'{experiment_name}_C{in_channels}_b{batch_size}_e{num_epochs}_OF{out_features}_{tag}'
    writer = SummaryWriter(os.path.join(log_dir, writer_name_tag))
    
    if config.get("resnext") == 101:
        print("Initialised ResNext101!")
        model = resnext101(in_channels=in_channels, out_features=out_features).to(device)
    elif config.get("resnext") == 152:
        print("Initialised ResNext152!")
        model = resnext152(in_channels=in_channels, out_features=out_features).to(device)
    elif config.get("resnext") == 50:
        print("Initialised ResNext50!")
        model = resnext50(in_channels=in_channels, out_features=out_features).to(device)
    elif config.get("resnext") == 0:
        print("Initialised SqueezeExcitation!")
        embedding_dim = 512
        model = SqueezeExcitation(in_channels, embedding_dim, out_features).to(device)
    else:
        print("Initialised HyperKon_2D_3D!")
        #embedding_dim = 512
        model = HyperKon_2D_3D(in_channels, out_features).to(device)

    criterion = NTXentLoss()
    # criterion = InfoNCE()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print(f'{writer_name_tag}: Training started successfully...')
    best_vloss = 1_000_000.0
    val_loss = 1_000_001.0

    model_name = f'{experiment_name}_C{in_channels}_b{batch_size}_e{num_epochs}_OF{out_features}_{tag}'
    model_dir = os.path.join(experiment_dir, model_name)
    model_path = os.path.join(model_dir, "best_model.pth")
    ensure_dir(model_path)
    start_epoch = 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model checkpoint from {model_path}")
        print(f'starting from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):

        train_loss, top_k_accuracy_train = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            writer,
            k=k,
            add_tb_images=True,
            compute_top_k=True,
            dataset_obj=train_dataset,
        )
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % config["val_frequency"] == 0:
            val_loss, top_k_accuracy_val = validate(
                dataloader=val_dataloader, 
                model=model, 
                criterion=criterion, 
                epoch=epoch, 
                tbwriter=writer, 
                k=k
            )

            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # if (epoch + 1) % config["val_frequency"] == 0:
            print(f" Top-k Accuracy (Train): {top_k_accuracy_train:.4f}, Top-k Accuracy (Val): {top_k_accuracy_val:.4f}")
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)

            # Save the model
            if val_loss < best_vloss:
                best_vloss = val_loss
                
                torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, model_path)

                metrics = {
                            'epoch': epoch, 'train_loss': train_loss, 
                            'top_k_accuracy_train': top_k_accuracy_train, 
                            'val_loss': val_loss, 'top_k_accuracy_val': top_k_accuracy_val}
                
                with open(os.path.join(model_dir, "best_metrics.json"), "w+") as outfile: 
                    json.dump(metrics, outfile)

    # Close Tensorboard writer
    writer.close()

    print(
        f'{writer_name_tag}: Training completed successfully...'
    )
   
from dataset.hyperspectral_ds_lmdb3 import HyperspectralPatchLMDBDataset # uses lmdb
import os
import argparse
import json
from tqdm import tqdm
import torch
import wandb
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary

from models.resnext_3D import resnext101
from models.squeeze_excitation_v3_1 import SqueezeExcitation
from models.hyperkon_v3_1 import HyperKon_V3
from models.resnext_seb import resnext101_seb
from models.amsam2 import RefinedAMSAM
from models.amsam_seb import RefinedAMSAM_SEB
# from models.hyperkon_v3_2 import HyperKonSwish
from models.resnext_seb_small import resnext101_seb_small

from loss_functions.kon_losses import NTXentLoss
from utils.utility import (save_best_model,
                           ensure_dir,
                           compute_top_k_accuracy,
                           init_gdal_datasets_and_dataloaders,
                           log_tripplet_images,
                        )

# from dataset.hyperspectral_ds_v2_1 import HyperspectralPatchDataset



def train(dataloader, model, criterion, optimizer, epoch, logger, k=1, 
          compute_top_k=True, test_dataloader=None, accumulation_steps=4,log_images=False,
          log_externally=0):
    
    # model.half()  # Cast the entire model to float16
    model.train()
    running_loss = 0.0
    top_k_accuracy_train = 0.0
    do_logging = True
    optimizer.zero_grad()
    scaler = GradScaler()
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (anchor, positive) in enumerate(tepoch):
            tepoch.set_description(f"Training: Epoch {epoch+1}")
            # anchor = anchor.to(device)  # Cast anchor to float16
            # positive = positive.to(device)  # Cast positive to float16

            with autocast():
                a_output = model(anchor.to(device))
                p_output = model(positive.to(device))
                # a_output = model(anchor.to(device, dtype=torch.float32).unsqueeze(2))
                # p_output = model(positive.to(device, dtype=torch.float32).unsqueeze(2))

                a_output = F.normalize(a_output)
                p_output = F.normalize(p_output)

                loss = criterion(a_output, p_output)

            scaler.scale(loss).backward()
            
            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() / accumulation_steps
            
        if ((epoch + 1) % 5 == 0) and compute_top_k:
            top_k_accuracy_train = compute_top_k_accuracy(
                dataloader=dataloader if test_dataloader is None else test_dataloader,
                fextractor=model, step_num=epoch+1, logger=logger,
                topk_tag="Top-k Accuracy/Train", k=k,
                log_externally=log_externally
            )
            
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / (i + 1):.4f}")

            # log anchor and positive images to wandb
            if log_images and logger is not None:
                print("Logging triplet images...")
                log_tripplet_images(logger, anchor, positive, 
                                    log_externally=log_externally,
                                    step_num=epoch+1)
                log_images = False

        tepoch.set_postfix(loss=loss.item())

    return running_loss / (i + 1), top_k_accuracy_train


def validate(dataloader, model, criterion, epoch, logger, k=1, 
             test_dataloader=None, log_externally=0):
    model.eval()
    val_loss = 0.0
    top_k_accuracy_val = 0.0
    
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, (vanchor, vpositive) in enumerate(tepoch):
                tepoch.set_description(f"Validation: Epoch {epoch}")
                # vanchor = vanchor.to(device)
                # vpositive = vpositive.to(device)

                va_output = model(vanchor.to(device))
                vp_output = model(vpositive.to(device))
    
                # va_output = model(vanchor.to(device, dtype=torch.float32).unsqueeze(2))
                # vp_output = model(vpositive.to(device, dtype=torch.float32).unsqueeze(2))

                va_output = F.normalize(va_output)
                vp_output = F.normalize(vp_output)

                vloss = criterion(va_output, vp_output)

                val_loss += vloss.item()

            top_k_accuracy_val = compute_top_k_accuracy(
                dataloader=dataloader if test_dataloader is None else test_dataloader,
                fextractor=model, step_num=epoch, logger=logger,
                topk_tag="Top-k Accuracy/Validation", k=k,
                log_externally=log_externally
            )
            
            tepoch.set_postfix(loss=val_loss / (i + 1))

    return val_loss / (i + 1), top_k_accuracy_val



if __name__ == "__main__":
    # Parse the arguments
    
    config_path = r'/vol/research/RobotFarming/Projects/hyperkon/config/lmdb/config_test.json'
    
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
    root_dir = config["root_dir"]
    normalize = config["normalize"]
    transform = config["transform"]
    k = config["k"]
    learning_rate = config["learning_rate"]
    stride = config["stride"]
    weka_mnt = config["weka_mnt"]
    load_only_L1 = config.get("load_only_L1", False)
    is_topk_test = config.get("is_topk_test")
    test_batch_size = config.get("test_batch_size")
    val_stride = config.get("val_stride")
    accumulation_steps = config.get("accumulation_steps", 1)
    num_workers = config.get("num_workers", 2)
    model_type = config.get("model_type", 0)
    log_externally = config.get("log_externally", 0) # 0: don't log, 1: log to wandb, 2: log to tensorboard
    log_images = True
    
    if log_externally == 1: # log to wandb
        print("Logging to wandb...")
        wandb.init(project="HyperKon", 
                   config=config, 
                   entity="kleffy", 
                   name=experiment_name,
                )
        logger = wandb
    
    elif log_externally == 2: # log to tensorboard
        print("Logging to tensorboard...")
        writer_name_tag = f'{experiment_name}_C{in_channels}_OF{out_features}_PS{patch_size}_S{stride}_{tag}'
        logger = SummaryWriter(os.path.join(log_dir, writer_name_tag))
    else:
        logger = None
    

    if weka_mnt:
        log_dir = os.path.join(weka_mnt, log_dir[1:])
        experiment_dir = os.path.join(weka_mnt, experiment_dir[1:])
        root_dir = os.path.join(weka_mnt, root_dir[1:])
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the dataset and the dataloaders
    train_dataloader, val_dataloader, test_dataloader = init_gdal_datasets_and_dataloaders(
        HyperspectralPatchLMDBDataset, config)
    
    
    torch.cuda.empty_cache()
    overall_vloss = 1_000_000.0
    msg =''

    model_map = {
        0: SqueezeExcitation,
        1: resnext101_seb,
        2: resnext101,
        3: HyperKon_V3,
        4: RefinedAMSAM,
        5: RefinedAMSAM_SEB
    }

    model_class = model_map.get(model_type, 0)

    model = model_class(in_channels=in_channels, out_features=out_features).to(device)
    
    # summary(model, input_size=(1,in_channels, 32, 32),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=4)
    
    criterion = NTXentLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_vloss = 1_000_000.0
    val_loss = 1_000_001.0

    model_name = f'{experiment_name}_C{in_channels}_OF{out_features}_PS{patch_size}_S{stride}_{tag}'
    model_dir = os.path.join(experiment_dir, model_name)
    model_path = os.path.join(model_dir, "best_model.pth")

    ensure_dir(model_path)

    with open(os.path.join(model_dir, args.config.split('/')[-1]), "w+") as outfile: 
        json.dump(config, outfile, indent=4)

    start_epoch = 0

    print(f'{model_name}: Training started successfully...')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model checkpoint from {model_path}")
        print(f'starting from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):

        train_loss, top_k_accuracy_train = train(
            train_dataloader, model, criterion, optimizer,
            epoch, logger, k=k, compute_top_k=True,
            test_dataloader=test_dataloader, 
            accumulation_steps=accumulation_steps,
            log_images=log_images, log_externally=log_externally
        )
        
        if (epoch + 1) % config["val_frequency"] == 0:
            val_loss, top_k_accuracy_val = validate(
                dataloader=val_dataloader, model=model, 
                criterion=criterion, epoch=epoch+1, 
                logger=logger, k=k, test_dataloader=test_dataloader,
                log_externally=log_externally
            )

            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            print(f" Top-k Accuracy (Train): {top_k_accuracy_train:.4f}, Top-k Accuracy (Val): {top_k_accuracy_val:.4f}")
            if log_externally == 1: # log to wandb
                wandb.log({"Loss/Train": train_loss, "epoch": epoch+1})
                wandb.log({"Loss/Validation": val_loss, "epoch": epoch+1})
            
            if log_externally == 2: # log to tensorboard
                logger.add_scalar("Loss/Train", train_loss, epoch+1)
                logger.add_scalar("Loss/Validation", val_loss, epoch+1)
                
            # Save the model
            if val_loss < best_vloss:
                best_vloss = val_loss
                
                save_best_model(epoch+1, model, train_loss, top_k_accuracy_train, val_loss, top_k_accuracy_val, model_path, model_dir)

        scheduler.step()
        # scheduler.step(val_loss)
        
    if log_externally == 1: # log to wandb
        wandb.finish()
    if log_externally == 2:
        logger.close()

    print(f'{model_name}: Training completed successfully...')
   
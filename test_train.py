import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm

from dataset.hyperspectral_ds_lmdb import HyperspectralPatchLmdbDataset
from models.squeeze_excitation import SqueezeExcitation
from models.hyperkon_2D_3D import HyperKon_2D_3D
from models.resnext_3D import resnext101, resnext50
from loss_functions.kon_losses import NTXentLoss, ContrastiveLoss



channels = 224
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
batch_size = 5
learning_rate = 1e-2

height = width = 64
# num_classes = 10
embedding_dim = 128
# model = HyperKon_2D_3D(channels, embedding_dim).to(device)
# model = resnext50(in_channels=channels, out_features=embedding_dim).to(device)
model = SqueezeExcitation(channels, embedding_dim).to(device)

criterion = ContrastiveLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# initialise random tensor of size (batch_size, channels, height, width)
anchor = torch.rand(batch_size, channels, height, width)
positive = torch.rand(batch_size, channels, height, width)
# Training loop
for epoch in range(epochs):
    
    
    anchor, positive = anchor.to(device), positive.to(device)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    a_output = model(anchor)
    p_output = model(positive)

    a_output = F.normalize(a_output)
    p_output = F.normalize(p_output)
    
    loss = criterion(a_output, p_output)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}")

print("Finished Training")

import os
import lmdb
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from dataset.hyperspectral_ds_lmdb import HyperspectralPatchLmdbDataset

@pytest.fixture(scope="module")
def sample_lmdb():
    lmdb_dir = r"C:\Project\Surrey\Code\hyperkon\database"
    lmdb_path = os.path.join(lmdb_dir, "sample.lmdb")
    
    env = lmdb.open(lmdb_path, map_size=int(1e9))
    
    sample_data = {
        "CHW_224_672_0_224_224_32_32": np.random.rand(224, 32, 32).astype(np.float32),
        "CHW_224_772_0_224_224_64_64": np.random.rand(224, 64, 64).astype(np.float32),
    }
    
    with env.begin(write=True) as txn:
        for key, value in sample_data.items():
            txn.put(key.encode(), value.tobytes())
    
    yield sample_data, lmdb_dir
    
    env.close()

def test_hyperspectral_patch_lmdb_dataset(sample_lmdb):
    sample_data, lmdb_path = sample_lmdb
    keys = list(sample_data.keys())
    dataset = HyperspectralPatchLmdbDataset(keys, lmdb_path, "sample.lmdb", device=torch.device('cpu'))
    
    assert len(dataset) == len(sample_data)
    
    for idx, (anchor, positive) in enumerate(DataLoader(dataset, batch_size=1)):
        key = keys[idx]
        original_patch = sample_data[key]
        
        assert anchor.shape == torch.Size((1, dataset.channels, *original_patch.shape[1:]))
        assert positive.shape == anchor.shape
        
        # anchor_np = anchor.squeeze().cpu().numpy()
        # assert np.allclose(anchor_np, original_patch, atol=1e-6)
        

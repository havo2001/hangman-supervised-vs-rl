from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.cnn_transformers_create_training_examples import *

class MaskedCharDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, mmap_mode='r')
        # convert to torch tensors
        self.toks = torch.from_numpy(data['toks']).long() # shape (N, 40)
        self.labels = torch.from_numpy(data['labels']).long() # shape (N)
        self.mask_pos = torch.from_numpy(data['mas_pos']).long()   # shape (N)
        self.pad_mask = torch.from_numpy(data['pad_mask']).bool()  # shape (N, 40)

    def __len__(self):
        return self.toks.size(0)
    
    def __getitem__(self, idx):
        return (self.toks[idx], 
                self.labels[idx], 
                self.mask_pos[idx],
                self.pad_mask[idx])
    

def collate_fn(batch):
    """
    Turns a list of N samples into one batch
    Returns:
        tokens: shape (B, 40) - integers 0-28
        labels: shape (B)     - integers 1-26 
        maskpos: shape (B)    - integers 0-39 
        pad_mask: shape (B, 40) bool (True for padding positions)
    """
    tokens, labels, maskpos, pad_mask = zip(*batch)
    tokens = torch.stack(tokens)
    labels = torch.stack(labels)
    maskpos = torch.stack(maskpos)
    pad_mask = torch.stack(pad_mask)
    return tokens, labels, maskpos, pad_mask

def make_train_dataloader(npz_path, batch_size=512, num_workers=4, pin_memory=True) -> DataLoader:
    """
    Create a DataLoader for the masked character dataset.
    
    Args:
        npz_path (str): Path to the .npz file containing the dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        
    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = MaskedCharDataset(npz_path)
    return DataLoader(dataset, 
                      batch_size=batch_size,
                      shuffle=True, 
                      collate_fn=collate_fn, 
                      num_workers=num_workers,
                      pin_memory=pin_memory)



# if __name__ == "__main__":
#     dataset = make_train_dataloader('data/dataset.npz', batch_size=2, num_workers=0)
#     for toks, labels, maskpos, pad_mask in dataset:
#         print('Tokens:', toks)
#         print('Labels:', labels)
#         print('Mask positions:', maskpos)
#         print('Padding mask:', pad_mask)
#         break
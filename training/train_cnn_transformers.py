import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from contextlib import nullcontext

import argparse
import time

from model.model_cnn_transformers import *
from utils.cnn_transformers_dataloader import *

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def maybe_autocast(device, use_amp=False):
    if use_amp and device.type == 'mps':
        return torch.autocast(device_type='mps', dtype=torch.float16)
    return nullcontext()

def run_epoch(model, dataloader, criterion, optimizer, device, use_amp):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(dataloader, leave=False)
    for tokens, labels, maskpos, pad_mask in pbar:
        tokens, labels, maskpos, pad = (
            tokens.to(device),
            labels.to(device),
            maskpos.to(device),
            pad_mask.to(device) 
        )
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, use_amp):
            logits = model(tokens, maskpos, pad)
            loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        loss_sum += loss.item() * labels.size(0)
        pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {correct / total:.4%}")

    return correct / total, loss_sum / total


def train_model(args):
    device = get_device()
    print(f'trainging on {device} (AMP={args.amp})')

    # Load dataset
    loader = make_train_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type != 'cpu')
    )

    # Model
    model = build_model(d_model=args.d_model, num_encoder_layers=args.num_encoder_layers).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Training loop
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        acc, loss = run_epoch(model, loader, criterion, optimizer, device, args.amp)
        dt = time.perf_counter() - t0
        print(f'Epoch {epoch+1}/{args.epochs} - acc={acc:.4%}, loss={loss:.4f}, time={dt:.2f}s')

        # Save the model checkpoint
        # create the folder if not exists yet
        os.makedirs('model/checkpoint', exist_ok=True)
        checkpoint = f'model/checkpoint/cnn_transformers_checkpoint_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint)
        print(f'Model checkpoint saved to {checkpoint}')

    print('Training complete!')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/dataset.npz', help='Path to the dataset .npz file')
    ap.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    ap.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    ap.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    ap.add_argument('--d_model', type=int, default=256, help='Dimension of the embdedding')
    ap.add_argument('--num_encoder_layers', type=int, default=4, help='Number of Transformer encoder layers')
    ap.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    ap.add_argument('--amp', action='store_true', help='Use automatic mixed precision (AMP) for training')
    args = ap.parse_args()

    train_model(args)








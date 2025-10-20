#!/usr/bin/env python3
"""
train_guepier_cnn.py

Trains a small CNN classifier on 2s .wav chunks saved in:
 - guepier_sounds  (label 1)
 - other_sounds    (label 0)
 - other_noise     (label 0)

Prints accuracy and F4 after each epoch (train + val).
"""

import os
import random
from pathlib import Path
import math
import time
import argparse

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# -------------------------
# Utils: F-beta metric
# -------------------------
def fbeta_score_from_preds(y_true, y_pred, beta=4.0, eps=1e-8):
    # y_true, y_pred are numpy arrays of 0/1
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp + eps)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn + eps)
    b2 = beta * beta
    if precision + recall == 0:
        return 0.0
    fbeta = (1.0 + b2) * precision * recall / (b2 * precision + recall + eps)
    return fbeta

# -------------------------
# Dataset
# -------------------------
class ChunkDataset(Dataset):
    def __init__(self, filepaths, labels, sr=32000, n_mels=64, n_fft=1024, hop_length=256, transform=None):
        """
        filepaths: list of .wav file paths
        labels: list of 0/1 ints
        transform: optional function applied to (mel_db) for augmentation
        """
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        # load audio - 2 sec chunk expected
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        # ensure length is exactly 2s by padding or trimming if needed
        desired_len = int(2.0 * self.sr)
        if len(y) < desired_len:
            y = np.pad(y, (0, desired_len - len(y)))
        elif len(y) > desired_len:
            y = y[:desired_len]

        # compute mel spectrogram (power)
        S = librosa.feature.melspectrogram(y=y,
                                           sr=self.sr,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           n_mels=self.n_mels,
                                           power=2.0)
        # log-mels
        mel_db = librosa.power_to_db(S, ref=np.max)
        # Normalize per-sample (optional; helps training)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

        # shape -> (n_mels, time). Convert to float32 and channel-first
        x = mel_db.astype(np.float32)
        # Optional transform (data augmentation)
        if self.transform is not None:
            x = self.transform(x)

        x = np.expand_dims(x, axis=0)  # (1, n_mels, time)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)

# -------------------------
# Small CNN model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, n_mels=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # use adaptive pooling to handle time dimension variability
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # global pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (2,2))
        x = self.pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.dropout(x)
        x = self.fc(x)  # (B, 1)
        return x.squeeze(1)  # logits

# -------------------------
# Training utilities
# -------------------------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)  # (B,)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * xb.shape[0]
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.append(preds)
        all_labels.append(yb.detach().cpu().numpy().astype(int))
    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    f4 = fbeta_score_from_preds(all_labels, all_preds, beta=4.0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, f4

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * xb.shape[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_labels.append(yb.cpu().numpy().astype(int))
    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    f4 = fbeta_score_from_preds(all_labels, all_preds, beta=4.0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, f4

# -------------------------
# Main training script
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/automathon/users/mig_user3/Automathon_Sujet/dataset/chunks_dataset",
                        help="root folder that contains guepier_sounds, other_sounds, other_noise")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data_root)
    guepier_dir = root / "guepier_sounds"
    other1 = root / "other_sounds"
    other2 = root / "other_noise"

    # collect filepaths and labels
    pos_files = sorted([str(p) for p in guepier_dir.glob("*.wav")])
    neg_files = sorted([str(p) for p in other1.glob("*.wav")] + [str(p) for p in other2.glob("*.wav")])

    print(f"Positive (guepier_sounds): {len(pos_files)}")
    print(f"Negative (other_sounds + other_noise): {len(neg_files)}")

    all_files = pos_files + neg_files
    all_labels = [1] * len(pos_files) + [0] * len(neg_files)

    # split train/val stratified
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=args.val_ratio, random_state=args.seed, stratify=all_labels)

    print(f"Train size: {len(train_files)}, Val size: {len(val_files)}")

    # create datasets
    train_ds = ChunkDataset(train_files, train_labels)
    val_ds = ChunkDataset(val_files, val_labels)

    # create sampler to address imbalance (oversample minority)
    # compute weights per sample: inverse of class frequency
    class_sample_count = np.array([ (np.array(train_labels) == c).sum() for c in [0,1] ])
    # weight for each class = 1 / count
    class_weights = {0: 1.0 / (class_sample_count[0] + 1e-12), 1: 1.0 / (class_sample_count[1] + 1e-12)}
    sample_weights = np.array([ class_weights[label] for label in train_labels ], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(in_ch=1).to(device)

    # pos_weight for BCEWithLogitsLoss: give higher weight to positive class
    # pos_weight = (#neg examples) / (#pos examples)
    pos_count = float((np.array(train_labels) == 1).sum())
    neg_count = float((np.array(train_labels) == 0).sum())
    pos_weight = torch.tensor([ (neg_count / (pos_count + 1e-12)) ], device=device)
    print(f"Using pos_weight = {float(pos_weight.item()):.3f} in BCEWithLogitsLoss")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

    # training loop with best model saving
    best_val_f4 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f4 = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc, val_f4 = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | time {time.time()-t0:.1f}s")
        print(f"  Train loss {train_loss:.4f}  acc {train_acc:.4f}  F4 {train_f4:.4f}")
        print(f"  Val   loss {val_loss:.4f}  acc {val_acc:.4f}  F4 {val_f4:.4f}")
        print("-"*60)

        # ✅ Save best model so far
        if val_f4 > best_val_f4:
            best_val_f4 = val_f4
            best_epoch = epoch
            torch.save(model.state_dict(), "guepier_cnn_best.pth")
            print(f"  ✅ New best model saved (epoch {epoch}, F4={val_f4:.4f})")

    print(f"\nTraining finished. Best model at epoch {best_epoch} with F4={best_val_f4:.4f}")
    print("Saved as guepier_cnn_best.pth")


if __name__ == "__main__":
    main()

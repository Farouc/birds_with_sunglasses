import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# ================= CONFIG =================
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
CACHE_DIR = "/automathon/users/mig_user3/Automathon_Sujet/spectrogram_cache"  # path where .npy files were saved

# ================= DATA LOADING =================
def load_cached_data(cache_dir=CACHE_DIR):
    spec_files = sorted([
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.endswith(".npy") and f != "labels.npy"
    ])
    X = np.array([np.load(f) for f in tqdm(spec_files, desc="Loading cached spectrograms")])
    y = np.load(os.path.join(cache_dir, "labels.npy"))
    print(f"[INFO] Loaded {len(X)} spectrograms from cache.")
    return X, y

# ================= TORCH DATASET =================
class AudioDataset(Dataset):
    def __init__(self, X, y):
        # Normalize each spectrogram
        X = (X - X.mean(axis=(1, 2, 3), keepdims=True)) / (X.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ================= MODEL =================
class CNNBiGRU(nn.Module):
    def __init__(self, n_mels=128):
        super(CNNBiGRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(
            input_size=(n_mels // 16) * 256,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)
        _, h = self.gru(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        out = torch.sigmoid(self.fc(h))
        return out.squeeze()

# ================= LOSS =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()

# ================= TRAINING WITH BEST MODEL SAVE =================
def train_model(model, train_loader, val_loader, optimizer, criterion, save_path="best_model.pt"):
    model.to(DEVICE)
    best_recall = 0.0  # track the best validation recall
    
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        
        # ===== TRAIN =====
        model.train()
        train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"Batch {i+1}/{len(train_loader)} | Train Loss: {train_loss/((i+1)*BATCH_SIZE):.4f}")

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)
                all_preds.append(y_pred.cpu())
                all_labels.append(y_batch.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        preds_binary = (all_preds >= 0.5).float()

        tp = ((preds_binary == 1) & (all_labels == 1)).sum().item()
        fn = ((preds_binary == 0) & (all_labels == 1)).sum().item()
        recall = tp / (tp + fn + 1e-8)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader.dataset):.4f} | "
              f"Val Loss: {val_loss/len(val_loader.dataset):.4f} | "
              f"Val Recall: {recall:.4f}")

        # ===== SAVE BEST MODEL =====
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Best model saved with Val Recall: {best_recall:.4f}")

    print(f"\nTraining completed. Best Val Recall: {best_recall:.4f}")
    return model


# ================= MAIN =================
if __name__ == "__main__":
    X, y = load_cached_data(CACHE_DIR)
    X, y = shuffle(X, y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = CNNBiGRU()
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trained_model = train_model(model, train_loader, val_loader, optimizer, criterion)
    torch.save(trained_model.state_dict(), "guepier_classifier_torch.pt")

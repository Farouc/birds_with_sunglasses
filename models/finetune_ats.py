import os
import glob
import torch
import torchaudio
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import recall_score
from collections import Counter

# ======================
# 1️⃣ Dataset
# ======================
class BirdDataset(Dataset):
    def __init__(self, base_dir, feature_extractor, sr=16000):
        self.samples, self.labels = [], []
        self.sr = sr
        self.feat_ext = feature_extractor

        # Regroupement : 3 classes
        self.label_map = {
            "guepier_sounds": 0,
            "other_sounds": 1,
            "guepier_noise": 2,
            "other_noise": 2,
        }

        for cls_dir, label_idx in self.label_map.items():
            folder = os.path.join(base_dir, cls_dir)
            if not os.path.exists(folder):
                continue
            wavs = sorted(glob.glob(os.path.join(folder, "*.wav")))
            for w in wavs:
                self.samples.append(w)
                self.labels.append(label_idx)

        print(f"[INFO] Found {len(set(self.labels))} classes and {len(self.samples)} files total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        waveform, sr = sf.read(path, dtype="float32")
        waveform = torch.tensor(waveform)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)

        inputs = self.feat_ext(waveform, sampling_rate=self.sr, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, torch.tensor(label)

# ======================
# 2️⃣ Utils
# ======================
def compute_class_weights(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = {cls: total / (len(counter) * count) for cls, count in counter.items()}
    ordered_weights = [weights[i] for i in range(len(counter))]
    return torch.tensor(ordered_weights, dtype=torch.float)

# ---------- FOCAL LOSS ----------
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ======================
# 3️⃣ Config
# ======================
base_dir = "/automathon/users/mig_user3/Automathon_Sujet/dataset/chunks_dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
num_batches_to_train = 20000
print_interval = 100
save_interval = 1000
lr = 1e-5

# ======================
# 4️⃣ Dataset & Model
# ======================
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
dataset = BirdDataset(base_dir, feature_extractor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Class weights (α in focal loss)
class_weights = compute_class_weights(dataset.labels).to(device)
print(f"[INFO] Class weights: {class_weights}")

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Freeze tout sauf dernière couche + LayerNorm + classifieur
for param in model.parameters():
    param.requires_grad = False
encoder_layers = model.audio_spectrogram_transformer.encoder.layer
for param in encoder_layers[-1].parameters():
    param.requires_grad = True
for param in model.audio_spectrogram_transformer.layernorm.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# ======================
# 5️⃣ Entraînement (streaming)
# ======================
save_dir = "models_ast_focal"
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, "ast_bird_best.pt")
last_model_path = os.path.join(save_dir, "ast_bird_last.pt")

best_recall_guepier = 0.0
running_loss = 0.0
batches_done = 0

model.train()
print(f"[INFO] Starting streaming fine-tuning for {num_batches_to_train} batches...")

while batches_done < num_batches_to_train:
    for batch, labels in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)

        outputs = model(**batch)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches_done += 1

        # Print moyenne chaque 100 batches
        if batches_done % print_interval == 0:
            avg_loss = running_loss / print_interval
            print(f"[Batch {batches_done}] mean_focal_loss={avg_loss:.4f}")
            running_loss = 0.0

        # Save + mini-validation tous les 1000 batches
        if batches_done % save_interval == 0:
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for _ in range(5):  # échantillon rapide
                    val_batch, val_labels = next(iter(dataloader))
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    val_labels = val_labels.to(device)
                    val_outputs = model(**val_batch)
                    preds = val_outputs.logits.argmax(dim=1)
                    y_true.extend(val_labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            recall_guepier = recall_score(y_true, y_pred, labels=[0], average='macro')
            print(f"🔍 Validation mini-batch | Recall(guepier)={recall_guepier:.3f}")

            # Sauvegarde checkpoints
            torch.save(model.state_dict(), last_model_path)
            if recall_guepier > best_recall_guepier:
                best_recall_guepier = recall_guepier
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] 🌟 New best model saved (recall_guepier={best_recall_guepier:.3f})")

            model.train()

        if batches_done >= num_batches_to_train:
            break

print(f"[INFO] ✅ Training complete after {batches_done} batches. Best recall(guepier)={best_recall_guepier:.3f}")

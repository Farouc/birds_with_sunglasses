import os
import glob
import random
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# ======================
# 1️⃣ Dataset (utilise TOUS les fichiers)
# ======================
class BirdDataset(Dataset):
    def __init__(self, base_dir, feature_extractor, sr=16000):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.sr = sr
        self.feat_ext = feature_extractor

        subdirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
        print(f"[INFO] Found {len(subdirs)} species")

        for label_idx, sp_dir in enumerate(subdirs):
            species = os.path.basename(sp_dir)
            wavs = sorted(glob.glob(os.path.join(sp_dir, "*.wav")))
            if not wavs:
                continue
            self.label_map[label_idx] = species

            for w in wavs:
                self.samples.append(w)
                self.labels.append(label_idx)

        print(f"[INFO] Loaded {len(self.samples)} audio files from {len(self.label_map)} species")

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
# 2️⃣ Initialisation
# ======================
# base_dir = "/automathon/users/mig_user3/Automathon_Sujet/dataset/chunks_dataset"
base_dir = "/automathon/users/mig_user3/Automathon_Sujet/dataset/cleaned_bird_songs"
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

dataset = BirdDataset(base_dir, feature_extractor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(dataset.label_map),
    ignore_mismatched_sizes=True
)

# ======================
# 3️⃣ Fine-tuning sélectif : dernières couches uniquement
# ======================
for param in model.parameters():
    param.requires_grad = False

num_layers_to_unfreeze = 2
encoder_layers = model.audio_spectrogram_transformer.encoder.layer
num_total_layers = len(encoder_layers)

for i in range(num_total_layers - num_layers_to_unfreeze, num_total_layers):
    print(f"🔓 Déblocage de la couche ASTLayer n°{i}")
    for param in encoder_layers[i].parameters():
        param.requires_grad = True

for param in model.audio_spectrogram_transformer.layernorm.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

# ======================
# 4️⃣ Contrastive loss
# ======================
def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    emb = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(emb, emb.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()

def supervised_contrastive_loss_weighted(embeddings, labels, temperature=0.07):
    emb = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(emb, emb.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    # --- 🧩 pondération par classe ---
    unique_labels, counts = torch.unique(labels, return_counts=True)
    freq = torch.zeros_like(labels, dtype=torch.float)
    for ul, c in zip(unique_labels, counts):
        freq[labels == ul] = c.float()
    inv_freq = 1.0 / (freq + 1e-8)
    weights = inv_freq / inv_freq.sum() * len(labels)  # normalisation douce

    exp_sim = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    # --- appliquer les poids inverses ---
    loss = -(weights * mean_log_prob_pos).sum() / weights.sum()
    return loss

# ======================
# 5️⃣ Entraînement + Checkpointing
# ======================
num_epochs = 100
checkpoint_interval = 10
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

best_model_path = os.path.join(save_dir, "ast_bird_best.pt")
last_checkpoint_path = os.path.join(save_dir, "ast_bird_last.pt")

start_epoch = 0
best_loss = float("inf")

# 🔁 Reprise automatique si checkpoint existe
if os.path.exists(last_checkpoint_path):
    print(f"[INFO] 🔄 Resuming training from {last_checkpoint_path}")
    checkpoint = torch.load(last_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    print(f"[INFO] Resumed at epoch {start_epoch} (best_loss={best_loss:.4f})")

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0.0
    batch_loss_accum = 0.0
    count_since_last_log = 0

    for batch_idx, (batch, labels) in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)

        outputs = model(**batch, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1)
        loss = supervised_contrastive_loss_weighted(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate for mean logging
        batch_loss_accum += loss.item()
        total_loss += loss.item()
        count_since_last_log += 1

        # --- afficher une moyenne tous les 100 batchs ---
        if batch_idx % 100 == 0 or batch_idx == len(dataloader):
            avg_100_loss = batch_loss_accum / count_since_last_log
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | mean(100) loss={avg_100_loss:.4f}")
            batch_loss_accum = 0.0
            count_since_last_log = 0

    mean_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}/{num_epochs} | mean_epoch_loss={mean_loss:.4f}")

    # ✅ Sauvegarde checkpoint
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "label_map": dataset.label_map,
    }
    torch.save(checkpoint_data, last_checkpoint_path)

    # ✅ Meilleur modèle
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(checkpoint_data, best_model_path)
        print(f"[INFO] 🌟 New best model saved at epoch {epoch+1} | loss={best_loss:.4f}")

    # ✅ Checkpoint périodique
    if (epoch + 1) % checkpoint_interval == 0:
        path = os.path.join(save_dir, f"ast_bird_epoch{epoch+1}.pt")
        torch.save(checkpoint_data, path)
        print(f"[INFO] 💾 Checkpoint saved at {path}")

print(f"[INFO] ✅ Training complete. Best model: {best_model_path} (loss={best_loss:.4f})")

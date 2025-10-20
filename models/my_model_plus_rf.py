import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import torchaudio
import glob, os
from torch.utils.data import Dataset

# ======================
# 1️⃣ Dataset (même logique que ton training)
# ======================
class BirdDataset(Dataset):
    def __init__(self, base_dir, feature_extractor, sr=16000):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.sr = sr
        self.feat_ext = feature_extractor

        subdirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
        for label_idx, sp_dir in enumerate(subdirs):
            species = os.path.basename(sp_dir)
            wavs = sorted(glob.glob(os.path.join(sp_dir, "*.wav")))
            if not wavs:
                continue
            self.label_map[label_idx] = species
            for w in wavs:
                self.samples.append(w)
                self.labels.append(label_idx)

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
# 2️⃣ Charger modèle fine-tuné
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "../models/ast_bird_best.pt"  # ou ast_bird_last.pt
checkpoint = torch.load(checkpoint_path, map_location=device)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "../models/ast_bird_best.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Recréer le feature extractor
from transformers import ASTFeatureExtractor
feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
label_map = checkpoint["label_map"]

# Charger le modèle fine-tuné
from transformers import ASTForAudioClassification
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(label_map),
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("[INFO] ✅ Model and feature extractor loaded successfully")


# ======================
# 3️⃣ Extraire les embeddings
# ======================
base_dir = "/automathon/users/mig_user3/Automathon_Sujet/dataset/cleaned_bird_songs"
dataset = BirdDataset(base_dir, feature_extractor)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

emb_list, y_list = [], []
with torch.no_grad():
    for batch, labels in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_hidden_states=True)
        emb = outputs.hidden_states[-1].mean(dim=1)  # moyenne temporelle
        emb_list.append(emb.cpu().numpy())
        y_list.append(labels.numpy())

X = np.vstack(emb_list)
y = np.concatenate(y_list)

print(f"[INFO] Embeddings shape: {X.shape}, Labels: {len(np.unique(y))}")

# ======================
# 4️⃣ Entraîner un Random Forest sur les embeddings
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

# ======================
# 5️⃣ Calculer la loss et accuracy sur le train
# ======================
y_pred_proba = rf.predict_proba(X_scaled)
y_pred = rf.predict(X_scaled)

acc = accuracy_score(y, y_pred)
rf_loss = log_loss(y, y_pred_proba)

print(f"[RESULT] Random Forest accuracy (train): {acc*100:.2f}%")
print(f"[RESULT] Random Forest log-loss (train): {rf_loss:.4f}")

# ======================
# 6️⃣ Sauvegarde du Random Forest et du scaler
# ======================
import joblib
os.makedirs("../models", exist_ok=True)

rf_path = "../models/random_forest_bird.pkl"
scaler_path = "../models/scaler_bird.pkl"

joblib.dump(rf, rf_path)
joblib.dump(scaler, scaler_path)

print(f"[INFO] ✅ Random Forest saved to {rf_path}")
print(f"[INFO] ✅ Scaler saved to {scaler_path}")

import os
import json
import glob
import math
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# =====================
# CONFIG
# =====================
# AUDIO_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation/"
AUDIO_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset/parc_audios/data/"
CHECKPOINT_DIR = "./models_ast"              # dossier où sont les checkpoints
BEST_PATTERN = "ast_bird_best.pt"
LAST_PATTERN = "ast_bird_last.pt"
OUT_JSON = "pred_timestamps_ast.json"

# Fenêtrage
WINDOW_SEC = 2.0
HOP_SEC = 1.0
SR_AST = 16000
BATCH_SIZE = 32  # augmente si la VRAM le permet

# Classes du modèle fine-tuné (ordre utilisé au training)
CLASS_NAMES = ["guepier_sounds", "other_sounds", "noise"]
TARGET_CLASS = "guepier_sounds"
THR_AST = 0.50  # seuil de proba pour déclarer "guepier présent" sur une fenêtre

# =====================
# UTILS
# =====================
def pick_checkpoint(ckpt_dir: str, best_name: str, last_name: str) -> str:
    best_path = os.path.join(ckpt_dir, best_name)
    last_path = os.path.join(ckpt_dir, last_name)
    if os.path.exists(best_path):
        print(f"[INFO] Using BEST checkpoint: {best_path}")
        return best_path
    if os.path.exists(last_path):
        print(f"[INFO] Using LAST checkpoint: {last_path}")
        return last_path
    raise FileNotFoundError(f"Aucun checkpoint trouvé dans {ckpt_dir} "
                            f"(cherché {best_name} puis {last_name}).")

def list_audio_files(folder: str):
    exts = (".wav", ".ogg")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def chunk_audio(y: np.ndarray, sr: int, win_sec: float, hop_sec: float):
    """Découpe un long audio en fenêtres (overlap hop_sec). Renvoie (chunks, starts_sec)."""
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if len(y) < win:
        return np.empty((0, win), dtype=np.float32), np.array([], dtype=np.float32)
    starts = np.arange(0, len(y) - win + 1, hop, dtype=int)
    chunks = np.stack([y[s:s + win] for s in starts])
    times = starts.astype(np.float32) / sr
    return chunks, times

def merge_intervals(starts_sec: np.ndarray, mask: np.ndarray, win_sec: float):
    """Fusionne les fenêtres contiguës positives en intervalles [t_start, t_end]."""
    intervals = []
    start = None
    for i, is_pos in enumerate(mask):
        if is_pos and start is None:
            start = float(starts_sec[i])
        elif (not is_pos) and (start is not None):
            end = float(starts_sec[i - 1] + win_sec)
            intervals.append([start, end])
            start = None
    if start is not None and len(starts_sec) > 0:
        intervals.append([start, float(starts_sec[-1] + win_sec)])
    return intervals

# =====================
# LOAD MODEL
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

ckpt_path = pick_checkpoint(CHECKPOINT_DIR, BEST_PATTERN, LAST_PATTERN)

# NB: le script d'entraînement sauvait uniquement state_dict()
# On reconstruit l'architecture puis on charge le state_dict.
feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
)
state = torch.load(ckpt_path, map_location=device)
# state peut être soit un dict direct (state_dict), soit un dict wrap (model_state_dict)
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
elif isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)

model.to(device)
model.eval()
print("[INFO] ✅ AST model loaded.")

# index de la classe cible
try:
    target_idx = CLASS_NAMES.index(TARGET_CLASS)
except ValueError:
    raise ValueError(f"Classe '{TARGET_CLASS}' absente de CLASS_NAMES={CLASS_NAMES}")

# =====================
# MAIN
# =====================
results = {"audios": {}, "next_id": None}
next_id = 1

audio_files = list_audio_files(AUDIO_DIR)
if not audio_files:
    print(f"[WARN] Aucun fichier .wav/.ogg trouvé dans {AUDIO_DIR}")

for fname in tqdm(audio_files, desc="Scanning audios"):
    audio_id = os.path.splitext(fname)[0]
    path = os.path.join(AUDIO_DIR, fname)

    # Chargement direct à 16 kHz (évite un resample séparé)
    try:
        y, _ = librosa.load(path, sr=SR_AST, mono=True)
    except Exception as e:
        print(f"[ERROR] Lecture impossible: {fname} -> {e}")
        results["audios"][str(audio_id)] = {"id": str(audio_id), "timestamps": []}
        continue

    chunks, times = chunk_audio(y, SR_AST, WINDOW_SEC, HOP_SEC)
    if chunks.shape[0] == 0:
        results["audios"][str(audio_id)] = {"id": str(audio_id), "timestamps": []}
        # sauvegarde immédiate
        results["next_id"] = next_id
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
        continue

    # Prédiction par batch
    probs_target = []
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_np = chunks[i:i + BATCH_SIZE]
            # feature extractor sur LISTE -> padding + tensorisation
            inputs = feature_extractor(
                list(batch_np),
                sampling_rate=SR_AST,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs)
            logits = out.logits  # [B, 3]
            prob = torch.softmax(logits, dim=1)[:, target_idx]  # proba "guepier_sounds"
            probs_target.append(prob.detach().cpu().numpy())

    probs_target = np.concatenate(probs_target, axis=0)
    positive_mask = probs_target >= THR_AST

    # Fusion temporelle
    intervals = merge_intervals(times, positive_mask, WINDOW_SEC)
    results["audios"][str(audio_id)] = {
        "id": str(audio_id),
        "timestamps": [[float(s), float(e)] for (s, e) in intervals]
    }

    # next_id incrémental si l’ID est numérique
    try:
        numeric_id = int(audio_id)
        next_id = max(next_id, numeric_id + 1)
    except ValueError:
        pass

    # Écriture du JSON au fil de l’eau
    results["next_id"] = next_id
    try:
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"[WARN] Échec sauvegarde JSON intermédiaire pour {fname}: {e}")

print(f"[INFO] ✅ Terminé. Résultats dans {OUT_JSON}")

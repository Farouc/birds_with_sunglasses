import os
import json
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# ======================
# CONFIG
# ======================
AUDIO_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation/"
CHECKPOINT_DIR = "./models_ast_focal"
BEST_PATTERN = "ast_bird_best.pt"
LAST_PATTERN = "ast_bird_last.pt"
OUT_JSON = "pred_timestamps_final_validation_ats_focal.json"

WINDOW_SEC = 2.0
HOP_SEC = 1.0
SR_AST = 16000
BATCH_SIZE = 32
THR_AST = 0.5  # seuil de détection

CLASS_NAMES = ["guepier_sounds", "other_sounds", "noise"]
TARGET_CLASS = "guepier_sounds"

# ======================
# UTILS
# ======================
def pick_checkpoint(ckpt_dir, best, last):
    best_path = os.path.join(ckpt_dir, best)
    last_path = os.path.join(ckpt_dir, last)
    if os.path.exists(best_path):
        print(f"[INFO] Using BEST checkpoint: {best_path}")
        return best_path
    if os.path.exists(last_path):
        print(f"[INFO] Using LAST checkpoint: {last_path}")
        return last_path
    raise FileNotFoundError("No checkpoint found")

def chunk_audio(y, sr, win_sec, hop_sec):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if len(y) < win:
        return np.empty((0, win), dtype=np.float32), np.array([])
    starts = np.arange(0, len(y) - win + 1, hop)
    chunks = np.stack([y[s:s+win] for s in starts])
    times = starts / sr
    return chunks, times

def merge_intervals(starts_sec, mask, win_sec):
    intervals, start = [], None
    for i, val in enumerate(mask):
        if val and start is None:
            start = float(starts_sec[i])
        elif not val and start is not None:
            end = float(starts_sec[i-1] + win_sec)
            intervals.append([start, end])
            start = None
    if start is not None and len(starts_sec) > 0:
        intervals.append([float(start), float(starts_sec[-1] + win_sec)])
    return intervals

def list_audio_files(folder):
    exts = (".wav", ".ogg")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

# ======================
# LOAD MODEL
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

ckpt_path = pick_checkpoint(CHECKPOINT_DIR, BEST_PATTERN, LAST_PATTERN)

feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True,
)
state = torch.load(ckpt_path, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.to(device).eval()
target_idx = CLASS_NAMES.index(TARGET_CLASS)
print("[INFO] ✅ AST model ready.")

# ======================
# MAIN LOOP
# ======================
results = {"audios": {}, "next_id": None}
next_id = 1

audio_files = list_audio_files(AUDIO_DIR)
print(f"[INFO] Found {len(audio_files)} files.")

for fname in tqdm(audio_files, desc="Predicting"):
    # 🔹 Extraire identifiant numérique du fichier
    raw_id = os.path.splitext(fname)[0]
    try:
        numeric_id = int(raw_id.replace("audio_", ""))  # gère "audio_12" -> 12
        audio_id_str = str(numeric_id)
    except ValueError:
        # fallback : nom brut
        audio_id_str = raw_id

    file_path = os.path.join(AUDIO_DIR, fname)

    try:
        y, _ = librosa.load(file_path, sr=SR_AST, mono=True)
    except Exception as e:
        print(f"[WARN] Could not read {fname}: {e}")
        results["audios"][audio_id_str] = {"id": audio_id_str, "timestamps": []}
        continue

    chunks, times = chunk_audio(y, SR_AST, WINDOW_SEC, HOP_SEC)
    if chunks.shape[0] == 0:
        results["audios"][audio_id_str] = {"id": audio_id_str, "timestamps": []}
        continue

    # 🔹 prédictions par batch
    probs_target = []
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_np = chunks[i:i+BATCH_SIZE]
            inputs = feature_extractor(
                list(batch_np),
                sampling_rate=SR_AST,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            p = torch.softmax(out.logits, dim=1)[:, target_idx]
            probs_target.append(p.cpu().numpy())
    probs_target = np.concatenate(probs_target, axis=0)
    positive_mask = probs_target >= THR_AST

    # 🔹 fusion temporelle
    intervals = merge_intervals(times, positive_mask, WINDOW_SEC)
    results["audios"][audio_id_str] = {
        "id": audio_id_str,
        "timestamps": [[float(s), float(e)] for s, e in intervals],
    }

    # 🔹 gestion next_id
    try:
        numeric_id = int(audio_id_str)
        next_id = max(next_id, numeric_id + 1)
    except ValueError:
        pass

    # 🔹 sauvegarde progressive
    results["next_id"] = next_id
    try:
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"[WARN] Could not save intermediate JSON: {e}")

print(f"[INFO] ✅ Done. Final JSON saved to {OUT_JSON}")

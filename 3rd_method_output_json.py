#!/usr/bin/env python3
"""
generate_guepier_timestamps.py

Uses the trained SimpleCNN model to detect guepier songs in long recordings
and outputs a JSON file with start/end timestamps (in seconds).
"""

import os
import json
import librosa
import numpy as np
from tqdm import tqdm
import torch
from classifier_3rd_method import SimpleCNN  # ✅ use your exact model class

# -----------------------------
# Parameters
# -----------------------------
AUDIO_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset/parc_audios/wav_data"
OUTPUT_JSON = "/automathon/users/mig_user3/Automathon_Sujet/guepier_predictions.json"
MODEL_PATH = "/automathon/users/mig_user3/Automathon_Sujet/models/guepier_cnn_best.pth"

SR = 32000
CHUNK_SEC = 2.0
OVERLAP = 0.5   # 50% overlap
THRESHOLD = 0.4 # probability threshold for detection

# -----------------------------
# Helper: mel spectrogram computation
# -----------------------------
def audio_to_mel(y, sr=SR, n_mels=64, n_fft=1024, hop_length=256):
    desired_len = int(CHUNK_SEC * sr)
    if len(y) < desired_len:
        y = np.pad(y, (0, desired_len - len(y)))
    elif len(y) > desired_len:
        y = y[:desired_len]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(S, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    x = np.expand_dims(mel_db, axis=(0, 1))  # shape (1, 1, n_mels, T)
    return torch.tensor(x, dtype=torch.float32)

# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN(in_ch=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Process each long audio
# -----------------------------
results = {"audios": {}, "next_id": 1}
audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])

for idx, fname in enumerate(tqdm(audio_files, desc="Processing audios")):
    path = os.path.join(AUDIO_DIR, fname)
    y, sr = librosa.load(path, sr=SR, mono=True)
    total_len = len(y)
    step = int(SR * CHUNK_SEC * (1 - OVERLAP))
    chunk_len = int(SR * CHUNK_SEC)

    timestamps = []
    probs = []

    # Slide over the audio with overlap
    for start in range(0, total_len - chunk_len + 1, step):
        end = start + chunk_len
        chunk = y[start:end]
        x = audio_to_mel(chunk).to(device)
        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    # convert chunk indices to time intervals
    active = np.array(probs) >= THRESHOLD
    intervals = []
    i = 0
    while i < len(active):
        if active[i]:
            start_t = i * step / SR
            while i < len(active) and active[i]:
                i += 1
            end_t = start_t + CHUNK_SEC
            # merge overlapping detections by extending if close
            if len(intervals) > 0 and start_t - intervals[-1][1] < CHUNK_SEC:
                intervals[-1][1] = end_t
            else:
                intervals.append([float(start_t), float(end_t)])
        else:
            i += 1

    # Store results
    audio_id = str(idx + 1)
    results["audios"][audio_id] = {"id": audio_id, "timestamps": intervals}
    results["next_id"] = idx + 2

# -----------------------------
# Save JSON
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ JSON file saved to {OUTPUT_JSON}")

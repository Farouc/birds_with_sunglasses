import os
import json
import torch
import librosa
from pathlib import Path
from model_classif import CNNBiGRU
from panns_inference import AudioTagging, labels
import numpy as np

# -------------------------- CONFIG --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 22050
PANN_SR = 32000
CHUNK_DURATION = 2.0
OVERLAP = 0.5
THRESHOLD = 0.35
INPUT_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation"
OUTPUT_JSON = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation_results.json"

# -------------------------- AUDIO SEGMENTATION --------------------------
def segment_audio(y, sr, chunk_duration=CHUNK_DURATION, overlap=OVERLAP):
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))
    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, hop_samples):
        end = start + chunk_samples
        chunks.append((start / sr, end / sr, y[start:end]))
    return chunks

# -------------------------- MEL-SPECTROGRAM --------------------------
def audio_to_melspec(y, sr=SR, n_mels=128, hop_length=512, fmin=20, fmax=8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       hop_length=hop_length, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    delta = librosa.feature.delta(S_db)
    delta2 = librosa.feature.delta(S_db, order=2)
    return np.stack([S_db, delta, delta2], axis=0)

# -------------------------- CLASSIFIER PREDICTION --------------------------
def predict_chunk(model, chunk):
    model.eval()
    spec = audio_to_melspec(chunk)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = model(spec_tensor).item()
    return prob

# -------------------------- PANNs BIRD CHECK --------------------------
pann_model = AudioTagging(device="cuda" if torch.cuda.is_available() else "cpu")
bird_classes = [
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Crow",
    "Caw",
    "Owl"
]
bird_indices = [labels.index(c) for c in bird_classes if c in labels]
BIRD_THRESHOLD = 0.02

def is_bird_present(chunk, sr=SR):
    # Resample to PANN_SR if needed
    if sr != PANN_SR:
        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=PANN_SR)
    chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scores, _ = pann_model.inference(chunk_tensor)
    scores = scores.flatten()
    bird_score = sum([scores[idx] for idx in bird_indices])
    return bird_score > BIRD_THRESHOLD

# -------------------------- PREDICT LONG AUDIO --------------------------
def predict_long_audio_with_panns(model, audio_path):
    y, _ = librosa.load(audio_path, sr=SR)
    chunks = segment_audio(y, SR)
    timestamps = []

    for start_time, end_time, chunk in chunks:
        if not is_bird_present(chunk, sr=SR):
            continue
        prob = predict_chunk(model, chunk)
        if prob >= THRESHOLD:
            timestamps.append([start_time, end_time])
    return timestamps

def merge_timestamps(timestamps, max_gap=2.0):
    """
    Merge overlapping or nearby timestamps.
    - timestamps: list of [start, end]
    - max_gap: maximum gap (seconds) allowed between intervals to merge
    """
    if not timestamps:
        return []

    # Sort by start time
    timestamps = sorted(timestamps, key=lambda x: x[0])
    merged = [timestamps[0]]

    for start, end in timestamps[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:  # merge
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged




# -------------------------- MAIN --------------------------
if __name__ == "__main__":
    # Load classifier
    classifier = CNNBiGRU()
    classifier.load_state_dict(torch.load("/automathon/users/mig_user3/Automathon_Sujet/best_model.pt", map_location=DEVICE))
    classifier.to(DEVICE)


    # Prepare results dict
    results = {"audios": {}}
    audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".ogg") and not f.startswith(".")]

    for i, audio_file in enumerate(audio_files, start=1):
        file_path = os.path.join(INPUT_DIR, audio_file)
        print(f"Processing {audio_file} ({i}/{len(audio_files)})...")
        timestamps = predict_long_audio_with_panns(classifier, file_path)

        merged_timestamps = merge_timestamps(timestamps, max_gap=2.0)
        results["audios"][str(i)] = {
            "id": str(i),
            "timestamps": merged_timestamps
        }

        # Save after each audio
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Results saved to {OUTPUT_JSON}")

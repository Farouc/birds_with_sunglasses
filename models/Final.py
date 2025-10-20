import os
import json
import torch
import librosa
import re
import numpy as np
from model_classif import CNNBiGRU
from panns_inference import AudioTagging, labels
from torch.utils.data import DataLoader, TensorDataset

# -------------------------- CONFIG --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 22050
PANN_SR = 32000
CHUNK_DURATION = 2.0
OVERLAP = 0.5
THRESHOLD = 0.35
INPUT_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation"
OUTPUT_JSON = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation_results.json"
BATCH_SIZE_CNN = 32
BATCH_SIZE_PANN = 16
BIRD_THRESHOLD = 0.02

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

# -------------------------- PANNs --------------------------
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

def batch_is_bird_present(chunks):
    """Run PANNs on chunks of a single audio and return a boolean mask."""
    # Resample if needed
    batch_resampled = [librosa.resample(y=c, orig_sr=SR, target_sr=PANN_SR) if SR != PANN_SR else c for c in chunks]
    batch_resampled = np.array(batch_resampled)  # convert list of arrays to 2D or 1D array
    batch_tensor = torch.tensor(batch_resampled, dtype=torch.float32).to(DEVICE)

    mask = []
    for i in range(0, len(batch_tensor), BATCH_SIZE_PANN):
        x = batch_tensor[i:i+BATCH_SIZE_PANN]
        with torch.no_grad():
            scores, _ = pann_model.inference(x)
        bird_scores = scores[:, bird_indices].sum(axis=1)
        mask.extend(bird_scores > BIRD_THRESHOLD)
    return mask

# -------------------------- CNN --------------------------
def predict_chunks_cnn(model, chunks):
    specs = [audio_to_melspec(c) for c in chunks]
    specs = np.stack(specs)
    # Normalize
    specs = (specs - specs.mean(axis=(1,2,3), keepdims=True)) / (specs.std(axis=(1,2,3), keepdims=True)+1e-6)
    specs_tensor = torch.tensor(specs, dtype=torch.float32).to(DEVICE)
    dataset = TensorDataset(specs_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_CNN)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            batch_probs = model(x).cpu().numpy()
            batch_probs = np.atleast_1d(batch_probs) 
            probs.extend(batch_probs)
    return np.array(probs)

# -------------------------- MERGE TIMESTAMPS --------------------------
def merge_timestamps(timestamps, max_gap=2.0):
    if not timestamps:
        return []
    timestamps = sorted([list(t) for t in timestamps], key=lambda x: x[0])
    merged = [timestamps[0]]
    for start, end in timestamps[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged

# -------------------------- NATURAL SORT --------------------------
def natural_sort_key(filename):
    parts = re.split(r'(\d+)', filename)
    return [int(p) if p.isdigit() else p for p in parts]

# -------------------------- PREDICT SINGLE AUDIO --------------------------
def predict_long_audio(model, audio_path):
    y, _ = librosa.load(audio_path, sr=SR)
    chunks_data = segment_audio(y, SR)
    
    chunks = [c for _, _, c in chunks_data]
    timestamps_all = [[start, end] for start, end, _ in chunks_data]
    
    # PANNs mask (parallelized over batches)
    bird_mask = batch_is_bird_present(chunks)
    
    # Select only chunks with birds
    bird_chunks = [chunks[i] for i, present in enumerate(bird_mask) if present]
    bird_timestamps = [timestamps_all[i] for i, present in enumerate(bird_mask) if present]
    
    # CNN classifier
    if bird_chunks:
        probs = predict_chunks_cnn(model, bird_chunks)
        selected = [bird_timestamps[i] for i, p in enumerate(probs) if p >= THRESHOLD]
    else:
        selected = []
    
    # Merge timestamps
    merged = merge_timestamps(selected)
    return merged

# -------------------------- MAIN --------------------------
if __name__ == "__main__":
    classifier = CNNBiGRU()
    classifier.load_state_dict(torch.load(
        "/automathon/users/mig_user3/Automathon_Sujet/best_model.pt",
        map_location=DEVICE
    ))
    classifier.to(DEVICE)

    results = {"audios": {}}
    audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".ogg") and not f.startswith(".")]
    audio_files = sorted(audio_files, key=natural_sort_key)

    for i, audio_file in enumerate(audio_files, start=1):
        file_path = os.path.join(INPUT_DIR, audio_file)
        print(f"Processing {audio_file} ({i}/{len(audio_files)})...")
        merged_timestamps = predict_long_audio(classifier, file_path)

        results[str(i)] = {
            "id": str(i),
            "file": audio_file,
            "timestamps": merged_timestamps
        }

    # Save JSON **once after all audios**
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {OUTPUT_JSON}")


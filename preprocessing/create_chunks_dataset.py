import os
import torch
import librosa
import numpy as np
from panns_inference import AudioTagging, labels
from pathlib import Path
import soundfile as sf

# =======================
# Paths
# =======================
base_path = "/automathon/users/mig_user3/Automathon_Sujet/dataset/cleaned_bird_songs"
output_path = "/automathon/users/mig_user3/Automathon_Sujet/dataset/chunks_dataset"
Path(output_path).mkdir(exist_ok=True)

# Dataset folders
datasets = ["guepier_sounds", "guepier_noise", "other_sounds", "other_noise"]
for d in datasets:
    Path(os.path.join(output_path, d)).mkdir(exist_ok=True)

# =======================
# PANNs setup
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = AudioTagging(device=device)

# Bird detection classes
bird_classes = [
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Crow",
    "Caw",
    "Owl"
]
bird_indices = [labels.index(c) for c in bird_classes if c in labels]
threshold = 0.02  # score threshold for bird presence

# =======================
# Chunk parameters
# =======================
chunk_sec = 2.0
batch_size = 16  # number of chunks per batch

# =======================
# Iterate over species
# =======================
for species in os.listdir(base_path):
    species_path = os.path.join(base_path, species)
    if not os.path.isdir(species_path):
        continue

    print(f"Processing species: {species}")

    for audio_file in os.listdir(species_path):
        if not audio_file.endswith(".wav"):
            continue

        file_path = os.path.join(species_path, audio_file)
        y, sr = librosa.load(file_path, sr=32000, mono=True)
        chunk_samples = int(chunk_sec * sr)
        num_chunks = len(y) // chunk_samples

        # Prepare chunks
        chunks = []
        chunk_infos = []
        for i in range(num_chunks):
            chunk = y[i*chunk_samples : (i+1)*chunk_samples]
            if len(chunk) < chunk_samples:
                continue
            chunks.append(chunk)
            chunk_infos.append(i)

        if not chunks:
            continue

        # Convert to tensor and move to GPU
        chunks_array = np.array(chunks, dtype=np.float32)  # much faster
        chunks_tensor = torch.from_numpy(chunks_array).to(device)

        # Inference in batches
        for start in range(0, len(chunks_tensor), batch_size):
            end = start + batch_size
            batch = chunks_tensor[start:end]

            scores_batch, _ = model.inference(batch)  # already numpy

            for j, scores in enumerate(scores_batch):
                scores = scores.flatten()
                bird_score = sum([scores[idx] for idx in bird_indices])

                # Decide dataset
                chunk_idx = chunk_infos[start + j]
                if species == "#eubeat1":
                    dataset_name = "guepier_sounds" if bird_score > threshold else "guepier_noise"
                else:
                    dataset_name = "other_sounds" if bird_score > threshold else "other_noise"

                # Save chunk as WAV
                chunk_filename = f"{species}_{audio_file.replace('.wav','')}_chunk{chunk_idx+1}.wav"
                chunk_path = os.path.join(output_path, dataset_name, chunk_filename)
                sf.write(chunk_path, chunks[chunk_idx], sr)

print("All datasets created successfully!")

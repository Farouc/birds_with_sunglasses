import os
import numpy as np
import librosa
from tqdm import tqdm

# ================= CONFIG =================
SR = 22050
DURATION = 2.0
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

def audio_to_melspectrogram(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       hop_length=hop_length, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    delta = librosa.feature.delta(S_db)
    delta2 = librosa.feature.delta(S_db, order=2)
    return np.stack([S_db, delta, delta2], axis=0)  # (3, n_mels, time)

def preprocess_and_save(audio_paths, labels, output_dir="spectrogram_cache"):
    os.makedirs(output_dir, exist_ok=True)
    spec_paths = []
    label_list = []

    print(f"[INFO] Generating and saving spectrograms to {output_dir}/ ...")

    for i, (path, label) in enumerate(tqdm(list(zip(audio_paths, labels)), desc="Processing")):
        try:
            y, _ = librosa.load(path, sr=SR, duration=DURATION)
            if len(y) < int(SR * DURATION):
                y = np.pad(y, (0, int(SR * DURATION) - len(y)))
            spec = audio_to_melspectrogram(y)
            save_path = os.path.join(output_dir, f"spec_{i:05d}.npy")
            np.save(save_path, spec)
            spec_paths.append(save_path)
            label_list.append(label)
        except Exception as e:
            print(f"[WARN] Skipped {path}: {e}")
            continue

    np.save(os.path.join(output_dir, "labels.npy"), np.array(label_list))
    print(f"[INFO] Saved {len(spec_paths)} spectrograms and labels.\n")
    return spec_paths, label_list


if __name__ == "__main__":
    guepier_folder = "/automathon/users/mig_user3/Automathon_Sujet/dataset/chunks_dataset/guepier_sounds"
    other_folder = "/automathon/users/mig_user3/Automathon_Sujet/dataset/chunks_dataset/other_sounds"

    guepier_files = [os.path.join(guepier_folder, f) for f in os.listdir(guepier_folder) if f.endswith(".wav")]
    other_files = [os.path.join(other_folder, f) for f in os.listdir(other_folder) if f.endswith(".wav")]

    file_paths = guepier_files + other_files
    labels = [1]*len(guepier_files) + [0]*len(other_files)

    preprocess_and_save(file_paths, labels, output_dir="spectrogram_cache")

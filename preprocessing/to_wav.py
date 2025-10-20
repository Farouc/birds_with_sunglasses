import os
import librosa
import soundfile as sf
from tqdm import tqdm

# ------------------------------
# Settings
# ------------------------------
input_dir = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation"
output_dir = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation_wav"
target_sr = 22050  # Hz

os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Get all .ogg files in the folder
# ------------------------------
audio_files = [f for f in os.listdir(input_dir) if f.endswith(".ogg") and not f.startswith(".")]

# ------------------------------
# Process each audio file
# ------------------------------
for filename in tqdm(audio_files, desc="Processing .ogg files"):
    print(filename)
    file_path = os.path.join(input_dir, filename)
    out_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")

    # Load ogg with librosa (resample)
    y, sr = librosa.load(file_path, sr=target_sr)

    # Save as wav
    sf.write(out_file, y, target_sr)

print("All .ogg files processed successfully.")

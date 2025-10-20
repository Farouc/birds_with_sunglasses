import json
import numpy as np

# Load your dataset
with open('/automathon/users/mig_user3/Automathon_Sujet/dataset/parc_audios/timestamps.json', 'r') as f:
    data = json.load(f)

# Store all positive gaps along with audio_id and their indices
all_gaps = []

for audio_id, audio_data in data['audios'].items():
    timestamps = audio_data['timestamps']
    if len(timestamps) < 2:
        continue

    timestamps = np.array(timestamps)
    timestamps = timestamps[np.argsort(timestamps[:,0])]

    gaps = timestamps[1:,0] - timestamps[:-1,1]  # start[t+1] - end[t]
    positive_indices = np.where(gaps > 0)[0]

    for idx in positive_indices:
        all_gaps.append({
            "audio_id": audio_id,
            "gap": gaps[idx],
            "index": idx,
            "timestamps": timestamps
        })

# Sort all gaps by value
all_gaps_sorted = sorted(all_gaps, key=lambda x: x["gap"])

# Print the 5 smallest gaps along with previous 2 intervals
print("5 smallest gaps and previous intervals:")
for i in range(min(5, len(all_gaps_sorted))):
    g = all_gaps_sorted[i]
    idx = g["index"]
    ts = g["timestamps"]
    print(f"\nAudio {g['audio_id']}, gap {g['gap']:.3f} seconds between intervals:")
    print(f"  Interval before: start={ts[idx][0]:.3f}, end={ts[idx][1]:.3f}")
    print(f"  Interval after:  start={ts[idx+1][0]:.3f}, end={ts[idx+1][1]:.3f}")
    
    # Print up to 3 previous intervals if they exist
    for j in range(1,4):
        if idx-j >= 0:
            print(f"  {j} interval(s) before: start={ts[idx-j][0]:.3f}, end={ts[idx-j][1]:.3f}")

import json

# === 1️⃣ Chemin du fichier d'entrée et de sortie ===
input_path = "pred_timestamps_ast.json"        # ton fichier actuel (avec "audio_1", etc.)
output_path = "pred_timestamps_ast_not_validation_converted.json"  # fichier reformatté

# === 2️⃣ Chargement du fichier d'origine ===
with open(input_path, "r") as f:
    data = json.load(f)

audios = data.get("audios", {})
new_audios = {}

# === 3️⃣ Conversion du format des clés et IDs ===
for old_key, content in audios.items():
    # Extraire le numéro (ex: "audio_12" → "12")
    num = ''.join(c for c in old_key if c.isdigit())
    if not num:
        continue  # ignorer si pas de numéro identifiable

    new_audios[num] = {
        "id": num,
        "timestamps": content.get("timestamps", [])
    }

# === 4️⃣ Recalcul de next_id ===
if new_audios:
    numeric_ids = [int(k) for k in new_audios.keys()]
    next_id = max(numeric_ids) + 1
else:
    next_id = 1

# === 5️⃣ Construction du nouveau JSON ===
new_data = {
    "audios": new_audios,
    "next_id": next_id
}

# === 6️⃣ Sauvegarde propre ===
with open(output_path, "w") as f:
    json.dump(new_data, f, indent=4)

print(f"[INFO] ✅ JSON reformatté sauvegardé dans '{output_path}'")
print(f"[INFO] {len(new_audios)} entrées converties. next_id = {next_id}")

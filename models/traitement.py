import os, json, torch, librosa, numpy as np
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from panns_inference import AudioTagging, labels
import joblib

# =====================
# CONFIG
# =====================
# AUDIO_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset/parc_audios/wav_data"
AUDIO_DIR = "/automathon/users/mig_user3/Automathon_Sujet/dataset_validation/"
CHECKPOINT_AST = "../models/ast_bird_best.pt"
RF_PATH = "../models/random_forest_bird.pkl"
SCALER_PATH = "../models/scaler_bird.pkl"
OUT_JSON = "pred_timestamps.json"

WINDOW_SEC = 2.0
HOP_SEC = 1.0  # 50% overlap
SR_PANNS = 32000
SR_AST = 16000
BATCH_SIZE = 8

# PANNS params
BIRD_CLASSES = [
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Crow",
    "Caw",
    "Owl"
]
THR_BIRD = 0.02   # seuil PANNS
THR_RF = 0.6      # seuil Random Forest

# =====================
# MODELS LOADING
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# 🐦 PANNS (CPU plus sûr)
panns = AudioTagging(device="cuda")
bird_indices = [labels.index(c) for c in BIRD_CLASSES if c in labels]

# Random Forest & Scaler
rf = joblib.load(RF_PATH)
scaler = joblib.load(SCALER_PATH)

# AST
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
checkpoint = torch.load(CHECKPOINT_AST, map_location=device)

# Vérifiez que len(rf.classes_) correspond au nombre de labels de votre modèle
num_labels_rf = len(rf.classes_)
print(f"[INFO] Chargement du modèle AST pour {num_labels_rf} classes (détectées depuis le RF).")

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=num_labels_rf,
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("[INFO] ✅ All models loaded successfully.")


# =====================
# [CORRECTION] VÉRIFICATION CLASSE CIBLE (RF)
# =====================
# L'utilisateur a confirmé que 'guepier_sounds' correspond à l'index 0
target_class_index = 0  

try:
    if target_class_index not in rf.classes_:
        raise IndexError(f"L'index cible {target_class_index} n'a pas été trouvé dans les classes du RF {rf.classes_}")
    
    # Stocker l'index cible pour l'utiliser dans la boucle
    rf_target_idx = target_class_index
    print(f"[INFO] Index cible vérifié. Utilisation de l'index {rf_target_idx} (pour 'guepier_sounds') sur les {len(rf.classes_)} classes du RF.")

except (IndexError, TypeError) as e:
    print(f"[ERREUR FATALE] Problème avec l'index cible. {e}")
    print("Veuillez vérifier l'index 'target_class_index' dans le script. Sortie.")
    exit(1) # Quitter le script


# =====================
# FUNCTIONS
# =====================
def chunk_audio(y, sr, win_sec=2.0, hop_sec=1.0):
    """Découpe un long audio en chunks avec overlap."""
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    
    # Assurer que l'audio est assez long pour au moins une fenêtre
    if len(y) < win:
        return np.array([]), np.array([])
        
    starts = np.arange(0, len(y) - win + 1, hop)
    chunks = np.stack([y[s:s + win] for s in starts])
    times = starts / sr
    return chunks, times


def merge_intervals(times, mask, win_sec=2.0, hop_sec=1.0):
    """Fusionne les fenêtres positives adjacentes."""
    intervals, start = [], None
    if not len(mask):
        return []
        
    for i, val in enumerate(mask):
        if val and start is None:
            start = times[i]
        elif not val and start is not None:
            # La fenêtre précédente était la dernière positive
            end = times[i - 1] + win_sec
            intervals.append([float(start), float(end)])
            start = None
            
    # Gérer le cas où le fichier se termine sur une détection
    if start is not None:
        intervals.append([float(start), float(times[-1] + win_sec)])
    return intervals

# =====================
# MAIN LOOP
# =====================
results = {"audios": {}, "next_id": None}
next_id = 1
# audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".ogg")])

for fname in tqdm(audio_files, desc="Processing long audios"):
    audio_id = os.path.splitext(fname)[0]
    file_path = os.path.join(AUDIO_DIR, fname)

    try:
        y, _ = librosa.load(file_path, sr=SR_PANNS, mono=True)
    except Exception as e:
        print(f"[ERROR] Impossible de lire {fname}. Erreur: {e}")
        continue

    chunks, times = chunk_audio(y, SR_PANNS, WINDOW_SEC, HOP_SEC)
    
    if chunks.size == 0:
        print(f"[WARN] Audio {fname} est plus court que la fenêtre (2s) et sera ignoré.")
        results["audios"][str(audio_id)] = {"id": str(audio_id), "timestamps": []}
        continue

    positive_windows = np.zeros(len(chunks), dtype=bool)

    # ---- 1️⃣ PANNS pass (bird presence) ----
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        
        # [MODIF] Tu avais .cpu() ici, mais PANNs est sur 'cuda'. 
        # On doit envoyer le tenseur sur le bon device.
        batch_tensor = torch.from_numpy(batch_chunks).float().to(device)
        
        try:
            # [MODIF] PANNs sur GPU renvoie des tenseurs
            scores_batch, _ = panns.inference(batch_tensor)
            
            # Calcul en batch (plus rapide)
            bird_scores = scores_batch[:, bird_indices].sum(dim=1)
            positive_mask_tensor = (bird_scores > THR_BIRD)
            
            # Appliquer le masque au array numpy
            positive_mask_numpy = positive_mask_tensor.cpu().numpy()
            positive_windows[i : i + len(positive_mask_numpy)][positive_mask_numpy] = True

        except Exception as e:
            print(f"[ERROR] Échec de PANNS sur un batch de {fname}. Erreur: {e}")
            continue # Passe au batch suivant

    # ---- 2️⃣ AST + RF fine classification ----
    if positive_windows.any():
        # Indices de *tous* les chunks qui ont passé PANNS
        idxs_panns_positive = np.where(positive_windows)[0]
        
        emb_chunks = []
        
        # Nous devons suivre les indices qui passent *vraiment*
        # le filtre de longueur, car certains peuvent être < 400 échantillons.
        valid_indices_processed = []

        for i in range(0, len(idxs_panns_positive), BATCH_SIZE):
            batch_original_indices = idxs_panns_positive[i:i + BATCH_SIZE]
            
            segs_list_16k = []
            valid_indices_in_batch = []

            for original_idx in batch_original_indices:
                seg_32k = chunks[original_idx]
                
                # Rééchantillonnage pour AST
                seg_16k = librosa.resample(seg_32k, orig_sr=SR_PANNS, target_sr=SR_AST)
                
                # Le check de sécurité (AST fbank a besoin d'au moins 400 échantillons)
                if len(seg_16k) >= 400:
                    segs_list_16k.append(seg_16k)
                    valid_indices_in_batch.append(original_idx) # On garde l'indice
                else:
                    # Ce segment est trop court, on l'ignore.
                    pass 

            if not segs_list_16k: # Aucun segment valide dans ce batch
                continue

            # Passer la LISTE des segments au feature_extractor.
            inputs = feature_extractor(
                segs_list_16k, 
                sampling_rate=SR_AST, 
                return_tensors="pt", 
                padding=True
            )

            # Transférer le batch sur GPU pour le modèle
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Récupérer les embeddings et les remettre sur CPU pour numpy/sklearn
                emb = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()

            emb_chunks.append(emb)
            # Ajouter les indices valides de ce batch à la liste globale
            valid_indices_processed.extend(valid_indices_in_batch)
            torch.cuda.empty_cache()

        if emb_chunks:
            X = np.vstack(emb_chunks)
            X_scaled = scaler.transform(X)
            probs = rf.predict_proba(X_scaled)

            # [CORRECTION] Utilise la variable rf_target_idx (qui vaut 0) 
            # vérifiée au démarrage du script.
            pred_mask_rf = (probs[:, rf_target_idx] > THR_RF)

            # Mettre à jour 'positive_windows'
            
            # D'abord, on met à False *tous* les indices qui avaient passé PANNS.
            positive_windows[idxs_panns_positive] = False
            
            # On s'assure que les longueurs correspondent
            if len(valid_indices_processed) == len(pred_mask_rf):
                # On sélectionne les indices valides qui ont aussi passé le RF
                final_positive_indices = np.array(valid_indices_processed)[pred_mask_rf]
                # On met True pour ces indices finaux
                positive_windows[final_positive_indices] = True
            else:
                 print(f"[ERROR] Désynchronisation des indices dans {fname}. Attendu {len(valid_indices_processed)} prédictions, obtenu {len(pred_mask_rf)}")


    # ---- 3️⃣ Fusion temporelle ----
    intervals = merge_intervals(times, positive_windows, WINDOW_SEC, HOP_SEC)
    results["audios"][str(audio_id)] = {"id": str(audio_id), "timestamps": intervals}
    
    # Gestion de next_id
    try:
        numeric_id = int(audio_id)
        next_id = max(next_id, numeric_id + 1)
    except ValueError:
        # Si l'ID n'est pas un nombre, on ignore pour next_id
        pass

    # ==========================================================
    # [MODIFICATION D'URGENCE] - SAUVER LE JSON À CHAQUE FOIS
    # ==========================================================
    results["next_id"] = next_id  # Mettre à jour le next_id dans le dict
    try:
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"[WARN] Échec de la sauvegarde JSON intermédiaire: {e}")
    # ==========================================================

# --- FIN DE LA BOUCLE FOR ---

# [MODIFICATION] Le bloc de sauvegarde final a été déplacé dans la boucle.
# On imprime juste un message final.
print(f"[INFO] ✅ Traitement terminé. Résultats finaux dans {OUT_JSON}")
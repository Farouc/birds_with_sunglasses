import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def merge_overlapping_intervals(intervals):
    """
    Fusionne une liste d'intervalles qui se chevauchent.
    Ex: [[1, 5], [3, 7], [9, 10]] -> [[1, 7], [9, 10]]
    """
    if not intervals:
        return []
    
    # Trier les intervalles par leur début
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    current_start, current_end = intervals[0]
    
    for i in range(1, len(intervals)):
        next_start, next_end = intervals[i]
        
        if next_start <= current_end:
            # Il y a chevauchement, étendre l'intervalle courant
            current_end = max(current_end, next_end)
        else:
            # Pas de chevauchement, ajouter l'intervalle précédent et en commencer un nouveau
            merged.append([current_start, current_end])
            current_start, current_end = next_start, next_end
            
    # Ajouter le dernier intervalle
    merged.append([current_start, current_end])
    return merged

def calculate_iou(interval_a, interval_b):
    """
    Calcule l'Intersection sur l'Union (IoU) de deux intervalles [start, end].
    """
    start_a, end_a = interval_a
    start_b, end_b = interval_b
    
    # Calcul de l'intersection
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calcul de l'union
    len_a = end_a - start_a
    len_b = end_b - start_b
    union = len_a + len_b - intersection
    
    if union == 0:
        # Évite la division par zéro si les deux intervalles sont de longueur 0
        return 0.0
        
    return intersection / union

def evaluate_timestamps(pred_json_path, ref_json_path, iou_threshold=0.5, beta=4):
    """
    Calcule les métriques de détection (Précision, Rappel, F-beta) 
    en comparant deux fichiers JSON de timestamps.
    """
    try:
        with open(pred_json_path, 'r') as f:
            pred_data = json.load(f)
        with open(ref_json_path, 'r') as f:
            ref_data = json.load(f)
    except Exception as e:
        print(f"Erreur lors de la lecture des fichiers JSON: {e}")
        return

    global_tp = 0  # Vrais Positifs
    global_fp = 0  # Faux Positifs
    global_fn = 0  # Faux Négatifs
    
    total_predictions = 0
    total_truths = 0
    
    # Itérer sur tous les audios du fichier de RÉFÉRENCE
    for audio_id, ref_entry in ref_data["audios"].items():
        
        # 1. Obtenir les listes d'intervalles de référence et de prédiction
        ref_intervals_raw = ref_entry.get("timestamps", [])
        
        # Récupérer les prédictions pour cet audio_id, ou une liste vide si non trouvé
        pred_entry = pred_data.get("audios", {}).get(audio_id, {})
        pred_intervals_raw = pred_entry.get("timestamps", [])
        
        # 2. Fusionner les intervalles qui se chevauchent (crucial)
        ref_intervals = merge_overlapping_intervals(ref_intervals_raw)
        pred_intervals = merge_overlapping_intervals(pred_intervals_raw)
        
        num_truths = len(ref_intervals)
        num_preds = len(pred_intervals)
        
        total_truths += num_truths
        total_predictions += num_preds
        
        # Si pas de prédictions ou pas de vérité, les TP sont 0
        if num_preds == 0 or num_truths == 0:
            file_tp = 0
        else:
            # 3. Construire la matrice IoU
            iou_matrix = np.zeros((num_preds, num_truths))
            for i in range(num_preds):
                for j in range(num_truths):
                    iou_matrix[i, j] = calculate_iou(pred_intervals[i], ref_intervals[j])
            
            # 4. Trouver le matching optimal (coût = -IoU car on veut maximiser)
            # L'algorithme Hongrois (linear_sum_assignment) trouve le matching
            # 1-pour-1 qui maximise l'IoU totale.
            pred_indices, ref_indices = linear_sum_assignment(-iou_matrix)
            
            file_tp = 0
            for pred_idx, ref_idx in zip(pred_indices, ref_indices):
                # Si le IoU du match est au-dessus du seuil, c'est un Vrai Positif
                if iou_matrix[pred_idx, ref_idx] >= iou_threshold:
                    file_tp += 1
        
        # 5. Mettre à jour les compteurs globaux
        file_fp = num_preds - file_tp  # Faux Positifs (prédictions non matchées)
        file_fn = num_truths - file_tp # Faux Négatifs (vérités non matchées)
        
        global_tp += file_tp
        global_fp += file_fp
        global_fn += file_fn

    # 6. Calculer les métriques finales
    print("--- Statistiques Globales ---")
    print(f"Total Vérités (après fusion) : {total_truths}")
    print(f"Total Prédictions (après fusion): {total_predictions}")
    print(f"Vrais Positifs (TP)   (IoU > {iou_threshold}): {global_tp}")
    print(f"Faux Positifs (FP) : {global_fp}")
    print(f"Faux Négatifs (FN) : {global_fn}")
    print("-----------------------------")

    # Précision = TP / (TP + FP) = TP / Total Prédictions
    if (global_tp + global_fp) == 0:
        precision = 0.0
        print("Précision: 0.0 (aucune prédiction)")
    else:
        precision = global_tp / (global_tp + global_fp)
        print(f"Précision: {precision:.4f}")

    # Rappel = TP / (TP + FN) = TP / Total Vérités
    if (global_tp + global_fn) == 0:
        recall = 0.0
        print("Rappel (Recall): 0.0 (aucune vérité)")
    else:
        recall = global_tp / (global_tp + global_fn)
        print(f"Rappel (Recall): {recall:.4f}")

    # F-beta score
    beta_sq = beta ** 2
    f_beta_denominator = (beta_sq * precision) + recall
    
    if f_beta_denominator == 0:
        f_beta_score = 0.0
        print(f"F{beta}-Score: 0.0")
    else:
        f_beta_score = (1 + beta_sq) * (precision * recall) / f_beta_denominator
        print(f"F{beta}-Score: {f_beta_score:.4f}")
        
    return {
        "precision": precision,
        "recall": recall,
        f"f{beta}_score": f_beta_score,
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
        "iou_threshold": iou_threshold
    }

# =====================
# EXÉCUTION
# =====================
if __name__ == "__main__":
    
    # Mettez les noms de vos fichiers ici
    PRED_FILE = "pred_timestamps_ast_not_validation_converted.json"
    REF_FILE = "dataset/parc_audios/timestamps.json" # Le fichier de référence que vous avez montré
    
    print(f"Évaluation de '{PRED_FILE}' contre '{REF_FILE}'...\n")
    
    # Vous pouvez changer le seuil IoU (0.5 est standard) 
    # et le bêta (4 pour votre F4-score)
    metrics = evaluate_timestamps(
        pred_json_path=PRED_FILE, 
        ref_json_path=REF_FILE, 
        iou_threshold=0.5,
        beta=4
    )
    
    print(f"\nScore F4 final: {metrics['f4_score']:.4f}")
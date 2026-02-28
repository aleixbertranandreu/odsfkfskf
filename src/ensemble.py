"""
ensemble.py — Final unified inference: FashionCLIP (Estética) + ConvNeXt (Textura) + YOLO (Cajas) + Color
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from reranker import category_mask_score, YOLO_TO_CATEGORIES

def extract_crop_histogram(img_bgr, box, h_bins=12, s_bins=4, v_bins=4):
    """Extrae el de HSV de una caja YOLO específica en la imagen."""
    x1, y1, x2, y2 = map(int, box)
    h, w = img_bgr.shape[:2]
    
    # Asegurar márgenes
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(3 * h_bins * s_bins * v_bins, dtype=np.float32)
        
    img_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_crop = img_hsv.shape[0]
    blocks = [
        img_hsv[0 : h_crop // 3, :],
        img_hsv[h_crop // 3 : 2 * h_crop // 3, :],
        img_hsv[2 * h_crop // 3 :, :]
    ]

    hists = []
    for block in blocks:
        hist = cv2.calcHist([block], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist.flatten())

    return np.concatenate(hists)


def color_similarity(hist1, hist2):
    """Intersección de histogramas: 1.0 (idéntico) a 0.0 (diferente)"""
    if hist1 is None or hist2 is None:
        return 0.5
    if np.sum(hist1) == 0 or np.sum(hist2) == 0:
        return 0.5
    # Bhattacharyya distance works better but Intersection is faster and monotonic
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)


def load_csv_ranks(csv_path):
    """Convierte el CSV de Kaggle en un diccionario {bundle_id: {product_id: rank}}"""
    df = pd.read_csv(csv_path)
    ranks = defaultdict(dict)
    
    # pandas groupby preserves order
    grouped = df.groupby("bundle_asset_id")
    for bundle_id, group in grouped:
        for rank, product_id in enumerate(group["product_asset_id"]):
            ranks[bundle_id][product_id] = rank + 1  # rank 1-indexed
    return ranks


def reciprocal_rank_fusion(ranks_clip, ranks_metric, k=60):
    """
    RRF: Fusión de dos sistemas de recomendación sin necesitar la distancia bruta (que a veces no está en misma escala).
    score = 1 / (k + rank1) + 1 / (k + rank2)
    """
    fused_scores = defaultdict(lambda: defaultdict(float))
    all_bundles = set(ranks_clip.keys()).union(set(ranks_metric.keys()))
    
    for b in all_bundles:
        clip_b = ranks_clip.get(b, {})
        metric_b = ranks_metric.get(b, {})
        
        all_products = set(clip_b.keys()).union(set(metric_b.keys()))
        for p in all_products:
            r_clip = clip_b.get(p, 1000)   # Penalti si no está en el top de uno
            r_metric = metric_b.get(p, 1000)
            
            score = (1.0 / (k + r_clip)) + (1.0 / (k + r_metric))
            fused_scores[b][p] = score
            
    return fused_scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_csv", type=str, required=True, help="CSV from FashionCLIP")
    parser.add_argument("--metric_csv", type=str, required=True, help="CSV from ConvNeXt")
    parser.add_argument("--yolo_json", type=str, default=os.path.join(BASE_DIR, "data", "yolo_test_bboxes.json"))
    parser.add_argument("--color_json", type=str, default=os.path.join(BASE_DIR, "data", "product_color_hists.json"))
    parser.add_argument("--bundles_dir", type=str, default=os.path.join(BASE_DIR, "data", "images", "bundles"))
    parser.add_argument("--output", type=str, default="final_unified_submission.csv")
    args = parser.parse_args()

    print("🚀 Iniciando EL GRAN ENSAMBLE...")
    
    print("1️⃣ Cargando predicciones de CLIP y ConvNeXt...")
    ranks_clip = load_csv_ranks(args.clip_csv)
    ranks_metric = load_csv_ranks(args.metric_csv)
    base_scores = reciprocal_rank_fusion(ranks_clip, ranks_metric, k=60)
    
    print("2️⃣ Cargando metadatos (Catálogo, YOLO, Colores)...")
    df_products = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv"))
    product_category = dict(zip(df_products['product_asset_id'].astype(str), df_products['product_description']))
    
    with open(args.yolo_json, "r") as f:
        yolo_boxes = json.load(f)
        
    with open(args.color_json, "r") as f:
        raw_colors = json.load(f)
        catalog_colors = {k: np.array(v, dtype=np.float32) for k, v in raw_colors.items()}

    print("3️⃣ Re-Ranking Final (Color + Categorías YOLO)...")
    final_output = []
    
    w_color = 0.2
    
    # Process bundle by bundle
    for bundle_id, candidates in tqdm(base_scores.items()):
        img_path = os.path.join(args.bundles_dir, f"{bundle_id}.jpg")
        img_bgr = cv2.imread(img_path) if os.path.exists(img_path) else None
        
        boxes = yolo_boxes.get(bundle_id, [])
        
        # Calculate color for each YOLO crop in the bundle
        bundle_crop_colors = []
        if img_bgr is not None:
            for b in boxes:
                hist = extract_crop_histogram(img_bgr, b["box"])
                bundle_crop_colors.append({"class": b["class"], "hist": hist})
        
        final_candidates = []
        for pid, score in candidates.items():
            cat = product_category.get(pid, "UNKNOWN")
            
            # Si no hay cajas detectadas, dejamos el score base
            if not boxes:
                final_candidates.append((score, pid))
                continue
                
            # Buscar la mejor coincidencia para este producto entre todas las cajas YOLO
            best_cat_mask = 0.05
            best_color_sim = 0.5
            
            for crop in bundle_crop_colors:
                # 1. ¿Encaja la categoría del producto en lo que dice esta caja YOLO?
                c_mask = category_mask_score(crop["class"], cat)
                
                # 2. ¿Se parece el color de esta caja al color del catálogo de este producto?
                p_hist = catalog_colors.get(pid, None)
                col_sim = color_similarity(crop["hist"], p_hist)
                
                # Maximizar la mejor caja para este producto
                if c_mask > best_cat_mask:
                    best_cat_mask = c_mask
                if col_sim > best_color_sim:
                    best_color_sim = col_sim

            # Fórmula final de Grandmaster:
            # Score_Base (RRF) * Penalización de Categoría * Empuje de Color
            reranked_score = score * best_cat_mask
            
            # El empuje de color suma hasta un +20% al score si es idéntico
            if best_cat_mask > 0.1:  # Ignorar color si la categoría está mal
                reranked_score = reranked_score * (1.0 + w_color * best_color_sim)
                
            final_candidates.append((reranked_score, pid))
            
        # Ordenar de mayor a menor score y quedarnos con el top 150
        final_candidates.sort(key=lambda x: x[0], reverse=True)
        top_150 = final_candidates[:150]
        
        for _, pid in top_150:
            final_output.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})

    print(f"\n✅ Guardando presentación final con {len(final_output)} predicciones...")
    df_final = pd.DataFrame(final_output)
    df_final.to_csv(args.output, index=False)
    print(f"🎉 TODO LISTO: {args.output}")


if __name__ == "__main__":
    main()

"""
smart_inference_v8.py — V8: Grandmaster Re-Ranking Pipeline (The Big Leap)
══════════════════════════════════════════════════════════════════════════
Uses precomputed YOLO bounding boxes (arriba, abajo, otros), encodes them with 
CLIP, retrieves 300 candidates per crop via FAISS, and performs a 2-stage 
RERANKING to destroy false positives:
  1. HARD CATEGORY MASK: If crop is "abajo", kill any candidate that is a "T-SHIRT".
  2. COLOR HISTOGRAM MATCHING: Use the precise precomputed 3-stripe HSV histograms 
     to boost candidates that share the EXACT same color distribution as the crop.
"""
import argparse
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


YOLO_TO_CATEGORIES_ES = {
    "arriba": {
        'T-SHIRT', 'SHIRT', 'SWEATER', 'JACKET', 'COAT', 'BLOUSE', 'TOP',
        'WAISTCOAT', 'CARDIGAN', 'POLO SHIRT', 'BODYSUIT', 'SWEATSHIRT',
        'OVERSHIRT', 'KNITTED WAISTCOAT', 'TOPS AND OTHERS', 'WIND-JACKET',
        'ANORAK', 'BLAZER', 'SLEEVELESS PAD. JACKET', 'VEST', 'HOODIE',
        'CROP TOP', 'TANK TOP', 'LONG SLEEVE T-SHIRT', 'RUGBY POLO SHIRT',
        'LEATHER JACKET', 'DENIM JACKET', 'BOMBER JACKET', 'PARKA', 'RAINCOAT', 'OVERSHIRT'
    },
    "abajo": {
        'TROUSERS', 'JEANS', 'SHORTS', 'BERMUDA', 'LEGGINGS', 'JOGGER',
        'CARGO TROUSERS', 'CHINO', 'SWEATPANTS', 'CARGO SHORTS',
        'SKIRT', 'MINI SKIRT', 'MIDI SKIRT', 'LONG SKIRT',
    },
    "cuerpo_entero": {
        'DRESS', 'JUMPSUIT', 'OVERALL', 'BIB OVERALL', 'ROMPER',
    },
    "otros": {
        'BAG', 'SHOES', 'BOOTS', 'SANDALS', 'BELT', 'HAT', 'CAP',
        'SCARF', 'SOCKS', 'GLOVES', 'SUNGLASSES', 'JEWELLERY',
        'NECKLACE', 'EARRINGS', 'RING', 'BRACELET', 'WATCH',
        'HAND BAG-RUCKSACK', 'MOCCASINS', 'SANDAL', 'WALLET', 
        'UMBRELLA', 'HAIR ACCESSORY', 'TIE', 'IMIT JEWELLER', 'SNEAKERS'
    }
}

def category_mask_score(yolo_class: str, product_category: str) -> float:
    if not yolo_class or yolo_class == "unknown":
        return 1.0  # unknown crop bypasses mask
    
    allowed_set = YOLO_TO_CATEGORIES_ES.get(yolo_class)
    if not allowed_set:
        return 1.0
        
    p_up = product_category.upper().strip()
    if p_up in allowed_set:
        return 1.0
        
    for ac in allowed_set:
        if ac in p_up or p_up in ac:
            return 1.0
            
    return 0.05  # DEATH PENALTY

# Must match src/precompute_colors.py bin counts!
def extract_block_histogram(image_rgb: np.ndarray, h_bins=12, s_bins=4, v_bins=4) -> np.ndarray:
    if image_rgb is None or image_rgb.size == 0:
        return np.zeros(3 * h_bins * s_bins * v_bins, dtype=np.float32)
        
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    img_h, img_w = hsv.shape[:2]
    
    # Avoid edges
    pad_y, pad_x = max(1, int(img_h * 0.1)), max(1, int(img_w * 0.1))
    hsv = hsv[pad_y:img_h-pad_y, pad_x:img_w-pad_x]
    if hsv.size == 0: return np.zeros(3 * h_bins * s_bins * v_bins, dtype=np.float32)

    img_h = hsv.shape[0]
    block_h = img_h // 3
    if block_h < 1: return np.zeros(3 * h_bins * s_bins * v_bins, dtype=np.float32)

    histograms = []
    blocks = [hsv[0:block_h, :], hsv[block_h:2*block_h, :], hsv[2*block_h:, :]]
    
    for block in blocks:
        hist = cv2.calcHist([block], [0, 1, 2], None, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
        bsum = hist.sum()
        if bsum > 0:
            hist = hist / bsum
        histograms.append(hist.flatten())

    return np.concatenate(histograms).astype(np.float32)

def compare_block_histograms(hist_query: np.ndarray, hist_candidate: np.ndarray) -> float:
    if len(hist_query) == 0 or len(hist_candidate) == 0: return 0.0
    bc = np.sum(np.sqrt(np.maximum(hist_query * hist_candidate, 0)))
    return float(np.clip(bc, 0.0, 1.0))

def extract_zara_code(url):
    try:
        m = re.search(r'([A-Z]?\d{8,11})', url.split('/')[-2])
        return m.group(1) if m else None
    except: return None

def build_metadata(base_dir):
    df_b = pd.read_csv(os.path.join(base_dir, "data", "raw", "bundles_dataset.csv"))
    df_p = pd.read_csv(os.path.join(base_dir, "data", "raw", "product_dataset.csv"))

    bundle_section = dict(zip(df_b['bundle_asset_id'], df_b['bundle_id_section'].astype(str)))
    bundle_urls = dict(zip(df_b['bundle_asset_id'], df_b['bundle_image_url']))
    product_desc = dict(zip(df_p['product_asset_id'], df_p['product_description']))
    product_url = dict(zip(df_p['product_asset_id'], df_p['product_image_url']))
    all_product_ids = df_p['product_asset_id'].tolist()

    bundle_codes = {bid: extract_zara_code(url) for bid, url in bundle_urls.items()}
    product_codes = {pid: extract_zara_code(url) for pid, url in product_url.items()}

    model_to_products = defaultdict(list)
    prefix4_to_products = defaultdict(list)
    for pid, code in product_codes.items():
        if code:
            clean = code.lstrip('TM')
            if len(clean) >= 5: model_to_products[clean[:5]].append(pid)
            if len(clean) >= 4: prefix4_to_products[clean[:4]].append(pid)

    return {
        "bundle_section": bundle_section,
        "product_desc": product_desc,
        "all_product_ids": all_product_ids,
        "bundle_codes": bundle_codes,
        "product_codes": product_codes,
        "model_to_products": model_to_products,
        "prefix4_to_products": prefix4_to_products,
    }

def encode_images_batch(images, model, processor, device):
    inputs = processor(images=list(images), return_tensors="pt", padding=True)
    pv = inputs['pixel_values'].to(device)
    with torch.no_grad():
        vout = model.vision_model(pixel_values=pv)
        embs = model.visual_projection(vout.pooler_output)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy().astype(np.float32)

def run_test(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | Grandmaster V8 Pipeline")

    meta = build_metadata(BASE)
    yolo_file = os.path.join(BASE, "data", "yolo_test_bboxes.json")
    with open(yolo_file) as f: yolo_boxes = json.load(f)
    print(f"📦 Loaded YOLO boxes for {len(yolo_boxes)} bundles")

    color_file = os.path.join(BASE, "data", "product_color_hists.json")
    with open(color_file) as f: raw_colors = json.load(f)
    product_colors = {pid: np.array(lst, dtype=np.float32) for pid, lst in raw_colors.items()}
    print(f"🎨 Loaded {len(product_colors)} product color histograms")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    if args.custom_weights:
        model.load_state_dict(torch.load(args.custom_weights, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()

    emb_dir = os.path.join(BASE, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    idx = faiss.read_index(os.path.join(emb_dir, f"clip_faiss_{suffix}.bin"))
    with open(os.path.join(emb_dir, f"clip_ids_{suffix}.pkl"), 'rb') as f: all_pids = pickle.load(f)

    test_ids = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_test.csv"))['bundle_asset_id'].tolist()
    bdir = os.path.join(BASE, "data/images/bundles")
    
    start_time = time.time()
    results = []

    for i, bundle_id in enumerate(test_ids):
        bpath = os.path.join(bdir, f"{bundle_id}.jpg")
        if not os.path.exists(bpath):
            results.append((bundle_id, all_pids[:15]))
            continue

        try:
            image = Image.open(bpath).convert("RGB")
            img_np = np.array(image)
            
            crops = [image]
            crop_classes = ["unknown"]  # Full image has unknown class
            
            boxes = yolo_boxes.get(bundle_id, [])
            for box in boxes:
                if box["confidence"] > 0.15:
                    x1, y1, x2, y2 = box["box"]
                    pad = 0.05
                    w, h = image.size
                    px, py = (x2-x1)*pad, (y2-y1)*pad
                    x1, y1 = max(0, x1-px), max(0, y1-py)
                    x2, y2 = min(w, x2+px), min(h, y2+py)
                    crops.append(image.crop((x1, y1, x2, y2)))
                    crop_classes.append(box["class"])

            crops = crops[:8]
            crop_classes = crop_classes[:8]

            crop_embs = encode_images_batch(crops, model, processor, device)
            
            candidate_pool = defaultdict(float) # pid -> max score
            
            # For each crop: retrieve FAISS candidates, then apply Grandmaster Re-Ranking
            for ci, (emb, yolo_cls) in enumerate(zip(crop_embs, crop_classes)):
                q = emb.reshape(1, -1).copy()
                faiss.normalize_L2(q)
                scores, faiss_idxs = idx.search(q, min(300, idx.ntotal))
                
                # Normalize FAISS scores to [0.01, 1]
                faiss_scores = scores[0]
                s_min, s_max = faiss_scores.min(), faiss_scores.max()
                rng = max(s_max - s_min, 1e-6)
                norm_faiss = np.clip((faiss_scores - s_min) / rng, 0.01, 1.0)
                
                # Precompute crop color histogram just once
                crop_hist = None
                if i == 0: # Full image color isn't as useful
                    pass 
                else: 
                    crop_hist = extract_block_histogram(np.array(crops[ci]))

                for rank, (faiss_s, j) in enumerate(zip(norm_faiss, faiss_idxs[0])):
                    if j < 0: continue
                    pid = all_pids[j]
                    cat = meta["product_desc"].get(pid, "UNKNOWN")
                    
                    # Stage 1: Hard Category Mask (Nuclear Option)
                    gate = category_mask_score(yolo_cls, cat)
                    if gate < 0.1:  # Filtered out
                        score = faiss_s * gate
                    else:
                        # Stage 2: Precise Color Matching
                        color_bonus = 0.0
                        if crop_hist is not None and pid in product_colors:
                            cand_hist = product_colors[pid]
                            color_bonus = compare_block_histograms(crop_hist, cand_hist)
                        
                        score = faiss_s * gate * (1.0 + 3.0 * color_bonus)  # Heavy color weight!
                        
                    candidate_pool[pid] = max(candidate_pool[pid], float(score))

            # URL Boost Layer
            b_code = meta["bundle_codes"].get(bundle_id)
            if b_code:
                b_clean = b_code.lstrip('TM')
                if len(b_clean) >= 4:
                    for pid in meta["prefix4_to_products"].get(b_clean[:4], []):
                        candidate_pool[pid] += 2.0
                if len(b_clean) >= 5:
                    for pid in meta["model_to_products"].get(b_clean[:5], []):
                        candidate_pool[pid] += 5.0

            scored = sorted(candidate_pool.items(), key=lambda x: x[1], reverse=True)
            top_15 = [p for p, _ in scored][:15]

            # Pad if needed
            if len(top_15) < 15:
                seen = set(top_15)
                for pid in all_pids:
                    if pid not in seen:
                        top_15.append(pid)
                        seen.add(pid)
                    if len(top_15) >= 15: break

            results.append((bundle_id, top_15))

        except Exception as e:
            print(f"Error {bundle_id}: {e}")
            results.append((bundle_id, all_pids[:15]))

        if (i+1) % 50 == 0:
            print(f" [{i+1}/{len(test_ids)}] {(i+1)/(time.time()-start_time):.1f} bundles/s")

    outfile = os.path.join(BASE, "GRANDMASTER_V8_submission.csv")
    pd.DataFrame([{"bundle_asset_id": b, "product_asset_id": p} for b, t in results for p in t]).to_csv(outfile, index=False)
    print(f"💾 Saved {outfile}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    p.add_argument("--custom_weights", default=None)
    run_test(p.parse_args())

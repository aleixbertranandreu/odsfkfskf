"""
smart_inference_yolo.py — V7: YOLO-Guided Multi-Crop FAISS + RRF Fusion
═════════════════════════════════════════════════════════════════════════

Uses precomputed YOLOv8 bounding boxes (data/yolo_test_bboxes.json) to 
extract exact clothing items (arriba, abajo, otros) from the bundle image.
Each precise crop is encoded by FashionCLIP, and results are fused via RRF.
"""
import argparse
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def extract_zara_code(url):
    try:
        m = re.search(r'([A-Z]?\d{8,11})', url.split('/')[-2])
        return m.group(1) if m else None
    except:
        return None


def build_metadata(base_dir):
    df_b = pd.read_csv(os.path.join(base_dir, "data", "raw", "bundles_dataset.csv"))
    df_p = pd.read_csv(os.path.join(base_dir, "data", "raw", "product_dataset.csv"))
    df_t = pd.read_csv(os.path.join(base_dir, "data", "raw", "bundles_product_match_train.csv"))

    bundle_section = dict(zip(df_b['bundle_asset_id'], df_b['bundle_id_section'].astype(str)))
    bundle_urls = dict(zip(df_b['bundle_asset_id'], df_b['bundle_image_url']))
    product_desc = dict(zip(df_p['product_asset_id'], df_p['product_description']))
    product_url = dict(zip(df_p['product_asset_id'], df_p['product_image_url']))
    all_product_ids = df_p['product_asset_id'].tolist()

    sec_cat = defaultdict(Counter)
    for _, r in df_t.iterrows():
        bid, pid = r['bundle_asset_id'], r['product_asset_id']
        if bid in bundle_section and pid in product_desc:
            sec_cat[bundle_section[bid]][product_desc[pid]] += 1
            
    section_cat_prob = {}
    for sec, counts in sec_cat.items():
        total = sum(counts.values())
        section_cat_prob[sec] = {c: n / total for c, n in counts.items()}

    bundle_codes = {bid: extract_zara_code(url) for bid, url in bundle_urls.items()}
    product_codes = {pid: extract_zara_code(url) for pid, url in product_url.items()}

    model_to_products = defaultdict(list)
    prefix4_to_products = defaultdict(list)
    for pid, code in product_codes.items():
        if code:
            clean = code.lstrip('TM')
            if len(clean) >= 5:
                model_to_products[clean[:5]].append(pid)
            if len(clean) >= 4:
                prefix4_to_products[clean[:4]].append(pid)

    return {
        "bundle_section": bundle_section,
        "product_desc": product_desc,
        "all_product_ids": all_product_ids,
        "section_cat_prob": section_cat_prob,
        "bundle_codes": bundle_codes,
        "product_codes": product_codes,
        "model_to_products": model_to_products,
        "prefix4_to_products": prefix4_to_products,
    }


def build_text_embeddings(model, processor, device, categories):
    cat_list = sorted(categories)
    prompts = [f"a photo of {c.lower()}" for c in cat_list]
    all_feats = []
    for start in range(0, len(prompts), 32):
        batch = prompts[start:start + 32]
        tok = processor.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.no_grad():
            out = model.text_model(**tok)
            feats = model.text_projection(out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
    text_embs = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    return cat_list, text_embs, cat_to_idx


def encode_images_batch(images, model, processor, device):
    inputs = processor(images=list(images), return_tensors="pt", padding=True)
    pv = inputs['pixel_values'].to(device)
    with torch.no_grad():
        vout = model.vision_model(pixel_values=pv)
        embs = model.visual_projection(vout.pooler_output)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy().astype(np.float32)


def rrf_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking):
            scores[pid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════
# YOLO PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(bundle_ids, bundle_images_dir, yolo_boxes, model, processor, device,
                 index, all_product_ids, all_embeddings, text_embs, cat_list, cat_to_idx,
                 metadata, args):
    
    pid_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
    bundle_codes = metadata["bundle_codes"]
    product_codes = metadata["product_codes"]
    model_to_products = metadata["model_to_products"]
    prefix4_to_products = metadata["prefix4_to_products"]
    product_desc = metadata["product_desc"]
    section_cat_prob = metadata["section_cat_prob"]
    bundle_section = metadata["bundle_section"]

    results = []
    errors = 0
    start_time = time.time()

    for i, bundle_id in enumerate(bundle_ids):
        try:
            bundle_path = os.path.join(bundle_images_dir, f"{bundle_id}.jpg")
            if not os.path.exists(bundle_path):
                results.append((bundle_id, all_product_ids[:15]))
                errors += 1
                continue

            image = Image.open(bundle_path).convert("RGB")
            crops = [image] # Always include the full image
            
            # Add YOLO crops if available
            boxes = yolo_boxes.get(bundle_id, [])
            for box_data in boxes:
                if box_data["confidence"] > args.yolo_conf:
                    x1, y1, x2, y2 = box_data["box"]
                    # Expand crop slightly to give context
                    w, h = image.size
                    pad_x = (x2 - x1) * args.yolo_pad
                    pad_y = (y2 - y1) * args.yolo_pad
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    crops.append(image.crop((x1, y1, x2, y2)))

            # Cap max crops to avoid OOM
            if len(crops) > 8:
                crops = crops[:8]

            # Encode all crops (index 0 is full image)
            crop_embs = encode_images_batch(crops, model, processor, device)

            # Per-crop FAISS
            per_crop_rankings = []
            k_search = min(args.faiss_k, index.ntotal)
            
            for ci, emb in enumerate(crop_embs):
                q = emb.reshape(1, -1).copy()
                faiss.normalize_L2(q)
                scores, idxs = index.search(q, k_search)
                
                ranking = []
                for score, j in zip(scores[0], idxs[0]):
                    if j >= 0:
                        ranking.append((all_product_ids[j], float(score)))
                
                per_crop_rankings.append(ranking)

            # RRF Fusion
            rrf_results = rrf_fusion(per_crop_rankings, k=args.rrf_k)

            # Text-image scoring (using full image)
            full_query = crop_embs[0:1].copy()
            faiss.normalize_L2(full_query)
            text_sims = (full_query @ text_embs.T)[0]

            # URL Injection
            url_scores = {}
            b_code = bundle_codes.get(bundle_id)
            if b_code:
                b_clean = b_code.lstrip('TM')
                if len(b_clean) >= 5:
                    for pid in model_to_products.get(b_clean[:5], []):
                        url_scores[pid] = 3.0
                if len(b_clean) >= 4:
                    for pid in prefix4_to_products.get(b_clean[:4], []):
                        if pid not in url_scores:
                            url_scores[pid] = 1.5
                for pid in list(url_scores.keys()):
                    p_code = product_codes.get(pid)
                    if p_code:
                        p_clean = p_code.lstrip('TM')
                        if b_clean == p_clean:
                            url_scores[pid] = 5.0

            # Combine
            section = str(bundle_section.get(bundle_id, "1"))
            sec_probs = section_cat_prob.get(section, {})

            candidate_pool = {}
            for pid, rrf_score in rrf_results[:args.top_rrf]:
                candidate_pool[pid] = rrf_score
            
            for pid in url_scores:
                if pid not in candidate_pool:
                    candidate_pool[pid] = 0.0

            scored = []
            for pid in candidate_pool:
                rrf_score = candidate_pool[pid]
                
                cat = product_desc.get(pid, "")
                idx_c = cat_to_idx.get(cat)
                text_score = float(text_sims[idx_c]) if idx_c is not None else 0.0
                sec_score = sec_probs.get(cat, 0.0)
                url_boost = url_scores.get(pid, 0.0)

                # Direct similarity fallback
                if rrf_score == 0.0 and pid in pid_to_idx:
                    p_emb = all_embeddings[pid_to_idx[pid]:pid_to_idx[pid]+1].copy()
                    faiss.normalize_L2(p_emb)
                    direct_sim = float((full_query @ p_emb.T).item())
                    rrf_score = direct_sim * 0.01

                final = (args.w_rrf * rrf_score +
                         args.w_text * text_score +
                         args.w_section * sec_score +
                         args.w_url * url_boost)
                
                scored.append((final, pid))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_15 = [pid for _, pid in scored[:15]]

            # Pad
            if len(top_15) < 15:
                seen = set(top_15)
                for pid in all_product_ids:
                    if pid not in seen:
                        top_15.append(pid)
                        seen.add(pid)
                    if len(top_15) >= 15:
                        break

            results.append((bundle_id, top_15[:15]))

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ❌ {bundle_id}: {e}")
            results.append((bundle_id, all_product_ids[:15]))

        if (i + 1) % 50 == 0 or (i + 1) == len(bundle_ids):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(bundle_ids)}] {rate:.1f} b/s")

    print(f"\n✅ {len(bundle_ids)} bundles in {time.time()-start_time:.1f}s | {errors} errors")
    return results


def load_everything(args, base_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # Load YOLO boxes
    yolo_file = os.path.join(base_dir, "data", "yolo_test_bboxes.json")
    if not os.path.exists(yolo_file):
        raise FileNotFoundError(f"Missing {yolo_file}! Run extract_yolo_boxes.py first.")
    with open(yolo_file) as f:
        yolo_boxes = json.load(f)
    print(f"📦 Loaded YOLO boxes for {len(yolo_boxes)} bundles")

    metadata = build_metadata(base_dir)

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    if args.custom_weights:
        ckpt = torch.load(args.custom_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("  ✅ Custom weights!")
    model.eval()

    unique_cats = sorted(set(metadata["product_desc"].values()))
    cat_list, text_embs, cat_to_idx = build_text_embeddings(model, processor, device, unique_cats)

    emb_dir = os.path.join(base_dir, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    idx = faiss.read_index(os.path.join(emb_dir, f"clip_faiss_{suffix}.bin"))
    with open(os.path.join(emb_dir, f"clip_ids_{suffix}.pkl"), 'rb') as f:
        all_pids = pickle.load(f)
    all_embs = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * idx.d).reshape(idx.ntotal, idx.d).copy()
    print(f"  Index: {idx.ntotal}×{idx.d}")

    return model, processor, device, idx, all_pids, all_embs, text_embs, cat_list, cat_to_idx, metadata, yolo_boxes


def run_test(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, meta, yolo = load_everything(args, BASE)
    test_ids = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_test.csv"))['bundle_asset_id'].tolist()
    bdir = args.bundles_dir or os.path.join(BASE, "data/images/bundles")

    print(f"\n🎯 YOLO Inference: {len(test_ids)} TEST bundles")
    results = run_pipeline(test_ids, bdir, yolo, model, proc, dev, idx, pids, embs, te, cl, ci, meta, args)

    out = os.path.join(BASE, args.output)
    rows = [{"bundle_asset_id": b, "product_asset_id": p} for b, t in results for p in t]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"💾 {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    p.add_argument("--custom_weights", default=None)
    p.add_argument("--output", default="smart_yolo_submission.csv")
    
    p.add_argument("--w_rrf", type=float, default=1.0)
    p.add_argument("--w_text", type=float, default=0.3)
    p.add_argument("--w_section", type=float, default=0.2)
    p.add_argument("--w_url", type=float, default=2.0)
    
    p.add_argument("--faiss_k", type=int, default=200)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--top_rrf", type=int, default=300)
    
    p.add_argument("--yolo_conf", type=float, default=0.2, help="Min confidence for YOLO box")
    p.add_argument("--yolo_pad", type=float, default=0.05, help="Padding fraction around box")
    
    p.add_argument("--bundles_dir", default=None)
    args = p.parse_args()
    
    run_test(args)

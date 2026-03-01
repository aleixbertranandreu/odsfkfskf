"""
smart_inference_v5.py — V5: Multi-Crop FAISS + RRF Fusion + URL Injection
═════════════════════════════════════════════════════════════════════════

KEY STRATEGY:
  Bundle images contain MULTIPLE items (avg 3.5). A single CLIP encoding 
  captures the dominant item but misses accessories. SOLUTION:
  
  1. MULTIPLE CROPS of bundle image → separate FAISS searches per crop
  2. RRF (Reciprocal Rank Fusion) to combine results from all crops
  3. URL family injection (kept from V3)
  4. Text-image scoring for category-based re-ranking
  5. Section compatibility

  Crops: full, top half, bottom half, left half, right half, center 60%
"""
import argparse
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

    # Section → category probability
    sec_cat = defaultdict(Counter)
    for _, r in df_t.iterrows():
        bid, pid = r['bundle_asset_id'], r['product_asset_id']
        if bid in bundle_section and pid in product_desc:
            sec_cat[bundle_section[bid]][product_desc[pid]] += 1
    section_cat_prob = {}
    for sec, counts in sec_cat.items():
        total = sum(counts.values())
        section_cat_prob[sec] = {c: n / total for c, n in counts.items()}

    # URL codes
    bundle_codes = {bid: extract_zara_code(url) for bid, url in bundle_urls.items()}
    product_codes = {pid: extract_zara_code(url) for pid, url in product_url.items()}

    # Model/prefix indices
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


# ═══════════════════════════════════════════════════════════════
# TEXT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# MULTI-CROP ENCODING + FAISS
# ═══════════════════════════════════════════════════════════════

def get_crops(image):
    """Get multiple strategic crops from bundle image."""
    w, h = image.size
    crops = {
        "full": image,
        "top": image.crop((0, 0, w, int(h * 0.55))),
        "bottom": image.crop((0, int(h * 0.45), w, h)),
        "left": image.crop((0, 0, int(w * 0.55), h)),
        "right": image.crop((int(w * 0.45), 0, w, h)),
        "center": image.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))),
    }
    return crops


def encode_images_batch(images, model, processor, device):
    """Encode multiple images in a single batch."""
    inputs = processor(images=list(images), return_tensors="pt", padding=True)
    pv = inputs['pixel_values'].to(device)
    with torch.no_grad():
        vout = model.vision_model(pixel_values=pv)
        embs = model.visual_projection(vout.pooler_output)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs.cpu().numpy().astype(np.float32)


def rrf_fusion(rankings, k=60):
    """
    Reciprocal Rank Fusion: combine multiple ranked lists.
    RRF score = sum over rankings of 1 / (k + rank_i)
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking):
            scores[pid] += 1.0 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════
# V5 PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(bundle_ids, bundle_images_dir, model, processor, device,
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

            # ─── MULTI-CROP ENCODING ───
            crops = get_crops(image)
            crop_images = list(crops.values())
            crop_names = list(crops.keys())
            
            # Encode all crops in one batch
            crop_embs = encode_images_batch(crop_images, model, processor, device)

            # ─── PER-CROP FAISS SEARCH ───
            per_crop_rankings = []
            k_search = min(args.faiss_k, index.ntotal)
            
            for ci, (name, emb) in enumerate(zip(crop_names, crop_embs)):
                q = emb.reshape(1, -1).copy()
                faiss.normalize_L2(q)
                scores, idxs = index.search(q, k_search)
                
                ranking = []
                for score, j in zip(scores[0], idxs[0]):
                    if j >= 0:
                        ranking.append((all_product_ids[j], float(score)))
                
                per_crop_rankings.append(ranking)

            # ─── RRF FUSION across crops ───
            rrf_results = rrf_fusion(per_crop_rankings, k=args.rrf_k)

            # ─── TEXT-IMAGE SCORING (use full-image embedding) ───
            full_query = crop_embs[0:1].copy()
            faiss.normalize_L2(full_query)
            text_sims = (full_query @ text_embs.T)[0]

            # ─── URL FAMILY INJECTION ───
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
                # Exact match boost
                for pid in list(url_scores.keys()):
                    p_code = product_codes.get(pid)
                    if p_code:
                        p_clean = p_code.lstrip('TM')
                        if b_clean == p_clean:
                            url_scores[pid] = 5.0

            # ─── COMBINE ALL SIGNALS ───
            section = str(bundle_section.get(bundle_id, "1"))
            sec_probs = section_cat_prob.get(section, {})

            # Start with RRF scores
            candidate_pool = {}
            for pid, rrf_score in rrf_results[:args.top_rrf]:
                candidate_pool[pid] = rrf_score
            
            # Add URL-injected products
            for pid in url_scores:
                if pid not in candidate_pool:
                    candidate_pool[pid] = 0.0

            # Final scoring
            scored = []
            for pid in candidate_pool:
                rrf_score = candidate_pool[pid]
                
                # Text-image
                cat = product_desc.get(pid, "")
                idx_c = cat_to_idx.get(cat)
                text_score = float(text_sims[idx_c]) if idx_c is not None else 0.0
                
                # Section
                sec_score = sec_probs.get(cat, 0.0)
                
                # URL
                url_boost = url_scores.get(pid, 0.0)

                # Direct CLIP similarity (for URL-injected items not in RRF)
                if rrf_score == 0.0 and pid in pid_to_idx:
                    p_emb = all_embeddings[pid_to_idx[pid]:pid_to_idx[pid]+1].copy()
                    faiss.normalize_L2(p_emb)
                    direct_sim = float((full_query @ p_emb.T).item())
                    rrf_score = direct_sim * 0.01  # Scale to be comparable

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
                import traceback; traceback.print_exc()
            results.append((bundle_id, all_product_ids[:15]))

        if (i + 1) % 50 == 0 or (i + 1) == len(bundle_ids):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (i + 1) / len(bundle_ids)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}% | {i+1}/{len(bundle_ids)} | {rate:.1f} b/s")

    print(f"\n✅ {len(bundle_ids)} bundles in {time.time()-start_time:.1f}s | {errors} errors")
    return results


def load_everything(args, base_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

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

    return model, processor, device, idx, all_pids, all_embs, text_embs, cat_list, cat_to_idx, metadata


def run_test(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, meta = load_everything(args, BASE)
    test_ids = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_test.csv"))['bundle_asset_id'].tolist()
    bdir = args.bundles_dir or os.path.join(BASE, "data/images/bundles")

    print(f"\n🎯 V5: {len(test_ids)} TEST bundles | rrf_k={args.rrf_k} top_rrf={args.top_rrf}")
    results = run_pipeline(test_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, meta, args)

    out = os.path.join(BASE, args.output)
    rows = [{"bundle_asset_id": b, "product_asset_id": p} for b, t in results for p in t]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"💾 {out} ({len(rows)} rows)")


def run_eval(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, meta = load_everything(args, BASE)
    df = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_train.csv"))
    gt = defaultdict(list)
    for _, r in df.iterrows():
        gt[r['bundle_asset_id']].append(r['product_asset_id'])

    eval_ids = list(gt.keys())[:args.max_eval] if args.max_eval else list(gt.keys())
    bdir = args.bundles_dir or os.path.join(BASE, "data/images/bundles")

    print(f"\n🎯 V5 Eval: {len(eval_ids)} bundles | rrf_k={args.rrf_k} top_rrf={args.top_rrf}")
    results = run_pipeline(eval_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, meta, args)

    mrr_sum = 0.0
    hit_at = {1: 0, 5: 0, 10: 0, 15: 0}
    total = 0
    for bid, top15 in results:
        for tp in gt.get(bid, []):
            total += 1
            if tp in top15:
                rank = top15.index(tp) + 1
                mrr_sum += 1.0 / rank
                for k in hit_at:
                    if rank <= k:
                        hit_at[k] += 1
    if total > 0:
        mrr = mrr_sum / total
        print(f"\n🏆 V5: MRR@15={mrr:.4f} ({mrr*100:.2f}%)")
        for k in sorted(hit_at):
            print(f"   Hit@{k:2d}: {hit_at[k]/total:.4f} ({hit_at[k]}/{total})")
        return mrr


def run_sweep(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, meta = load_everything(args, BASE)
    df = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_train.csv"))
    gt = defaultdict(list)
    for _, r in df.iterrows():
        gt[r['bundle_asset_id']].append(r['product_asset_id'])

    eval_ids = list(gt.keys())[:args.max_eval] if args.max_eval else list(gt.keys())
    bdir = args.bundles_dir or os.path.join(BASE, "data/images/bundles")

    configs = [
        # V3 equivalent (no multi-crop, single global search)
        {"w_rrf": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "faiss_k": 200, "rrf_k": 60, "top_rrf": 300, "label": "V3-equiv"},
        # Multi-crop + RRF
        {"w_rrf": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 0.0, "faiss_k": 100, "rrf_k": 60, "top_rrf": 300, "label": "RRF only"},
        {"w_rrf": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 2.0, "faiss_k": 100, "rrf_k": 60, "top_rrf": 300, "label": "RRF+URL2"},
        {"w_rrf": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "faiss_k": 100, "rrf_k": 60, "top_rrf": 300, "label": "RRF+all"},
        {"w_rrf": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 3.0, "faiss_k": 100, "rrf_k": 60, "top_rrf": 300, "label": "RRF+highURL"},
        {"w_rrf": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "faiss_k": 200, "rrf_k": 60, "top_rrf": 500, "label": "RRF+K200"},
        {"w_rrf": 1.0, "w_text": 0.5, "w_section": 0.3, "w_url": 3.0, "faiss_k": 150, "rrf_k": 60, "top_rrf": 400, "label": "RRF+balanced"},
        # Different RRF k values
        {"w_rrf": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "faiss_k": 100, "rrf_k": 30, "top_rrf": 300, "label": "RRF_k30"},
        {"w_rrf": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "faiss_k": 100, "rrf_k": 100, "top_rrf": 300, "label": "RRF_k100"},
    ]

    print(f"\n🔬 V5 Sweep: {len(configs)} configs on {len(eval_ids)} bundles\n")
    best_mrr = 0
    best_cfg = None

    for cfg in configs:
        for k, v in cfg.items():
            if k != "label":
                setattr(args, k, v)

        results = run_pipeline(eval_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, meta, args)

        mrr_sum = 0.0
        tot = 0
        for bid, top15 in results:
            for tp in gt.get(bid, []):
                tot += 1
                if tp in top15:
                    mrr_sum += 1.0 / (top15.index(tp) + 1)

        mrr = mrr_sum / tot if tot > 0 else 0
        m = " ★" if mrr > best_mrr else ""
        print(f"  {cfg['label']:20s} → {mrr:.4f} ({mrr*100:.2f}%){m}")
        if mrr > best_mrr:
            best_mrr = mrr
            best_cfg = cfg.copy()

    print(f"\n🏆 Best: {best_cfg}\n   MRR = {best_mrr:.4f}")
    return best_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    p.add_argument("--custom_weights", default=None)
    p.add_argument("--output", default="smart_v5_submission.csv")
    
    p.add_argument("--w_rrf", type=float, default=1.0)
    p.add_argument("--w_text", type=float, default=0.3)
    p.add_argument("--w_section", type=float, default=0.2)
    p.add_argument("--w_url", type=float, default=2.0)
    p.add_argument("--faiss_k", type=int, default=100)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--top_rrf", type=int, default=300)
    
    p.add_argument("--local_eval", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--bundles_dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.local_eval:
        run_eval(args)
    else:
        run_test(args)

"""
smart_inference_v4.py — V4: Category-Diversified Multi-Query Search
════════════════════════════════════════════════════════════════════════════

KEY INSIGHT: Bundles contain ~3.5 items across ~3.3 categories. Current FAISS 
returns 100 items biased toward the DOMINANT category. But we need to find EACH
item: the trousers AND the shirt AND the shoes AND the bag.

V4 STRATEGY:
  1. CLIP text-image: detect likely categories in the bundle image
  2. Per-category FAISS: do SEPARATE searches within each detected category's 
     product subset — this finds the BEST match per category
  3. Co-occurrence boost: categories that frequently appear together get boosted
  4. URL family injection: same as V3 (5-digit Zara model code)
  5. Global FAISS: fallback full search for items the category search misses
  6. Rank fusion: combine all signals into final top-15
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
# DATA / METADATA
# ═══════════════════════════════════════════════════════════════

def extract_zara_code(url):
    try:
        m = re.search(r'([A-Z]?\d{8,11})', url.split('/')[-2])
        return m.group(1) if m else None
    except:
        return None


def build_metadata(base_dir):
    bundles_csv = os.path.join(base_dir, "data", "raw", "bundles_dataset.csv")
    products_csv = os.path.join(base_dir, "data", "raw", "product_dataset.csv")
    train_csv = os.path.join(base_dir, "data", "raw", "bundles_product_match_train.csv")

    df_b = pd.read_csv(bundles_csv)
    df_p = pd.read_csv(products_csv)
    df_t = pd.read_csv(train_csv)

    bundle_section = dict(zip(df_b['bundle_asset_id'], df_b['bundle_id_section'].astype(str)))
    bundle_urls = dict(zip(df_b['bundle_asset_id'], df_b['bundle_image_url']))

    product_desc = dict(zip(df_p['product_asset_id'], df_p['product_description']))
    product_url = dict(zip(df_p['product_asset_id'], df_p['product_image_url']))
    all_product_ids = df_p['product_asset_id'].tolist()

    # Section → category probability
    sec_cat = defaultdict(Counter)
    for _, row in df_t.iterrows():
        bid, pid = row['bundle_asset_id'], row['product_asset_id']
        if bid in bundle_section and pid in product_desc:
            sec_cat[bundle_section[bid]][product_desc[pid]] += 1
    
    section_cat_prob = {}
    for sec, counts in sec_cat.items():
        total = sum(counts.values())
        section_cat_prob[sec] = {c: n / total for c, n in counts.items()}

    # Co-occurrence matrix: P(cat_B | cat_A) for same-bundle pairs
    cooccur = defaultdict(Counter)
    bundle_cats = {}
    for _, row in df_t.iterrows():
        bid, pid = row['bundle_asset_id'], row['product_asset_id']
        if pid in product_desc:
            bundle_cats.setdefault(bid, set()).add(product_desc[pid])
    
    for bid, cats in bundle_cats.items():
        for c1 in cats:
            for c2 in cats:
                if c1 != c2:
                    cooccur[c1][c2] += 1
    
    # Normalize
    cooccur_prob = {}
    for c1, others in cooccur.items():
        total = sum(others.values())
        cooccur_prob[c1] = {c2: n / total for c2, n in others.items()}

    # URL codes
    bundle_codes = {bid: extract_zara_code(url) for bid, url in bundle_urls.items()}
    product_codes = {pid: extract_zara_code(url) for pid, url in product_url.items()}

    # Model code → product list
    model_to_products = defaultdict(list)
    prefix4_to_products = defaultdict(list)
    for pid, code in product_codes.items():
        if code:
            clean = code.lstrip('TM')
            if len(clean) >= 5:
                model_to_products[clean[:5]].append(pid)
            if len(clean) >= 4:
                prefix4_to_products[clean[:4]].append(pid)

    # Category → product IDs (for per-category FAISS)
    cat_to_pids = defaultdict(list)
    for pid, desc in product_desc.items():
        cat_to_pids[desc].append(pid)

    print(f"  {len(cat_to_pids)} categories, {len(model_to_products)} model groups")
    print(f"  {len(cooccur_prob)} category co-occurrence entries")

    return {
        "bundle_section": bundle_section,
        "bundle_urls": bundle_urls,
        "product_desc": product_desc,
        "product_url": product_url,
        "all_product_ids": all_product_ids,
        "section_cat_prob": section_cat_prob,
        "cooccur_prob": cooccur_prob,
        "bundle_codes": bundle_codes,
        "product_codes": product_codes,
        "model_to_products": model_to_products,
        "prefix4_to_products": prefix4_to_products,
        "cat_to_pids": cat_to_pids,
    }


# ═══════════════════════════════════════════════════════════════
# TEXT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def build_text_embeddings(model, processor, device, categories):
    cat_list = sorted(categories)
    prompts = [f"a photo of {cat.lower()}" for cat in cat_list]
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
# PER-CATEGORY FAISS INDICES
# ═══════════════════════════════════════════════════════════════

def build_category_indices(all_product_ids, all_embeddings, cat_to_pids, pid_to_idx, dim):
    """Build a mini-FAISS index for each product category."""
    cat_indices = {}
    
    for cat, pids in cat_to_pids.items():
        valid_pids = [pid for pid in pids if pid in pid_to_idx]
        if len(valid_pids) < 2:
            continue
        
        indices = [pid_to_idx[pid] for pid in valid_pids]
        embs = all_embeddings[indices].copy()
        faiss.normalize_L2(embs)
        
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)
        
        cat_indices[cat] = {
            "index": idx,
            "pids": valid_pids,
        }
    
    print(f"  Built {len(cat_indices)} category mini-indices")
    return cat_indices


# ═══════════════════════════════════════════════════════════════
# V4 PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(bundle_ids, bundle_images_dir, model, processor, device,
                 index, all_product_ids, all_embeddings, text_embs, cat_list, cat_to_idx,
                 cat_indices, metadata, args):
    
    pid_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
    bundle_codes = metadata["bundle_codes"]
    product_codes = metadata["product_codes"]
    model_to_products = metadata["model_to_products"]
    prefix4_to_products = metadata["prefix4_to_products"]
    product_desc = metadata["product_desc"]
    section_cat_prob = metadata["section_cat_prob"]
    cooccur_prob = metadata["cooccur_prob"]
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

            # ─── ENCODE BUNDLE ───
            inputs = processor(images=image, return_tensors="pt")
            pv = inputs['pixel_values'].to(device)
            with torch.no_grad():
                vout = model.vision_model(pixel_values=pv)
                bundle_emb = model.visual_projection(vout.pooler_output)
                bundle_emb = bundle_emb / bundle_emb.norm(dim=-1, keepdim=True)
            bundle_np = bundle_emb.cpu().numpy().astype(np.float32)
            
            query = bundle_np.copy()
            faiss.normalize_L2(query)

            # ─── DETECT CATEGORIES IN BUNDLE ───
            text_sims = (query @ text_embs.T)[0]
            cat_scores = [(cat_list[j], float(text_sims[j])) for j in range(len(cat_list))]
            cat_scores.sort(key=lambda x: x[1], reverse=True)

            # Get section info
            section = str(bundle_section.get(bundle_id, "1"))
            sec_probs = section_cat_prob.get(section, {})

            # Select top categories, boosted by section probability & co-occurrence
            # Start with text-image similarity, then expand via co-occurrence
            top_cats_set = set()
            cat_weight = {}
            
            for cat, sim in cat_scores[:args.n_cats]:
                sec_p = sec_probs.get(cat, 0.0)
                combined = sim + args.w_section_cat * sec_p
                if combined > args.cat_threshold:
                    top_cats_set.add(cat)
                    cat_weight[cat] = combined
            
            # Expand via co-occurrence: if TROUSERS detected, also consider SHIRT, SHOES, etc.
            expanded_cats = set(top_cats_set)
            for cat in top_cats_set:
                coocs = cooccur_prob.get(cat, {})
                for co_cat, prob in sorted(coocs.items(), key=lambda x: -x[1])[:5]:
                    if prob >= 0.05:  # at least 5% co-occurrence
                        expanded_cats.add(co_cat)
                        cat_weight[co_cat] = cat_weight.get(co_cat, 0.0) + prob * 0.3
            
            # ─── PER-CATEGORY FAISS SEARCH ───
            per_cat_results = {}  # pid → score
            
            for cat in expanded_cats:
                if cat not in cat_indices:
                    continue
                cidx = cat_indices[cat]
                k = min(args.per_cat_k, cidx["index"].ntotal)
                s, idxs = cidx["index"].search(query, k)
                
                w = cat_weight.get(cat, 0.1)
                for score, j in zip(s[0], idxs[0]):
                    if j < 0:
                        continue
                    pid = cidx["pids"][j]
                    # Weight by category relevance
                    weighted_score = float(score) * (1.0 + args.w_cat_boost * w)
                    per_cat_results[pid] = max(per_cat_results.get(pid, 0.0), weighted_score)

            # ─── GLOBAL FAISS SEARCH ───
            k_global = min(args.faiss_k, index.ntotal)
            g_scores, g_idxs = index.search(query, k_global)
            global_results = {}
            for score, j in zip(g_scores[0], g_idxs[0]):
                if j >= 0:
                    pid = all_product_ids[j]
                    global_results[pid] = float(score)

            # ─── URL FAMILY INJECTION ───
            url_results = {}
            b_code = bundle_codes.get(bundle_id)
            if b_code:
                b_clean = b_code.lstrip('TM')
                
                if len(b_clean) >= 5:
                    for pid in model_to_products.get(b_clean[:5], []):
                        url_results[pid] = 3.0  # Same model
                
                if len(b_clean) >= 4:
                    for pid in prefix4_to_products.get(b_clean[:4], []):
                        if pid not in url_results:
                            url_results[pid] = 1.5  # Same broad family

                # Exact match gets highest boost
                for pid in url_results:
                    p_code = product_codes.get(pid)
                    if p_code:
                        p_clean = p_code.lstrip('TM')
                        if b_clean == p_clean:
                            url_results[pid] = 5.0

            # ─── COMBINE ALL SIGNALS ───
            all_candidates = set(global_results.keys()) | set(per_cat_results.keys()) | set(url_results.keys())
            
            scored = []
            for pid in all_candidates:
                # Image-image (global)
                img_score = global_results.get(pid, 0.0)
                
                # Per-category score  
                cat_score = per_cat_results.get(pid, 0.0)
                
                # URL boost
                url_score = url_results.get(pid, 0.0)
                
                # Text-image similarity for this product's category
                cat = product_desc.get(pid, "")
                idx_c = cat_to_idx.get(cat)
                text_score = float(text_sims[idx_c]) if idx_c is not None else 0.0
                
                # Section compatibility
                sec_score = sec_probs.get(cat, 0.0)
                
                # Direct embedding similarity if no FAISS score available
                if img_score == 0.0 and cat_score == 0.0 and pid in pid_to_idx:
                    p_emb = all_embeddings[pid_to_idx[pid]:pid_to_idx[pid]+1].copy()
                    faiss.normalize_L2(p_emb)
                    img_score = float((query @ p_emb.T).item())

                final = (args.w_img * img_score +
                         args.w_cat_search * cat_score +
                         args.w_text * text_score +
                         args.w_section * sec_score +
                         args.w_url * url_score)
                
                scored.append((final, pid))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Diversify: ensure we don't have too many items from same category
            if args.diversify:
                top_15 = diversify_results(scored, product_desc, max_per_cat=args.max_per_cat)
            else:
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

    total_time = time.time() - start_time
    print(f"\n✅ Done! {len(bundle_ids)} in {total_time:.1f}s | {errors} errors")
    return results


def diversify_results(scored, product_desc, max_per_cat=4):
    """Ensure no category dominates the top-15."""
    selected = []
    cat_count = Counter()
    
    for score, pid in scored:
        cat = product_desc.get(pid, "UNKNOWN")
        if cat_count[cat] < max_per_cat:
            selected.append(pid)
            cat_count[cat] += 1
        if len(selected) >= 15:
            break
    
    return selected


# ═══════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════

def load_everything(args, base_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    print("\n📋 Loading metadata...")
    metadata = build_metadata(base_dir)

    print(f"\n🧠 Loading FashionCLIP...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    if args.custom_weights:
        ckpt = torch.load(args.custom_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("  ✅ Custom weights loaded!")
    model.eval()

    print("\n📝 Text embeddings...")
    unique_cats = sorted(set(metadata["product_desc"].values()))
    cat_list, text_embs, cat_to_idx = build_text_embeddings(model, processor, device, unique_cats)

    print("\n📦 FAISS index...")
    emb_dir = os.path.join(base_dir, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    idx = faiss.read_index(os.path.join(emb_dir, f"clip_faiss_{suffix}.bin"))
    with open(os.path.join(emb_dir, f"clip_ids_{suffix}.pkl"), 'rb') as f:
        all_pids = pickle.load(f)
    all_embs = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * idx.d).reshape(idx.ntotal, idx.d).copy()
    print(f"  {idx.ntotal} × {idx.d}")

    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}

    print("\n🏷️  Building category indices...")
    cat_indices = build_category_indices(all_pids, all_embs, metadata["cat_to_pids"], pid_to_idx, idx.d)

    return model, processor, device, idx, all_pids, all_embs, text_embs, cat_list, cat_to_idx, cat_indices, metadata


# ═══════════════════════════════════════════════════════════════
# MODES
# ═══════════════════════════════════════════════════════════════

def run_test(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, cat_idx, meta = load_everything(args, BASE)

    test_csv = os.path.join(BASE, "data", "raw", "bundles_product_match_test.csv")
    test_ids = pd.read_csv(test_csv)['bundle_asset_id'].tolist()
    bdir = args.bundles_dir or os.path.join(BASE, "data", "images", "bundles")

    print(f"\n{'═'*60}")
    print(f"🎯 V4: {len(test_ids)} TEST bundles")
    print(f"   img={args.w_img} cat={args.w_cat_search} text={args.w_text} sec={args.w_section} url={args.w_url}")
    print(f"   K={args.faiss_k} per_cat_k={args.per_cat_k} n_cats={args.n_cats}")
    print(f"   diversify={'ON' if args.diversify else 'OFF'} max_per_cat={args.max_per_cat}")
    print(f"{'═'*60}\n")

    results = run_pipeline(test_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, cat_idx, meta, args)

    out = os.path.join(BASE, args.output)
    rows = [{"bundle_asset_id": bid, "product_asset_id": pid} for bid, top15 in results for pid in top15]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n💾 {out} ({len(rows)} rows)")


def run_eval(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, cat_idx, meta = load_everything(args, BASE)

    df_t = pd.read_csv(os.path.join(BASE, "data", "raw", "bundles_product_match_train.csv"))
    gt = defaultdict(list)
    for _, r in df_t.iterrows():
        gt[r['bundle_asset_id']].append(r['product_asset_id'])

    eval_ids = list(gt.keys())[:args.max_eval] if args.max_eval else list(gt.keys())
    bdir = args.bundles_dir or os.path.join(BASE, "data", "images", "bundles")

    print(f"\n{'═'*60}")
    print(f"🎯 V4 Eval: {len(eval_ids)} TRAIN bundles")
    print(f"   img={args.w_img} cat={args.w_cat_search} text={args.w_text} sec={args.w_section} url={args.w_url}")
    print(f"   K={args.faiss_k} per_cat_k={args.per_cat_k} n_cats={args.n_cats}")
    print(f"   diversify={'ON' if args.diversify else 'OFF'} max_per_cat={args.max_per_cat}")
    print(f"{'═'*60}\n")

    results = run_pipeline(eval_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, cat_idx, meta, args)

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
        print(f"\n{'═'*60}")
        print(f"🏆 V4 RESULTS")
        print(f"{'═'*60}")
        print(f"  MRR@15:  {mrr:.4f} ({mrr*100:.2f}%)")
        for k in sorted(hit_at):
            print(f"  Hit@{k:2d}:  {hit_at[k]/total:.4f} ({hit_at[k]}/{total})")
        print(f"  Total: {total}")
        return mrr


def run_sweep(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, cat_idx, meta = load_everything(args, BASE)

    df_t = pd.read_csv(os.path.join(BASE, "data", "raw", "bundles_product_match_train.csv"))
    gt = defaultdict(list)
    for _, r in df_t.iterrows():
        gt[r['bundle_asset_id']].append(r['product_asset_id'])

    eval_ids = list(gt.keys())[:args.max_eval] if args.max_eval else list(gt.keys())
    bdir = args.bundles_dir or os.path.join(BASE, "data", "images", "bundles")

    configs = [
        # Baseline: V3 equivalent (no per-category, no diversify)
        {"w_img": 1.0, "w_cat_search": 0.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "n_cats": 0, "per_cat_k": 5, "diversify": False, "max_per_cat": 15, "label": "V3 baseline"},
        # V4: +category search
        {"w_img": 1.0, "w_cat_search": 0.5, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "n_cats": 8, "per_cat_k": 5, "diversify": False, "max_per_cat": 15, "label": "+cat_search"},
        # V4: +category search + diversify
        {"w_img": 1.0, "w_cat_search": 0.5, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "n_cats": 8, "per_cat_k": 5, "diversify": True, "max_per_cat": 4, "label": "+cat+diversify4"},
        {"w_img": 1.0, "w_cat_search": 0.5, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "n_cats": 8, "per_cat_k": 5, "diversify": True, "max_per_cat": 3, "label": "+cat+diversify3"},
        # Higher cat weight
        {"w_img": 1.0, "w_cat_search": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "n_cats": 10, "per_cat_k": 5, "diversify": True, "max_per_cat": 4, "label": "high_cat K10"},
        # More per-cat results
        {"w_img": 1.0, "w_cat_search": 0.5, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "n_cats": 8, "per_cat_k": 10, "diversify": True, "max_per_cat": 4, "label": "per_cat_k=10"},
        # Heavy URL
        {"w_img": 1.0, "w_cat_search": 0.5, "w_text": 0.3, "w_section": 0.2, "w_url": 4.0, "n_cats": 8, "per_cat_k": 5, "diversify": True, "max_per_cat": 4, "label": "heavy_url"},
        # Text-heavy
        {"w_img": 1.0, "w_cat_search": 0.5, "w_text": 0.8, "w_section": 0.3, "w_url": 2.0, "n_cats": 8, "per_cat_k": 5, "diversify": True, "max_per_cat": 4, "label": "text_heavy"},
        # Balanced high
        {"w_img": 1.0, "w_cat_search": 0.8, "w_text": 0.5, "w_section": 0.3, "w_url": 3.0, "n_cats": 10, "per_cat_k": 8, "diversify": True, "max_per_cat": 3, "label": "balanced_high"},
        # Max diversity
        {"w_img": 1.0, "w_cat_search": 0.8, "w_text": 0.5, "w_section": 0.3, "w_url": 3.0, "n_cats": 12, "per_cat_k": 8, "diversify": True, "max_per_cat": 2, "label": "max_diversity"},
    ]

    print(f"\n🔬 V4 Sweep: {len(configs)} configs on {len(eval_ids)} bundles\n")
    best_mrr = 0
    best_cfg = None

    for cfg in configs:
        for k, v in cfg.items():
            if k != "label":
                setattr(args, k, v)

        results = run_pipeline(eval_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, cat_idx, meta, args)

        mrr_sum = 0.0
        tot = 0
        for bid, top15 in results:
            for tp in gt.get(bid, []):
                tot += 1
                if tp in top15:
                    rank = top15.index(tp) + 1
                    mrr_sum += 1.0 / rank

        mrr = mrr_sum / tot if tot > 0 else 0
        marker = " ★ BEST" if mrr > best_mrr else ""
        print(f"  {cfg['label']:25s} → MRR@15 = {mrr:.4f} ({mrr*100:.2f}%){marker}")
        if mrr > best_mrr:
            best_mrr = mrr
            best_cfg = cfg.copy()

    print(f"\n🏆 Best: {best_cfg}")
    print(f"   MRR@15 = {best_mrr:.4f}")
    return best_cfg


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Smart V4")
    p.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    p.add_argument("--custom_weights", type=str, default=None)
    p.add_argument("--output", default="smart_v4_submission.csv")
    
    p.add_argument("--w_img", type=float, default=1.0)
    p.add_argument("--w_cat_search", type=float, default=0.5)
    p.add_argument("--w_text", type=float, default=0.3)
    p.add_argument("--w_section", type=float, default=0.2)
    p.add_argument("--w_url", type=float, default=2.0)
    p.add_argument("--w_cat_boost", type=float, default=0.5, help="Category relevance boost factor")
    p.add_argument("--w_section_cat", type=float, default=0.5, help="Section-category affinity weight for detection")
    
    p.add_argument("--faiss_k", type=int, default=200)
    p.add_argument("--per_cat_k", type=int, default=5)
    p.add_argument("--n_cats", type=int, default=8, help="Top categories to search per-category")
    p.add_argument("--cat_threshold", type=float, default=0.0, help="Min score for category detection")
    
    p.add_argument("--diversify", action="store_true")
    p.add_argument("--max_per_cat", type=int, default=4)
    
    p.add_argument("--local_eval", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--bundles_dir", type=str, default=None)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.local_eval:
        run_eval(args)
    else:
        run_test(args)

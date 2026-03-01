"""
smart_inference_v3.py — V3 Pipeline: URL Family Injection + CLIP text-image + Smart Re-ranking
════════════════════════════════════════════════════════════════════════════════════════════════

BREAKTHROUGH STRATEGY:
  1. FULL FAISS search (top K=200 candidates from image-to-image CLIP)
  2. URL FAMILY INJECTION: Find ALL products sharing first 5 digits of Zara code 
     with the bundle → force-inject them into candidates (23% coverage, very precise)
  3. CLIP TEXT-to-IMAGE scoring: For each candidate, score how well its product 
     description matches the bundle image (cross-modal matching)
  4. Combined re-ranking: image_sim + text_sim + url_boost + section
"""
import argparse
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ═════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════

def extract_zara_code(url):
    """Extract the numeric product code from a Zara URL."""
    try:
        filename = url.split('/')[-2]
        match = re.search(r'([A-Z]?\d{8,11})', filename)
        if match:
            return match.group(1)
        return None
    except:
        return None


def build_metadata(base_dir):
    """Load all metadata and build lookup tables."""
    bundles_csv = os.path.join(base_dir, "data", "raw", "bundles_dataset.csv")
    products_csv = os.path.join(base_dir, "data", "raw", "product_dataset.csv")
    train_csv = os.path.join(base_dir, "data", "raw", "bundles_product_match_train.csv")

    df_bundles = pd.read_csv(bundles_csv)
    df_products = pd.read_csv(products_csv)
    df_train = pd.read_csv(train_csv)

    bundle_section = dict(zip(df_bundles['bundle_asset_id'], df_bundles['bundle_id_section'].astype(str)))
    bundle_urls = dict(zip(df_bundles['bundle_asset_id'], df_bundles['bundle_image_url']))

    product_desc = dict(zip(df_products['product_asset_id'], df_products['product_description']))
    product_url = dict(zip(df_products['product_asset_id'], df_products['product_image_url']))
    all_product_ids = df_products['product_asset_id'].tolist()

    # Section → category probability
    section_cat_counts = defaultdict(Counter)
    for _, row in df_train.iterrows():
        bid = row['bundle_asset_id']
        pid = row['product_asset_id']
        if bid in bundle_section and pid in product_desc:
            section_cat_counts[bundle_section[bid]][product_desc[pid]] += 1
    
    section_cat_prob = {}
    for sec, counts in section_cat_counts.items():
        total = sum(counts.values())
        section_cat_prob[sec] = {cat: cnt / total for cat, cnt in counts.items()}

    # Zara URL codes
    bundle_codes = {bid: extract_zara_code(url) for bid, url in bundle_urls.items()}
    product_codes = {pid: extract_zara_code(url) for pid, url in product_url.items()}

    # Build model code → product list (first 5 digits of Zara code)
    model_to_products = defaultdict(list)
    for pid, code in product_codes.items():
        if code:
            clean = code.lstrip('TM')
            if len(clean) >= 5:
                model_to_products[clean[:5]].append(pid)
    
    # Also build 4-digit prefix index for broader matching
    prefix4_to_products = defaultdict(list)
    for pid, code in product_codes.items():
        if code:
            clean = code.lstrip('TM')
            if len(clean) >= 4:
                prefix4_to_products[clean[:4]].append(pid)

    print(f"  Models (5-digit groups): {len(model_to_products)}")
    print(f"  Prefixes (4-digit groups): {len(prefix4_to_products)}")

    return {
        "bundle_section": bundle_section,
        "bundle_urls": bundle_urls,
        "product_desc": product_desc,
        "product_url": product_url,
        "all_product_ids": all_product_ids,
        "section_cat_prob": section_cat_prob,
        "bundle_codes": bundle_codes,
        "product_codes": product_codes,
        "model_to_products": model_to_products,
        "prefix4_to_products": prefix4_to_products,
    }


# ═════════════════════════════════════════════════════════════════
# TEXT-TO-IMAGE SCORING
# ═════════════════════════════════════════════════════════════════

def build_text_embeddings(model, processor, device, categories):
    """Encode product categories as CLIP text embeddings."""
    cat_list = sorted(categories)
    prompts = [f"a photo of {cat.lower()}" for cat in cat_list]

    all_text_features = []
    batch_size = 32
    
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        tokenized = processor.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            text_out = model.text_model(**tokenized)
            feats = model.text_projection(text_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        
        all_text_features.append(feats.cpu())
    
    text_embs = torch.cat(all_text_features, dim=0).numpy().astype(np.float32)
    cat_to_idx = {cat: i for i, cat in enumerate(cat_list)}
    
    return cat_list, text_embs, cat_to_idx


# ═════════════════════════════════════════════════════════════════
# V3 PIPELINE: Combined Scoring
# ═════════════════════════════════════════════════════════════════

def run_pipeline(bundle_ids, bundle_images_dir, model, processor, device,
                 index, all_product_ids, all_embeddings, text_embs, cat_list, cat_to_idx,
                 metadata, args):
    """V3 pipeline with URL family injection + text-image scoring."""
    
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
    stats = {"url_injected": 0, "url_boosted": 0}
    start_time = time.time()

    for i, bundle_id in enumerate(bundle_ids):
        try:
            bundle_path = os.path.join(bundle_images_dir, f"{bundle_id}.jpg")
            
            if not os.path.exists(bundle_path):
                results.append((bundle_id, all_product_ids[:15]))
                errors += 1
                continue

            image = Image.open(bundle_path).convert("RGB")

            # ─── CLIP Image Encoding ───
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=pixel_values)
                bundle_emb = model.visual_projection(vision_out.pooler_output)
                bundle_emb = bundle_emb / bundle_emb.norm(dim=-1, keepdim=True)
            
            bundle_emb_np = bundle_emb.cpu().numpy().astype(np.float32)

            # ─── FAISS Search ───
            query = bundle_emb_np.copy()
            faiss.normalize_L2(query)
            k_search = min(args.faiss_k, index.ntotal)
            scores, indices = index.search(query, k_search)
            
            faiss_pids = [all_product_ids[idx] for idx in indices[0] if idx >= 0]
            faiss_scores = {pid: float(s) for pid, s in zip(faiss_pids, scores[0][:len(faiss_pids)])}

            # ─── CLIP Text-Image Scoring ───
            # Compute similarity between bundle image and each category text embedding
            text_sims = (query @ text_embs.T)[0]  # similarity with each category
            cat_sim_dict = {cat: float(text_sims[idx]) for cat, idx in cat_to_idx.items()}

            # ─── URL Family Injection ───
            b_code = bundle_codes.get(bundle_id)
            injected_pids = set()
            
            if b_code:
                b_clean = b_code.lstrip('TM')
                
                # Find products sharing model code (first 5 digits)
                if len(b_clean) >= 5:
                    model_matches = model_to_products.get(b_clean[:5], [])
                    for pid in model_matches:
                        injected_pids.add(pid)
                
                # Also check 4-digit prefix for broader matches
                if len(b_clean) >= 4:
                    prefix4_matches = prefix4_to_products.get(b_clean[:4], [])
                    for pid in prefix4_matches:
                        injected_pids.add(pid)
            
            stats["url_injected"] += len(injected_pids)

            # ─── Build final candidate pool ───
            all_candidates = set(faiss_pids) | injected_pids
            
            # Get section info
            section = str(bundle_section.get(bundle_id, "1"))
            sec_probs = section_cat_prob.get(section, {})

            # ─── Score all candidates ───
            scored = []
            for pid in all_candidates:
                # 1. CLIP image-image similarity
                img_sim = faiss_scores.get(pid, 0.0)
                
                # If not in FAISS results, compute similarity directly
                if pid not in faiss_scores and pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    p_emb = all_embeddings[idx:idx+1].copy()
                    faiss.normalize_L2(p_emb)
                    img_sim = float(query @ p_emb.T)
                
                # 2. CLIP text-image similarity (product desc → bundle image)
                cat = product_desc.get(pid, "UNKNOWN")
                text_sim = cat_sim_dict.get(cat, 0.0)
                
                # 3. Section compatibility
                sec_score = sec_probs.get(cat, 0.0)
                
                # 4. URL code boost
                url_boost = 0.0
                if b_code:
                    p_code = product_codes.get(pid)
                    if p_code:
                        b_clean = b_code.lstrip('TM')
                        p_clean = p_code.lstrip('TM')
                        
                        if b_clean == p_clean:
                            url_boost = 3.0      # Exact match
                        elif len(b_clean) >= 7 and len(p_clean) >= 7 and b_clean[:7] == p_clean[:7]:
                            url_boost = 2.5      # Same model + color
                        elif len(b_clean) >= 5 and len(p_clean) >= 5 and b_clean[:5] == p_clean[:5]:
                            url_boost = 2.0      # Same model/collection
                        elif len(b_clean) >= 4 and len(p_clean) >= 4 and b_clean[:4] == p_clean[:4]:
                            url_boost = 1.0      # Same broad family
                        elif len(b_clean) >= 3 and len(p_clean) >= 3 and b_clean[:3] == p_clean[:3]:
                            url_boost = 0.3      # Same department
                        
                        if url_boost > 0:
                            stats["url_boosted"] += 1
                
                # Combined score
                final = (args.w_img * img_sim + 
                         args.w_text * text_sim + 
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
            if errors <= 5:
                print(f"  ❌ {bundle_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append((bundle_id, all_product_ids[:15]))

        if (i + 1) % 50 == 0 or (i + 1) == len(bundle_ids):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (i + 1) / len(bundle_ids)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}% | {i+1}/{len(bundle_ids)} | {rate:.1f} b/s")

    total_time = time.time() - start_time
    n = len(bundle_ids)
    print(f"\n✅ Done! {n} bundles in {total_time:.1f}s | {errors} errors")
    print(f"   Avg URL-injected candidates/bundle: {stats['url_injected']/n:.1f}")
    print(f"   URL-boosted product-bundle pairs: {stats['url_boosted']}")

    return results


def load_everything(args, base_dir):
    """Load model, index, text embeddings, metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    print("\n📋 Loading metadata...")
    metadata = build_metadata(base_dir)

    print(f"\n🧠 Loading FashionCLIP ({args.model_name})...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    if args.custom_weights:
        checkpoint = torch.load(args.custom_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✅ Custom weights loaded!")
    model.eval()

    print("\n📝 Building text embeddings...")
    unique_cats = sorted(set(metadata["product_desc"].values()))
    print(f"  {len(unique_cats)} categories")
    cat_list, text_embs, cat_to_idx = build_text_embeddings(model, processor, device, unique_cats)

    print("\n📦 Loading FAISS index...")
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    faiss_path = os.path.join(embeddings_dir, f"clip_faiss_{suffix}.bin")
    ids_path = os.path.join(embeddings_dir, f"clip_ids_{suffix}.pkl")

    index = faiss.read_index(faiss_path)
    with open(ids_path, 'rb') as f:
        all_product_ids = pickle.load(f)
    
    all_embeddings = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d).copy()
    print(f"  Index: {index.ntotal} × {index.d}")

    return model, processor, device, index, all_product_ids, all_embeddings, text_embs, cat_list, cat_to_idx, metadata


# ═════════════════════════════════════════════════════════════════
# SUBMISSION
# ═════════════════════════════════════════════════════════════════

def run_test(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model, processor, device, index, all_pids, all_embs, text_embs, cat_list, cat_to_idx, metadata = \
        load_everything(args, BASE_DIR)

    test_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv")
    df_test = pd.read_csv(test_csv)
    test_ids = df_test['bundle_asset_id'].tolist()
    bundles_dir = args.bundles_dir or os.path.join(BASE_DIR, "data", "images", "bundles")

    print(f"\n{'═'*60}")
    print(f"🎯 V3 Pipeline: {len(test_ids)} TEST bundles")
    print(f"   img={args.w_img} text={args.w_text} sec={args.w_section} url={args.w_url} K={args.faiss_k}")
    print(f"{'═'*60}\n")

    results = run_pipeline(test_ids, bundles_dir, model, processor, device,
                          index, all_pids, all_embs, text_embs, cat_list, cat_to_idx,
                          metadata, args)

    output_path = os.path.join(BASE_DIR, args.output)
    rows = []
    for bid, top_15 in results:
        for pid in top_15:
            rows.append({"bundle_asset_id": bid, "product_asset_id": pid})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path} ({len(df)} rows, {df['bundle_asset_id'].nunique()} bundles)")


# ═════════════════════════════════════════════════════════════════
# LOCAL EVAL
# ═════════════════════════════════════════════════════════════════

def run_local_eval(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model, processor, device, index, all_pids, all_embs, text_embs, cat_list, cat_to_idx, metadata = \
        load_everything(args, BASE_DIR)

    train_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    df_train = pd.read_csv(train_csv)
    train_gt = defaultdict(list)
    for _, row in df_train.iterrows():
        train_gt[row['bundle_asset_id']].append(row['product_asset_id'])

    eval_ids = list(train_gt.keys())
    if args.max_eval:
        eval_ids = eval_ids[:args.max_eval]
    bundles_dir = args.bundles_dir or os.path.join(BASE_DIR, "data", "images", "bundles")

    print(f"\n{'═'*60}")
    print(f"🎯 V3 Local Eval: {len(eval_ids)} TRAIN bundles")
    print(f"   img={args.w_img} text={args.w_text} sec={args.w_section} url={args.w_url} K={args.faiss_k}")
    print(f"{'═'*60}\n")

    results = run_pipeline(eval_ids, bundles_dir, model, processor, device,
                          index, all_pids, all_embs, text_embs, cat_list, cat_to_idx,
                          metadata, args)

    # Compute MRR@15
    mrr_sum = 0.0
    hit_at = {1: 0, 5: 0, 10: 0, 15: 0}
    total = 0

    for bid, top_15 in results:
        for true_pid in train_gt.get(bid, []):
            total += 1
            if true_pid in top_15:
                rank = top_15.index(true_pid) + 1
                mrr_sum += 1.0 / rank
                for k in hit_at:
                    if rank <= k:
                        hit_at[k] += 1

    if total > 0:
        mrr = mrr_sum / total
        print(f"\n{'═'*60}")
        print(f"🏆 RESULTS (V3)")
        print(f"{'═'*60}")
        print(f"  MRR@15:  {mrr:.4f} ({mrr*100:.2f}%)")
        for k in sorted(hit_at):
            print(f"  Hit@{k:2d}:  {hit_at[k]/total:.4f} ({hit_at[k]}/{total})")
        print(f"  Total pairs: {total}")
        return mrr


# ═════════════════════════════════════════════════════════════════
# SWEEP
# ═════════════════════════════════════════════════════════════════

def run_sweep(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model, processor, device, index, all_pids, all_embs, text_embs, cat_list, cat_to_idx, metadata = \
        load_everything(args, BASE_DIR)

    train_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    df_train = pd.read_csv(train_csv)
    train_gt = defaultdict(list)
    for _, row in df_train.iterrows():
        train_gt[row['bundle_asset_id']].append(row['product_asset_id'])

    eval_ids = list(train_gt.keys())[:args.max_eval] if args.max_eval else list(train_gt.keys())
    bundles_dir = args.bundles_dir or os.path.join(BASE_DIR, "data", "images", "bundles")

    configs = [
        {"w_img": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 0.0, "faiss_k": 100, "label": "CLIP img-only"},
        {"w_img": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 1.0, "faiss_k": 100, "label": "img+url(1.0)"},
        {"w_img": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 2.0, "faiss_k": 100, "label": "img+url(2.0)"},
        {"w_img": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 3.0, "faiss_k": 100, "label": "img+url(3.0)"},
        {"w_img": 1.0, "w_text": 0.3, "w_section": 0.0, "w_url": 2.0, "faiss_k": 100, "label": "img+text+url"},
        {"w_img": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 2.0, "faiss_k": 100, "label": "all signals"},
        {"w_img": 1.0, "w_text": 0.5, "w_section": 0.3, "w_url": 3.0, "faiss_k": 200, "label": "all+high K200"},
        {"w_img": 0.5, "w_text": 0.5, "w_section": 0.3, "w_url": 3.0, "faiss_k": 200, "label": "balanced"},
        {"w_img": 1.0, "w_text": 0.0, "w_section": 0.0, "w_url": 5.0, "faiss_k": 200, "label": "ultra-url(5.0)"},
        {"w_img": 1.0, "w_text": 0.3, "w_section": 0.2, "w_url": 5.0, "faiss_k": 200, "label": "all+ultra-url"},
    ]

    print(f"\n🔬 V3 Sweep: {len(configs)} configs on {len(eval_ids)} bundles\n")
    best_mrr = 0
    best_cfg = None

    for cfg in configs:
        args.w_img = cfg["w_img"]
        args.w_text = cfg["w_text"]
        args.w_section = cfg["w_section"]
        args.w_url = cfg["w_url"]
        args.faiss_k = cfg["faiss_k"]

        results = run_pipeline(eval_ids, bundles_dir, model, processor, device,
                              index, all_pids, all_embs, text_embs, cat_list, cat_to_idx,
                              metadata, args)

        mrr_sum = 0.0
        total = 0
        for bid, top_15 in results:
            for true_pid in train_gt.get(bid, []):
                total += 1
                if true_pid in top_15:
                    rank = top_15.index(true_pid) + 1
                    mrr_sum += 1.0 / rank

        mrr = mrr_sum / total if total > 0 else 0
        marker = " ★ BEST" if mrr > best_mrr else ""
        print(f"  {cfg['label']:30s} → MRR@15 = {mrr:.4f} ({mrr*100:.2f}%){marker}")
        
        if mrr > best_mrr:
            best_mrr = mrr
            best_cfg = cfg.copy()

    print(f"\n🏆 Best: {best_cfg}")
    print(f"   MRR@15 = {best_mrr:.4f} ({best_mrr*100:.2f}%)")
    return best_cfg


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Inference V3")
    parser.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    parser.add_argument("--custom_weights", type=str, default=None)
    parser.add_argument("--output", type=str, default="smart_v3_submission.csv")
    
    parser.add_argument("--w_img", type=float, default=1.0)
    parser.add_argument("--w_text", type=float, default=0.3)
    parser.add_argument("--w_section", type=float, default=0.2)
    parser.add_argument("--w_url", type=float, default=2.0)
    parser.add_argument("--faiss_k", type=int, default=200)
    
    parser.add_argument("--local_eval", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--max_eval", type=int, default=None)
    parser.add_argument("--bundles_dir", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.local_eval:
        run_local_eval(args)
    else:
        run_test(args)

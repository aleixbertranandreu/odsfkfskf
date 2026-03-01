"""
smart_inference.py — Smart Pipeline v2: Soft Category Boosting + Section Boost + URL Re-ranking
════════════════════════════════════════════════════════════════════════════════════════════════

KEY INSIGHT: Hard category filtering hurts (10→13% MRR) by excluding correct products.
Instead, this pipeline does a FULL FAISS search and then RE-RANKS with soft boosts:
  1. Full CLIP image-to-image FAISS search (top 100 candidates)
  2. Section compatibility boost (+50% for section-matching products)
  3. Category similarity boost (CLIP text-to-image)
  4. URL code re-ranking (23.7% of train matches share Zara URL codes)
  5. Output top 15
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
# SECTION ANALYSIS: Learn section → category distributions
# ═════════════════════════════════════════════════════════════════

def build_metadata(base_dir):
    """Load all metadata and build lookup tables."""
    bundles_csv = os.path.join(base_dir, "data", "raw", "bundles_dataset.csv")
    products_csv = os.path.join(base_dir, "data", "raw", "product_dataset.csv")
    train_csv = os.path.join(base_dir, "data", "raw", "bundles_product_match_train.csv")

    df_bundles = pd.read_csv(bundles_csv)
    bundle_section = dict(zip(df_bundles['bundle_asset_id'], df_bundles['bundle_id_section'].astype(str)))
    bundle_urls = dict(zip(df_bundles['bundle_asset_id'], df_bundles['bundle_image_url']))

    df_products = pd.read_csv(products_csv)
    product_desc = dict(zip(df_products['product_asset_id'], df_products['product_description']))
    product_url = dict(zip(df_products['product_asset_id'], df_products['product_image_url']))
    all_product_ids_list = df_products['product_asset_id'].tolist()

    # Learn which categories are MORE likely in each section
    df_train = pd.read_csv(train_csv)
    section_cat_counts = defaultdict(Counter)
    for _, row in df_train.iterrows():
        bid = row['bundle_asset_id']
        pid = row['product_asset_id']
        if bid in bundle_section and pid in product_desc:
            sec = bundle_section[bid]
            cat = product_desc[pid]
            section_cat_counts[sec][cat] += 1

    # Build section_cat_score: P(category | section) normalized
    section_cat_score = {}
    for sec, cat_counts in section_cat_counts.items():
        total = sum(cat_counts.values())
        section_cat_score[sec] = {cat: cnt / total for cat, cnt in cat_counts.items()}

    # Categories that NEVER appear in a section (strong negative signal)
    all_cats = set(product_desc.values())
    section_forbidden = {}
    for sec in section_cat_counts:
        appeared = set(section_cat_counts[sec].keys())
        section_forbidden[sec] = all_cats - appeared

    print("  Section category coverage:")
    for sec in sorted(section_cat_counts):
        n_cats = len(section_cat_counts[sec])
        n_forbidden = len(section_forbidden.get(sec, set()))
        print(f"    Section {sec}: {n_cats} categories appear, {n_forbidden} never appear")

    return {
        "bundle_section": bundle_section,
        "bundle_urls": bundle_urls,
        "product_desc": product_desc,
        "product_url": product_url,
        "all_product_ids": all_product_ids_list,
        "section_cat_score": section_cat_score,
        "section_forbidden": section_forbidden,
    }


# ═════════════════════════════════════════════════════════════════
# TEXT-TO-IMAGE CATEGORY DETECTION
# ═════════════════════════════════════════════════════════════════

def build_text_embeddings(model, processor, device, categories):
    """Encode all product categories as CLIP text embeddings."""
    cat_list = sorted(categories)
    prompts = [f"a photo of {cat.lower()}" for cat in cat_list]

    all_text_features = []
    batch_size = 32
    
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        tokenized = processor.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            text_out = model.text_model(**tokenized)
            text_features = model.text_projection(text_out.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        all_text_features.append(text_features.cpu())
    
    text_embeddings = torch.cat(all_text_features, dim=0).numpy().astype(np.float32)
    return cat_list, text_embeddings


def detect_categories_in_bundle(bundle_embedding, text_embeddings, cat_list):
    """Return category → similarity_score dict for a bundle image."""
    similarities = bundle_embedding @ text_embeddings.T
    cat_scores = {}
    for idx, cat in enumerate(cat_list):
        cat_scores[cat] = float(similarities[0][idx])
    return cat_scores


# ═════════════════════════════════════════════════════════════════
# URL CODE EXTRACTION
# ═════════════════════════════════════════════════════════════════

def extract_zara_code(url):
    """Extract the numeric product code from a Zara URL."""
    try:
        filename = url.split('/')[-2]
        match = re.search(r'(\d{8,11})', filename)
        if match:
            return match.group(1).zfill(11)
        return None
    except:
        return None


# ═════════════════════════════════════════════════════════════════
# CORE SCORING: Re-rank FAISS candidates with all signals
# ═════════════════════════════════════════════════════════════════

def rerank_candidates(candidate_pids, candidate_scores, bundle_cat_scores, 
                      section, metadata, b_zara_code,
                      w_cat=0.3, w_section=0.3, w_url=1.0):
    """
    Re-rank FAISS candidates using:
    1. Original CLIP similarity score (normalized)
    2. Category match boost: how well does the product's category match what CLIP sees in the bundle?
    3. Section compatibility: is this product category typical for this section?
    4. URL code: strong boost for matching Zara codes
    """
    product_desc = metadata["product_desc"]
    section_cat_score = metadata["section_cat_score"]
    section_forbidden = metadata["section_forbidden"]
    product_zara_codes = metadata.get("product_zara_codes", {})

    ranked = []
    for pid, clip_score in zip(candidate_pids, candidate_scores):
        score = float(clip_score)
        cat = product_desc.get(pid, "UNKNOWN")

        # Category boost: how similar is this product's category to what CLIP detects in the bundle?
        cat_sim = bundle_cat_scores.get(cat, 0.0)
        # Normalize to [0, 1] range (CLIP text-image similarities are typically 0.1-0.4)
        cat_boost = max(0, (cat_sim - 0.1) / 0.3)  # maps 0.1→0.0, 0.4→1.0
        
        # Section compatibility
        sec_score = section_cat_score.get(section, {}).get(cat, 0.0)
        forbidden = section_forbidden.get(section, set())
        if cat in forbidden:
            sec_score = -0.5  # Penalty for impossible categories
        
        # URL code boost
        url_boost = 0.0
        if b_zara_code:
            p_code = product_zara_codes.get(pid)
            if p_code:
                if b_zara_code == p_code:
                    url_boost = 2.0   # Very strong: same product
                elif b_zara_code[:4] == p_code[:4]:
                    url_boost = 1.0   # Same Zara family
                elif b_zara_code[:3] == p_code[:3]:
                    url_boost = 0.3   # Similar department
        
        # Final score
        final_score = score + (w_cat * cat_boost) + (w_section * sec_score) + (w_url * url_boost)
        ranked.append((final_score, pid))

    # Also inject ALL products with exact URL match (they might not be in FAISS top-K)
    if b_zara_code:
        existing_pids = set(pid for _, pid in ranked)
        all_pids = metadata["all_product_ids"]
        for pid in all_pids:
            if pid in existing_pids:
                continue
            p_code = product_zara_codes.get(pid)
            if p_code and b_zara_code == p_code:
                ranked.append((10.0, pid))  # Force exact URL matches to top

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ═════════════════════════════════════════════════════════════════
# MULTI-CROP ENCODING
# ═════════════════════════════════════════════════════════════════

def encode_bundle_multicrop(image, model, processor, device):
    """Encode bundle with multiple crops for robust matching."""
    w, h = image.size
    
    crops = [
        image,                                           # Full
        image.crop((0, 0, w, h // 2)),                   # Top half
        image.crop((0, h // 2, w, h)),                   # Bottom half
        image.crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))),  # Center 80%
    ]
    
    inputs = processor(images=crops, return_tensors="pt", padding=True)
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        features = model.visual_projection(vision_out.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
    
    avg = features.mean(dim=0)
    avg = avg / avg.norm()
    
    return avg.cpu().numpy().astype(np.float32).reshape(1, -1)


# ═════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════

def run_pipeline(bundle_ids, bundle_images_dir, model, processor, device, 
                 index, all_product_ids, all_embeddings, text_embeddings, cat_list,
                 metadata, args):
    """Core pipeline used for both inference and evaluation."""
    
    pid_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
    
    # Precompute URL codes
    product_zara_codes = {}
    for pid, url in metadata["product_url"].items():
        product_zara_codes[pid] = extract_zara_code(url)
    metadata["product_zara_codes"] = product_zara_codes
    
    bundle_zara_codes = {}
    for bid, url in metadata["bundle_urls"].items():
        bundle_zara_codes[bid] = extract_zara_code(url)

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

            # Encode bundle
            if args.multicrop:
                bundle_emb = encode_bundle_multicrop(image, model, processor, device)
            else:
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device)
                with torch.no_grad():
                    vision_out = model.vision_model(pixel_values=pixel_values)
                    bundle_emb = model.visual_projection(vision_out.pooler_output)
                    bundle_emb = bundle_emb / bundle_emb.norm(dim=-1, keepdim=True)
                bundle_emb = bundle_emb.cpu().numpy().astype(np.float32)

            # FAISS search (full index, top K=200 candidates)
            query = bundle_emb.copy()
            faiss.normalize_L2(query)
            k_search = min(args.faiss_k, index.ntotal)
            scores, indices = index.search(query, k_search)
            
            candidate_pids = [all_product_ids[idx] for idx in indices[0] if idx >= 0]
            candidate_scores = scores[0][:len(candidate_pids)]

            # Detect categories in bundle
            bundle_cat_scores = detect_categories_in_bundle(
                query, text_embeddings, cat_list
            )

            # Get section
            section = str(metadata["bundle_section"].get(bundle_id, "1"))
            b_zara_code = bundle_zara_codes.get(bundle_id)

            # Re-rank!
            ranked = rerank_candidates(
                candidate_pids, candidate_scores, bundle_cat_scores,
                section, metadata, b_zara_code,
                w_cat=args.w_cat, w_section=args.w_section, w_url=args.w_url,
            )

            top_15 = [pid for _, pid in ranked[:15]]

            # Pad if needed
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
            results.append((bundle_id, all_product_ids[:15]))

        if (i + 1) % 50 == 0 or (i + 1) == len(bundle_ids):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (i + 1) / len(bundle_ids)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}% | {i+1}/{len(bundle_ids)} | {rate:.1f} b/s")

    total_time = time.time() - start_time
    print(f"\n✅ Done! {len(bundle_ids)} bundles in {total_time:.1f}s | {errors} errors")

    return results


def load_model_and_index(args, base_dir):
    """Load CLIP model, FAISS index, and text embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # Load metadata
    print("\n📋 Loading metadata...")
    metadata = build_metadata(base_dir)

    # Load CLIP
    print(f"\n🧠 Loading FashionCLIP ({args.model_name})...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    if args.custom_weights:
        checkpoint = torch.load(args.custom_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✅ Custom weights loaded!")
    
    model.eval()

    # Text embeddings
    print("\n📝 Building text embeddings...")
    unique_cats = sorted(set(metadata["product_desc"].values()))
    print(f"  {len(unique_cats)} unique categories")
    cat_list, text_embs = build_text_embeddings(model, processor, device, unique_cats)

    # FAISS index
    print("\n📦 Loading FAISS index...")
    embeddings_dir = os.path.join(base_dir, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    faiss_path = os.path.join(embeddings_dir, f"clip_faiss_{suffix}.bin")
    ids_path = os.path.join(embeddings_dir, f"clip_ids_{suffix}.pkl")

    index = faiss.read_index(faiss_path)
    with open(ids_path, 'rb') as f:
        all_product_ids = pickle.load(f)
    
    all_embeddings = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d).copy()
    print(f"  Index: {index.ntotal} products × {index.d} dims")

    return model, processor, device, index, all_product_ids, all_embeddings, text_embs, cat_list, metadata


# ═════════════════════════════════════════════════════════════════
# TEST INFERENCE
# ═════════════════════════════════════════════════════════════════

def run_test_inference(args):
    """Generate submission for Kaggle test bundles."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model, processor, device, index, all_product_ids, all_embeddings, text_embs, cat_list, metadata = \
        load_model_and_index(args, BASE_DIR)

    test_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv")
    df_test = pd.read_csv(test_csv)
    test_bundle_ids = df_test['bundle_asset_id'].tolist()

    bundle_images_dir = args.bundles_dir if args.bundles_dir else os.path.join(BASE_DIR, "data", "images", "bundles")

    print(f"\n{'═'*60}")
    print(f"🎯 Processing {len(test_bundle_ids)} TEST bundles")
    print(f"   Weights: cat={args.w_cat}, section={args.w_section}, url={args.w_url}")
    print(f"   FAISS K={args.faiss_k}, Multicrop={'ON' if args.multicrop else 'OFF'}")
    print(f"{'═'*60}\n")

    results = run_pipeline(
        test_bundle_ids, bundle_images_dir, model, processor, device,
        index, all_product_ids, all_embeddings, text_embs, cat_list, metadata, args
    )

    # Save CSV
    output_path = os.path.join(BASE_DIR, args.output)
    rows = []
    for bundle_id, top_15 in results:
        for pid in top_15:
            rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})

    df_sub = pd.DataFrame(rows)
    df_sub.to_csv(output_path, index=False)

    n_bundles = df_sub['bundle_asset_id'].nunique()
    print(f"\n💾 Saved: {output_path}")
    print(f"   {len(df_sub)} rows | {n_bundles} bundles | {len(df_sub)//n_bundles} products/bundle")

    return results


# ═════════════════════════════════════════════════════════════════
# LOCAL EVALUATION
# ═════════════════════════════════════════════════════════════════

def run_local_eval(args):
    """Run pipeline on TRAIN bundles for local MRR@15 evaluation."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model, processor, device, index, all_product_ids, all_embeddings, text_embs, cat_list, metadata = \
        load_model_and_index(args, BASE_DIR)

    # Load train ground truth
    train_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    df_train = pd.read_csv(train_csv)
    train_gt = defaultdict(list)
    for _, row in df_train.iterrows():
        train_gt[row['bundle_asset_id']].append(row['product_asset_id'])

    train_bundle_ids = list(train_gt.keys())
    bundle_images_dir = args.bundles_dir if args.bundles_dir else os.path.join(BASE_DIR, "data", "images", "bundles")
    
    eval_bundles = train_bundle_ids[:args.max_eval] if args.max_eval else train_bundle_ids
    
    print(f"\n{'═'*60}")
    print(f"🎯 Evaluating on {len(eval_bundles)} TRAIN bundles")
    print(f"   Weights: cat={args.w_cat}, section={args.w_section}, url={args.w_url}")
    print(f"   FAISS K={args.faiss_k}, Multicrop={'ON' if args.multicrop else 'OFF'}")
    print(f"{'═'*60}\n")

    results = run_pipeline(
        eval_bundles, bundle_images_dir, model, processor, device,
        index, all_product_ids, all_embeddings, text_embs, cat_list, metadata, args
    )

    # Evaluate MRR@15
    mrr_sum = 0.0
    hit_at = {1: 0, 5: 0, 10: 0, 15: 0}
    total_pairs = 0

    for bundle_id, top_15 in results:
        for true_pid in train_gt.get(bundle_id, []):
            total_pairs += 1
            if true_pid in top_15:
                rank = top_15.index(true_pid) + 1
                mrr_sum += 1.0 / rank
                for k in hit_at:
                    if rank <= k:
                        hit_at[k] += 1

    if total_pairs > 0:
        mrr = mrr_sum / total_pairs
        print(f"\n{'═'*60}")
        print(f"🏆 RESULTS")
        print(f"{'═'*60}")
        print(f"  MRR@15:  {mrr:.4f} ({mrr*100:.2f}%)")
        for k in sorted(hit_at):
            print(f"  Hit@{k:2d}:  {hit_at[k]/total_pairs:.4f} ({hit_at[k]}/{total_pairs})")
        print(f"  Total pairs: {total_pairs}")
    else:
        print("  No pairs to evaluate!")


# ═════════════════════════════════════════════════════════════════
# HYPERPARAMETER SWEEP
# ═════════════════════════════════════════════════════════════════

def run_sweep(args):
    """Sweep hyperparameters on train data to find best config."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model, processor, device, index, all_product_ids, all_embeddings, text_embs, cat_list, metadata = \
        load_model_and_index(args, BASE_DIR)

    # Load train ground truth
    train_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    df_train = pd.read_csv(train_csv)
    train_gt = defaultdict(list)
    for _, row in df_train.iterrows():
        train_gt[row['bundle_asset_id']].append(row['product_asset_id'])

    train_bundle_ids = list(train_gt.keys())
    bundle_images_dir = args.bundles_dir if args.bundles_dir else os.path.join(BASE_DIR, "data", "images", "bundles")
    eval_bundles = train_bundle_ids[:args.max_eval] if args.max_eval else train_bundle_ids

    # Parameter grid
    configs = [
        {"w_cat": 0.0, "w_section": 0.0, "w_url": 0.0, "multicrop": False, "faiss_k": 100, "label": "Baseline (CLIP only)"},
        {"w_cat": 0.0, "w_section": 0.0, "w_url": 0.0, "multicrop": True, "faiss_k": 100, "label": "Multicrop only"},
        {"w_cat": 0.3, "w_section": 0.0, "w_url": 0.0, "multicrop": False, "faiss_k": 100, "label": "+Category"},
        {"w_cat": 0.0, "w_section": 0.3, "w_url": 0.0, "multicrop": False, "faiss_k": 100, "label": "+Section"},
        {"w_cat": 0.0, "w_section": 0.0, "w_url": 1.0, "multicrop": False, "faiss_k": 100, "label": "+URL"},
        {"w_cat": 0.3, "w_section": 0.3, "w_url": 1.0, "multicrop": False, "faiss_k": 100, "label": "All signals"},
        {"w_cat": 0.3, "w_section": 0.3, "w_url": 1.0, "multicrop": True, "faiss_k": 100, "label": "All + Multicrop"},
        {"w_cat": 0.3, "w_section": 0.3, "w_url": 1.0, "multicrop": True, "faiss_k": 200, "label": "All + MC + K200"},
        {"w_cat": 0.5, "w_section": 0.5, "w_url": 2.0, "multicrop": True, "faiss_k": 200, "label": "High boost"},
        {"w_cat": 0.1, "w_section": 0.1, "w_url": 0.5, "multicrop": True, "faiss_k": 200, "label": "Low boost"},
    ]

    print(f"\n🔬 Sweeping {len(configs)} configs on {len(eval_bundles)} bundles...\n")
    best_mrr = 0
    best_config = None

    for cfg in configs:
        args.w_cat = cfg["w_cat"]
        args.w_section = cfg["w_section"]
        args.w_url = cfg["w_url"]
        args.multicrop = cfg["multicrop"]
        args.faiss_k = cfg["faiss_k"]

        results = run_pipeline(
            eval_bundles, bundle_images_dir, model, processor, device,
            index, all_product_ids, all_embeddings, text_embs, cat_list, metadata, args
        )

        mrr_sum = 0.0
        total = 0
        for bundle_id, top_15 in results:
            for true_pid in train_gt.get(bundle_id, []):
                total += 1
                if true_pid in top_15:
                    rank = top_15.index(true_pid) + 1
                    mrr_sum += 1.0 / rank

        mrr = mrr_sum / total if total > 0 else 0
        marker = " ★ BEST" if mrr > best_mrr else ""
        print(f"  {cfg['label']:30s} → MRR@15 = {mrr:.4f} ({mrr*100:.2f}%){marker}")
        
        if mrr > best_mrr:
            best_mrr = mrr
            best_config = cfg.copy()

    print(f"\n🏆 Best config: {best_config}")
    print(f"   MRR@15 = {best_mrr:.4f}")
    return best_config


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Inference Pipeline v2")
    parser.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    parser.add_argument("--custom_weights", type=str, default=None)
    parser.add_argument("--output", type=str, default="smart_submission.csv")
    
    # Pipeline params
    parser.add_argument("--w_cat", type=float, default=0.3, help="Category boost weight")
    parser.add_argument("--w_section", type=float, default=0.3, help="Section boost weight")
    parser.add_argument("--w_url", type=float, default=1.0, help="URL code boost weight")
    parser.add_argument("--faiss_k", type=int, default=200, help="FAISS top-K candidates")
    parser.add_argument("--multicrop", action="store_true", help="Use multi-crop encoding")
    
    # Mode
    parser.add_argument("--local_eval", action="store_true", help="Evaluate on train bundles")
    parser.add_argument("--sweep", action="store_true", help="Sweep hyperparameters")
    parser.add_argument("--max_eval", type=int, default=None, help="Max bundles for eval")
    parser.add_argument("--bundles_dir", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        run_sweep(args)
    elif args.local_eval:
        run_local_eval(args)
    else:
        run_test_inference(args)

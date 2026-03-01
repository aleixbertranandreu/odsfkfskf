"""
smart_inference_v6.py — V6 Pipeline using OpenCLIP ViT-L/14
════════════════════════════════════════════════════════════════════════════

Using the much larger ViT-L/14 from OpenCLIP.
Integrates the successful parts of V3 (URL matching, text-image scoring, 
section compatibility) but swaps out the visual backbone for the massive 
768-dim LAION-trained model.
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

import open_clip

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

# ═══════════════════════════════════════════════════════════════
# TEXT EMBEDDINGS (OPENCLIP)
# ═══════════════════════════════════════════════════════════════

def build_text_embeddings(model, tokenizer, device, categories):
    cat_list = sorted(categories)
    prompts = [f"a photo of {c.lower()}" for c in cat_list]
    all_feats = []
    
    for start in range(0, len(prompts), 32):
        batch = prompts[start:start + 32]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        all_feats.append(features.cpu())
        
    text_embs = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    return cat_list, text_embs, cat_to_idx

# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(bundle_ids, bundle_images_dir, model, preprocess, device,
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

            img = Image.open(bundle_path).convert("RGB")
            pixel_values = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(pixel_values)
                features = features / features.norm(dim=-1, keepdim=True)
            
            query = features.cpu().numpy().astype(np.float32)

            k_search = min(args.faiss_k, index.ntotal)
            scores, idxs = index.search(query, k_search)
            
            global_results = {}
            for score, j in zip(scores[0], idxs[0]):
                if j >= 0:
                    global_results[all_product_ids[j]] = float(score)

            text_sims = (query @ text_embs.T)[0]

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

            all_candidates = set(global_results.keys()) | set(url_scores.keys())
            section = str(bundle_section.get(bundle_id, "1"))
            sec_probs = section_cat_prob.get(section, {})

            scored = []
            for pid in all_candidates:
                img_score = global_results.get(pid, 0.0)
                url_score = url_scores.get(pid, 0.0)
                
                cat = product_desc.get(pid, "")
                idx_c = cat_to_idx.get(cat)
                text_score = float(text_sims[idx_c]) if idx_c is not None else 0.0
                sec_score = sec_probs.get(cat, 0.0)

                if img_score == 0.0 and pid in pid_to_idx:
                    p_emb = all_embeddings[pid_to_idx[pid]:pid_to_idx[pid]+1].copy()
                    faiss.normalize_L2(p_emb)
                    img_score = float((query @ p_emb.T).item())

                final = (args.w_img * img_score +
                         args.w_text * text_score +
                         args.w_section * sec_score +
                         args.w_url * url_score)
                
                scored.append((final, pid))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_15 = [pid for _, pid in scored[:15]]

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

    metadata = build_metadata(base_dir)

    print(f"🧠 Loading {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    unique_cats = sorted(set(metadata["product_desc"].values()))
    cat_list, text_embs, cat_to_idx = build_text_embeddings(model, tokenizer, device, unique_cats)

    emb_dir = os.path.join(base_dir, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    
    idx_path = os.path.join(emb_dir, f"openclip_faiss_{suffix}.bin")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing OpenCLIP index at {idx_path}. Run build_openclip_index.py first.")
        
    idx = faiss.read_index(idx_path)
    with open(os.path.join(emb_dir, f"openclip_ids_{suffix}.pkl"), 'rb') as f:
        all_pids = pickle.load(f)
    all_embs = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * idx.d).reshape(idx.ntotal, idx.d).copy()

    return model, preprocess, device, idx, all_pids, all_embs, text_embs, cat_list, cat_to_idx, metadata

def run_test(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, meta = load_everything(args, BASE)
    test_ids = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_test.csv"))['bundle_asset_id'].tolist()
    bdir = args.bundles_dir or os.path.join(BASE, "data/images/bundles")

    print(f"\n🎯 V6: {len(test_ids)} TEST bundles")
    results = run_pipeline(test_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, meta, args)

    out = os.path.join(BASE, args.output)
    rows = [{"bundle_asset_id": b, "product_asset_id": p} for b, t in results for p in t]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"💾 {out}")

def run_eval(args):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, proc, dev, idx, pids, embs, te, cl, ci, meta = load_everything(args, BASE)
    df = pd.read_csv(os.path.join(BASE, "data/raw/bundles_product_match_train.csv"))
    gt = defaultdict(list)
    for _, r in df.iterrows():
        gt[r['bundle_asset_id']].append(r['product_asset_id'])

    eval_ids = list(gt.keys())[:args.max_eval] if args.max_eval else list(gt.keys())
    bdir = args.bundles_dir or os.path.join(BASE, "data/images/bundles")

    print(f"\n🎯 V6 Eval: {len(eval_ids)} bundles")
    results = run_pipeline(eval_ids, bdir, model, proc, dev, idx, pids, embs, te, cl, ci, meta, args)

    mrr_sum = 0.0
    total = 0
    for bid, top15 in results:
        for tp in gt.get(bid, []):
            total += 1
            if tp in top15:
                mrr_sum += 1.0 / (top15.index(tp) + 1)
    if total > 0:
        print(f"\n🏆 V6: MRR@15={mrr_sum/total:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="ViT-L-14")
    p.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--output", default="smart_v6_submission.csv")
    
    p.add_argument("--w_img", type=float, default=1.0)
    p.add_argument("--w_text", type=float, default=0.3)
    p.add_argument("--w_section", type=float, default=0.2)
    p.add_argument("--w_url", type=float, default=2.0)
    p.add_argument("--faiss_k", type=int, default=200)
    
    p.add_argument("--local_eval", action="store_true")
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--bundles_dir", default=None)
    
    args = p.parse_args()
    if args.local_eval:
        run_eval(args)
    else:
        run_test(args)

"""
clip_inference.py — CLIP-Based Visual Search (No YOLO, No Cropping)
════════════════════════════════════════════════════════════════════

KEY INSIGHT: The top teams (52% MRR) are NOT using object detection.
They match the FULL bundle image against the FULL catalog using CLIP.

This catches EVERYTHING visible: shoes, bags, belts, jewelry, hats —
which represent 44.7% of correct matches that YOLO-based approaches miss.

Strategy:
1. Encode full bundle image with CLIP ViT-L/14
2. Multi-crop: also encode 5 sub-regions (top, mid, bottom, left, right)
3. Average all embeddings → stable representation
4. FAISS search → Top-15 per bundle
5. Optional: ensemble with ConvNeXt metric model scores
"""
import argparse
import glob
import os
import pickle
import sys
import time
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ═════════════════════════════════════════════════════════════════
# Auto-discover index files
# ═════════════════════════════════════════════════════════════════

def find_clip_index(embeddings_dir: str, model_suffix: str = None):
    """Find the CLIP FAISS index and product IDs."""
    if model_suffix:
        faiss_path = os.path.join(embeddings_dir, f"clip_faiss_{model_suffix}.bin")
        ids_path = os.path.join(embeddings_dir, f"clip_ids_{model_suffix}.pkl")
        if os.path.exists(faiss_path):
            return faiss_path, ids_path

    # Auto-discover
    faiss_files = glob.glob(os.path.join(embeddings_dir, "clip_faiss_*.bin"))
    if faiss_files:
        faiss_path = sorted(faiss_files, key=os.path.getmtime, reverse=True)[0]
        ids_path = faiss_path.replace("clip_faiss_", "clip_ids_").replace(".bin", ".pkl")
        return faiss_path, ids_path

    return None, None


# ═════════════════════════════════════════════════════════════════
# Multi-Crop CLIP Encoding
# ═════════════════════════════════════════════════════════════════

def encode_bundle_multicrop(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
) -> np.ndarray:
    """
    Encode a bundle image using multiple crops for robust matching.
    
    Crops:
    - Full image (captures everything)
    - Top half (upper body garments)
    - Bottom half (lower body, shoes)
    - Center 80% (removes edge noise)
    - Left half (sometimes different items on each side)
    - Right half
    
    Returns: averaged L2-normalized embedding
    """
    w, h = image.size
    
    crops = [
        image,                                           # Full
        image.crop((0, 0, w, h // 2)),                   # Top half
        image.crop((0, h // 2, w, h)),                   # Bottom half
        image.crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))),  # Center 80%
        image.crop((0, 0, w // 2, h)),                   # Left half
        image.crop((w // 2, 0, w, h)),                   # Right half
    ]
    
    # Process all crops at once
    inputs = processor(images=crops, return_tensors="pt", padding=True)
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            vision_out = model.vision_model(pixel_values=pixel_values)
            features = model.visual_projection(vision_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
    
    # Average all crop embeddings, re-normalize
    avg = features.mean(dim=0)
    avg = avg / avg.norm()
    
    return avg.cpu().numpy().astype(np.float32)


def encode_bundle_simple(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
) -> np.ndarray:
    """Simple single-image encoding (fastest)."""
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            vision_out = model.vision_model(pixel_values=pixel_values)
            features = model.visual_projection(vision_out.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
    
    return features.squeeze(0).cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════════
# Main Inference
# ═════════════════════════════════════════════════════════════════

def run_inference(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv")
    bundles_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_dataset.csv")
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    bundle_images_dir = args.bundles_dir if args.bundles_dir else os.path.join(BASE_DIR, "data", "images", "bundles")
    embeddings_dir = os.path.join(BASE_DIR, "data", "embeddings")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 CLIP Inference on: {device}")

    # ─── Load CLIP ───
    model_name = args.clip_model
    print(f"🧠 Loading base model {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    if getattr(args, 'custom_weights', None):
        print(f"🔄 Loading CUSTOM fine-tuned weights from {args.custom_weights}...")
        checkpoint = torch.load(args.custom_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Custom weights loaded successfully!")
        
    model.eval()
    print(f"   ✅ CLIP loaded")

    # ─── Load FAISS Index ───
    suffix = model_name.replace("/", "_").replace("-", "_")
    faiss_path, ids_path = find_clip_index(embeddings_dir, suffix)

    if not faiss_path or not os.path.exists(faiss_path):
        print(f"❌ No CLIP FAISS index found. Run build_clip_index.py first!")
        sys.exit(1)

    print(f"📦 Loading index: {os.path.basename(faiss_path)}")
    index = faiss.read_index(faiss_path)
    with open(ids_path, 'rb') as f:
        product_ids = pickle.load(f)
    print(f"   Index: {index.ntotal} products × {index.d} dims")

    # ─── Load product categories (for optional ensemble later) ───
    df_products = pd.read_csv(products_csv)
    product_categories = dict(
        zip(df_products['product_asset_id'].astype(str), df_products['product_description'])
    )

    # ─── Optional: Load ConvNeXt FAISS for ensemble ───
    ensemble_index = None
    ensemble_ids = None
    convnext_faiss = os.path.join(embeddings_dir, "faiss_index.bin")
    convnext_ids = os.path.join(embeddings_dir, "product_ids.pkl")
    if args.ensemble and os.path.exists(convnext_faiss):
        print(f"🔗 Loading ConvNeXt index for ensemble...")
        ensemble_index = faiss.read_index(convnext_faiss)
        with open(convnext_ids, 'rb') as f:
            ensemble_ids = pickle.load(f)
        print(f"   ConvNeXt index: {ensemble_index.ntotal} × {ensemble_index.d}")

    # ConvNeXt model for ensemble
    convnext_model = None
    if args.ensemble:
        try:
            from model import FashionEmbedder
            from final_inference import auto_discover_checkpoint, get_base_transform
            ckpt = auto_discover_checkpoint(BASE_DIR)
            if ckpt:
                chk = torch.load(ckpt, map_location=device, weights_only=False)
                convnext_model = FashionEmbedder(embed_dim=chk.get('embed_dim', 256), pretrained=False).to(device)
                convnext_model.load_state_dict(chk['model_state_dict'])
                convnext_model.eval()
                print(f"   ✅ ConvNeXt model loaded for ensemble")
        except Exception as e:
            print(f"   ⚠️ ConvNeXt not available: {e}")

    # ─── Process Test Bundles ───
    df_test = pd.read_csv(test_csv)
    bundle_ids = df_test['bundle_asset_id'].tolist()

    mode = "multi-crop" if args.multicrop else "single"
    print(f"\n{'═'*60}")
    print(f"🎯 Processing {len(bundle_ids)} test bundles")
    print(f"   Mode: {mode}")
    print(f"   Ensemble: {'ON' if args.ensemble and ensemble_index else 'OFF'}")
    print(f"{'═'*60}\n")

    results = []
    errors = 0
    start_time = time.time()

    for i, bundle_id in enumerate(bundle_ids):
        try:
            bundle_path = os.path.join(bundle_images_dir, f"{bundle_id}.jpg")

            if not os.path.exists(bundle_path):
                results.append((bundle_id, product_ids[:15]))
                continue

            image = Image.open(bundle_path).convert("RGB")

            # ─── CLIP Encoding ───
            if args.multicrop:
                clip_emb = encode_bundle_multicrop(image, model, processor, device)
            else:
                clip_emb = encode_bundle_simple(image, model, processor, device)

            clip_emb = clip_emb.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(clip_emb)

            # ─── FAISS Search ───
            k = min(args.top_k, index.ntotal)
            clip_scores, clip_indices = index.search(clip_emb, k)

            candidate_ids = [product_ids[idx] for idx in clip_indices[0] if idx >= 0]
            candidate_scores = clip_scores[0][:len(candidate_ids)]

            # ─── Optional: Ensemble with ConvNeXt ───
            if args.ensemble and ensemble_index and convnext_model:
                # Get ConvNeXt embedding for the bundle
                from torchvision import transforms
                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
                img_tensor = tf(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    cnx_emb = convnext_model(img_tensor)
                cnx_emb = cnx_emb.cpu().numpy().astype(np.float32)
                faiss.normalize_L2(cnx_emb)

                cnx_scores, cnx_indices = ensemble_index.search(cnx_emb, k)
                cnx_candidate_ids = [ensemble_ids[idx] for idx in cnx_indices[0] if idx >= 0]
                cnx_candidate_scores = cnx_scores[0][:len(cnx_candidate_ids)]

                # Merge: weighted combination
                merged = defaultdict(float)
                # CLIP gets 70% weight (handles accessories)
                for pid, score in zip(candidate_ids, candidate_scores):
                    merged[pid] += 0.7 * score
                # ConvNeXt gets 30% weight (fine-grained clothing)
                for pid, score in zip(cnx_candidate_ids, cnx_candidate_scores):
                    merged[pid] += 0.3 * score

                sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)
                top_15 = [pid for pid, _ in sorted_merged[:15]]
            else:
                # Pure CLIP — just take top 15
                top_15 = candidate_ids[:15]

            # Pad
            if len(top_15) < 15:
                seen = set(top_15)
                for pid in product_ids:
                    if pid not in seen:
                        top_15.append(pid)
                        seen.add(pid)
                    if len(top_15) >= 15:
                        break

            results.append((bundle_id, top_15))

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ❌ {bundle_id}: {e}")
            results.append((bundle_id, product_ids[:15]))

        if (i + 1) % 25 == 0 or (i + 1) == len(bundle_ids):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            pct = 100 * (i + 1) / len(bundle_ids)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:5.1f}% | {i+1}/{len(bundle_ids)} | {rate:.1f} b/s")

    total_time = time.time() - start_time
    print(f"\n✅ Done! {len(bundle_ids)} bundles in {total_time:.1f}s")
    if errors > 0:
        print(f"   ⚠️ {errors} errors")

    # ─── Generate CSV ───
    print("\n📝 Generating submission...")
    rows = []
    for bundle_id, top_15 in results:
        for pid in top_15:
            rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})

    df_sub = pd.DataFrame(rows)
    output_name = "submission_clip.csv"
    if args.ensemble:
        output_name = "submission_ensemble.csv"
    output_path = os.path.join(BASE_DIR, output_name)
    df_sub.to_csv(output_path, index=False)

    n_bundles = df_sub['bundle_asset_id'].nunique()
    print(f"💾 Saved: {output_path}")
    print(f"   {len(df_sub)} rows | {n_bundles} bundles | {len(df_sub)//n_bundles} products/bundle")

    # ─── Evaluate ───
    if args.evaluate:
        evaluate_mrr(results, BASE_DIR)


def evaluate_mrr(results, base_dir):
    """MRR@15 on training overlap."""
    train_csv = os.path.join(base_dir, "data", "raw", "bundles_product_match_train.csv")
    df_train = pd.read_csv(train_csv)
    result_dict = {str(bid): [str(p) for p in pids] for bid, pids in results}

    hit_at = {1: 0, 5: 0, 10: 0, 15: 0}
    mrr_sum = 0.0
    total = 0

    for _, row in df_train.iterrows():
        bid = str(row['bundle_asset_id'])
        true_pid = str(row['product_asset_id'])
        if bid not in result_dict:
            continue
        predicted = result_dict[bid]
        total += 1
        if true_pid in predicted:
            rank = predicted.index(true_pid) + 1
            mrr_sum += 1.0 / rank
            for k in hit_at:
                if rank <= k:
                    hit_at[k] += 1

    if total > 0:
        print(f"\n🏆 MRR@15:  {mrr_sum/total:.4f}")
        for k in sorted(hit_at):
            print(f"   Hit@{k}:  {hit_at[k]/total:.4f} ({hit_at[k]}/{total})")
    else:
        print("   No overlap for evaluation")


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP-based Visual Search")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--custom_weights", type=str, default=None, help="Path to fine-tuned .pt weights")
    parser.add_argument("--multicrop", action="store_true", default=True,
                        help="Use multi-crop encoding (6 views)")
    parser.add_argument("--no_multicrop", action="store_true",
                        help="Disable multi-crop")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble CLIP (70%) + ConvNeXt (30%)")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--bundles_dir", type=str, default=None, help="Custom folder for bundles")

    args = parser.parse_args()
    if args.no_multicrop:
        args.multicrop = False
    return args


if __name__ == "__main__":
    run_inference(parse_args())

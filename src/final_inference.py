"""
final_inference.py — End-to-End Visual Search Pipeline
YOLOv8 Detection → ConvNeXt Metric Embeddings → FAISS Search → Re-Ranking → CSV Output

This is the production inference script that generates submission_metric.csv
"""
import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FashionEmbedder
from reranker import Reranker, YOLO_TO_CATEGORIES


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

YOLO_CLASSES_CLOTHING = {"top", "trousers", "skirt", "dress", "outwear"}

# Mapping from YOLO class to the allowed product categories
MAPEO_CATEGORIAS_YOLO = {
    "top": [
        'T-SHIRT', 'SHIRT', 'SWEATER', 'JACKET', 'COAT', 'BLOUSE', 'TOP',
        'WAISTCOAT', 'CARDIGAN', 'POLO SHIRT', 'BODYSUIT', 'SWEATSHIRT',
    ],
    "trousers": ['TROUSERS', 'JEANS', 'SHORTS', 'BERMUDA', 'LEGGINGS'],
    "skirt": ['SKIRT'],
    "dress": ['DRESS', 'JUMPSUIT', 'OVERALL'],
    "outwear": ['JACKET', 'COAT', 'BLAZER', 'TRENCH RAINCOAT', 'ANORAK'],
}


# ─────────────────────────────────────────────────────────────────
# Image transforms (must match training)
# ─────────────────────────────────────────────────────────────────

def get_inference_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────

def run_inference(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    test_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv")
    bundles_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_dataset.csv")
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    bundle_images_dir = os.path.join(BASE_DIR, "data", "images", "bundles")
    product_images_dir = os.path.join(BASE_DIR, "data", "images", "products")
    faiss_path = os.path.join(BASE_DIR, "data", "embeddings", "faiss_index.bin")
    ids_path = os.path.join(BASE_DIR, "data", "embeddings", "product_ids.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference on: {device}")
    
    # ─── Load models ───
    print("🧠 Loading models...")
    
    # YOLOv8 for detection
    yolo_path = args.yolo_model or os.path.join(BASE_DIR, "models", "best.pt")
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
        print(f"   ✅ YOLO: {yolo_path}")
    else:
        # Fallback to default YOLOv8 
        yolo_model = YOLO("yolov8n.pt")
        print(f"   ⚠️ Using generic YOLOv8n (no fashion-specific model found)")
    
    # ConvNeXt metric model
    metric_model = FashionEmbedder(embed_dim=args.embed_dim, pretrained=False).to(device)
    
    checkpoint_path = args.checkpoint or os.path.join(BASE_DIR, "checkpoints", "best_metric_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        metric_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Metric model: {checkpoint_path}")
    else:
        print(f"   ⚠️ No checkpoint found, using pretrained ConvNeXt")
        metric_model = FashionEmbedder(embed_dim=args.embed_dim, pretrained=True).to(device)
    
    metric_model.eval()
    
    # FAISS index
    print("📦 Loading FAISS index...")
    index = faiss.read_index(faiss_path)
    with open(ids_path, 'rb') as f:
        product_ids = pickle.load(f)
    print(f"   Index: {index.ntotal} products × {index.d} dims")
    
    # Product categories
    df_products = pd.read_csv(products_csv)
    product_categories = dict(
        zip(df_products['product_asset_id'].astype(str), df_products['product_description'])
    )
    
    # Transform
    transform = get_inference_transform(args.img_size)
    
    # Re-ranker
    reranker = Reranker(w_embed=0.5, w_color=0.2, w_cat=0.3)
    
    # ─── Load test bundles ───
    df_test = pd.read_csv(test_csv)
    bundle_ids = df_test['bundle_asset_id'].tolist()
    print(f"\n🎯 Processing {len(bundle_ids)} test bundles...")
    
    results = []
    start_time = time.time()
    
    for i, bundle_id in enumerate(bundle_ids):
        try:
            bundle_path = os.path.join(bundle_images_dir, f"{bundle_id}.jpg")
            
            if not os.path.exists(bundle_path):
                # Fallback: return first 15 product IDs
                top_15 = product_ids[:15]
                results.append((bundle_id, top_15))
                continue
            
            image = Image.open(bundle_path).convert("RGB")
            w, h = image.size
            
            # ─── Phase 1: YOLOv8 Detection ───
            yolo_results = yolo_model(image, conf=0.15, verbose=False)[0]
            
            crops = []
            crop_labels = []
            
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]
                
                if cls_name in YOLO_CLASSES_CLOTHING:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = image.crop((x1, y1, x2, y2))
                    crops.append(crop)
                    crop_labels.append(cls_name)
            
            # Fallback: split image in half
            if len(crops) == 0:
                crops.append(image.crop((0, 0, w, h // 2)))
                crop_labels.append("top")
                crops.append(image.crop((0, h // 2, w, h)))
                crop_labels.append("trousers")
            
            # ─── Phase 2: Metric Embeddings ───
            crop_tensors = torch.stack([transform(c) for c in crops]).to(device)
            
            with torch.no_grad():
                with autocast():
                    crop_embeddings = metric_model(crop_tensors)
            
            crop_embeddings = crop_embeddings.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(crop_embeddings)
            
            # ─── Phase 3: FAISS Search (Top-50 per crop) ───
            k_search = min(args.top_k_search, index.ntotal)
            scores_all, indices_all = index.search(crop_embeddings, k_search)
            
            # ─── Phase 4: Re-Rank & Merge ───
            merged_scores = defaultdict(float)
            
            for crop_idx in range(len(crops)):
                candidate_ids = [product_ids[idx] for idx in indices_all[crop_idx] if idx >= 0]
                candidate_scores = scores_all[crop_idx][:len(candidate_ids)]
                yolo_class = crop_labels[crop_idx]
                
                # Re-rank using category consistency + embedding score
                reranked = reranker.rerank_simple(
                    candidate_ids=candidate_ids,
                    candidate_scores=candidate_scores,
                    yolo_class=yolo_class,
                    product_categories=product_categories,
                    top_k=args.top_k_search,
                )
                
                # Merge across crops: sum scores, de-duplicate
                for pid, score in reranked:
                    merged_scores[pid] = max(merged_scores[pid], score)
            
            # Take top 15
            sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
            top_15 = [pid for pid, _ in sorted_results[:15]]
            
            # Pad if needed
            if len(top_15) < 15:
                for pid in product_ids:
                    if pid not in top_15:
                        top_15.append(pid)
                    if len(top_15) >= 15:
                        break
            
            results.append((bundle_id, top_15))
            
        except Exception as e:
            print(f"  ❌ Error on {bundle_id}: {e}")
            results.append((bundle_id, product_ids[:15]))
        
        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(bundle_ids) - i - 1) / rate
            print(f"  [{i+1}/{len(bundle_ids)}] {rate:.1f} bundles/s | ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Processed {len(bundle_ids)} bundles in {elapsed:.1f}s")
    
    # ─── Generate CSV ───
    print("📝 Generating submission CSV...")
    rows = []
    for bundle_id, top_15 in results:
        for pid in top_15:
            rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})
    
    df_submission = pd.DataFrame(rows)
    output_path = os.path.join(BASE_DIR, "submission_metric.csv")
    df_submission.to_csv(output_path, index=False)
    print(f"💾 Saved: {output_path}")
    print(f"   Rows: {len(df_submission)} ({len(results)} bundles × 15 products)")
    
    # ─── Evaluate on training data (if requested) ───
    if args.evaluate:
        print("\n📊 Evaluating on training set (proxy MRR@15)...")
        train_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
        df_train = pd.read_csv(train_csv)
        
        # Check how many test bundles appear in training pairs
        train_bundles = set(df_train['bundle_asset_id'].tolist())
        result_dict = {bid: pids for bid, pids in results}
        
        hits = 0
        total = 0
        mrr_sum = 0.0
        
        for _, row in df_train.iterrows():
            bid = row['bundle_asset_id']
            true_pid = str(row['product_asset_id'])
            
            if bid in result_dict:
                predicted = [str(p) for p in result_dict[bid]]
                total += 1
                if true_pid in predicted:
                    rank = predicted.index(true_pid) + 1
                    mrr_sum += 1.0 / rank
                    hits += 1
        
        if total > 0:
            mrr = mrr_sum / total
            acc = hits / total
            print(f"   MRR@15: {mrr:.4f} | Hit@15: {acc:.4f} ({hits}/{total})")
        else:
            print("   No overlap between test and train bundles for evaluation")


def parse_args():
    parser = argparse.ArgumentParser(description="Run full inference pipeline")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to metric model checkpoint")
    parser.add_argument("--yolo_model", type=str, default=None, help="Path to YOLOv8 model weights")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--top_k_search", type=int, default=50, help="FAISS top-K candidates per crop")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate MRR@15 on training bundles")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

"""
final_inference.py — Kaggle Grandmaster Inference Pipeline
══════════════════════════════════════════════════════════════

Pipeline:
  1. Auto-discover best model checkpoint
  2. YOLOv8 garment detection (with smart fallback)
  3. Test-Time Augmentation: 3 passes per crop (original + hflip + center90)
  4. FAISS Top-100 retrieval per crop
  5. Hard category mask → Block HSV re-ranking
  6. Cross-crop fusion → Top-15 per bundle
  7. CSV output + MRR@15 evaluation

Every line here is engineered to squeeze MRR@15 decimals.
"""
import argparse
import glob
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
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FashionEmbedder
from reranker import Reranker


# ═════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════

YOLO_CLASSES_CLOTHING = {"top", "trousers", "skirt", "dress", "outwear"}

# How many YOLO detections max per bundle (avoid processing 20 accessories)
MAX_CROPS_PER_BUNDLE = 6

# FAISS search depth — wider funnel = more candidates for re-ranking
FAISS_TOP_K = 100


# ═════════════════════════════════════════════════════════════════
# AUTO-LOADER: Find the best model automatically
# ═════════════════════════════════════════════════════════════════

def auto_discover_checkpoint(base_dir: str, manual_path: str = None) -> str:
    """
    Auto-discover the best model checkpoint.
    
    Search order:
    1. Manual path (--checkpoint flag)
    2. checkpoints/best_metric_model.pt
    3. checkpoints/final_metric_model.pt  
    4. Any .pt file in checkpoints/ sorted by modification time (newest first)
    5. Any .pt file in outputs/ or models/ (common Kaggle patterns)
    
    Returns path to the best available checkpoint, or None if nothing found.
    """
    if manual_path and os.path.exists(manual_path):
        return manual_path

    # Priority-ordered candidate paths
    candidates = [
        os.path.join(base_dir, "checkpoints", "best_metric_model.pt"),
        os.path.join(base_dir, "checkpoints", "final_metric_model.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Glob search in common directories
    search_dirs = [
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "outputs"),
        os.path.join(base_dir, "models"),
    ]

    all_pts = []
    for d in search_dirs:
        if os.path.isdir(d):
            pts = glob.glob(os.path.join(d, "*.pt"))
            pts += glob.glob(os.path.join(d, "*.pth"))
            all_pts.extend(pts)

    if all_pts:
        # Sort by modification time, newest first
        all_pts.sort(key=os.path.getmtime, reverse=True)
        return all_pts[0]

    return None


# ═════════════════════════════════════════════════════════════════
# TEST-TIME AUGMENTATION (TTA)
# ═════════════════════════════════════════════════════════════════

def get_base_transform(img_size: int = 224):
    """Standard inference transform matching training normalization."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def tta_encode(
    model: torch.nn.Module,
    crop_pil: Image.Image,
    device: torch.device,
    img_size: int = 224,
) -> np.ndarray:
    """
    Test-Time Augmentation: 3 views of the same crop.
    
    View 1: Original (standard resize + normalize)
    View 2: Horizontal flip (catches asymmetric patterns)
    View 3: 90% center crop (removes edge noise — critical for noisy YOLO crops
             that may include bits of background or adjacent garments)
    
    The 3 embeddings are L2-normalized individually, then AVERAGED and
    re-normalized. This produces a more stable "fingerprint" that is
    robust to the exact crop boundaries.
    
    Returns: (embed_dim,) numpy array, L2-normalized
    """
    base_tf = get_base_transform(img_size)

    views = []

    # View 1: Original
    views.append(base_tf(crop_pil))

    # View 2: Horizontal flip
    flipped = TF.hflip(crop_pil)
    views.append(base_tf(flipped))

    # View 3: Center crop at 90% — shaves noisy edges
    w, h = crop_pil.size
    margin_w = int(w * 0.05)
    margin_h = int(h * 0.05)
    center_crop = crop_pil.crop((
        margin_w, margin_h,
        w - margin_w, h - margin_h
    ))
    views.append(base_tf(center_crop))

    # Stack all views into a single batch
    batch = torch.stack(views).to(device)  # (3, C, H, W)

    with torch.no_grad():
        with autocast():
            embeddings = model(batch)  # (3, embed_dim), each L2-normalized by model

    # Average the 3 normalized embeddings → re-normalize
    avg_embed = embeddings.mean(dim=0)
    avg_embed = F.normalize(avg_embed, p=2, dim=0)

    return avg_embed.cpu().numpy().astype(np.float32)


def tta_encode_batch(
    model: torch.nn.Module,
    crops: list,
    device: torch.device,
    img_size: int = 224,
) -> np.ndarray:
    """
    Batch TTA for multiple crops. More GPU-efficient than per-crop TTA.
    
    Returns: (n_crops, embed_dim) numpy array, each row L2-normalized
    """
    if len(crops) == 0:
        return np.zeros((0, 256), dtype=np.float32)

    base_tf = get_base_transform(img_size)
    all_tensors = []

    for crop_pil in crops:
        # View 1: Original
        all_tensors.append(base_tf(crop_pil))

        # View 2: Horizontal flip
        all_tensors.append(base_tf(TF.hflip(crop_pil)))

        # View 3: 90% center crop
        w, h = crop_pil.size
        mw, mh = int(w * 0.05), int(h * 0.05)
        center = crop_pil.crop((mw, mh, w - mw, h - mh))
        all_tensors.append(base_tf(center))

    # Single forward pass for ALL views of ALL crops
    batch = torch.stack(all_tensors).to(device)  # (n_crops * 3, C, H, W)

    with torch.no_grad():
        with autocast():
            all_embeds = model(batch)  # (n_crops * 3, embed_dim)

    # Reshape: (n_crops, 3, embed_dim)
    all_embeds = all_embeds.view(len(crops), 3, -1)

    # Average across 3 views, re-normalize
    avg_embeds = all_embeds.mean(dim=1)  # (n_crops, embed_dim)
    avg_embeds = F.normalize(avg_embeds, p=2, dim=1)

    return avg_embeds.cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════════
# SMART DETECTION: YOLO with crop quality filtering
# ═════════════════════════════════════════════════════════════════

def detect_and_crop(
    yolo_model,
    image: Image.Image,
    conf_threshold: float = 0.12,
    min_area_ratio: float = 0.01,
) -> tuple:
    """
    Run YOLOv8 detection with quality filtering.
    
    Filters out:
    - Non-clothing detections
    - Tiny crops (< 1% of image area — likely noise)
    - Too many crops (keeps top MAX_CROPS_PER_BUNDLE by confidence)
    
    Fallback: 3-way split (top third, mid third, bottom third) instead of
    just 2-way like the old code. The 3-way gives us neckline/torso/legs.
    
    Returns: (crops: list[PIL.Image], labels: list[str])
    """
    w, h = image.size
    image_area = w * h
    min_area = image_area * min_area_ratio

    yolo_results = yolo_model(image, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf = float(box.conf[0])

        if cls_name not in YOLO_CLASSES_CLOTHING:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_area = (x2 - x1) * (y2 - y1)
        if crop_area < min_area:
            continue

        detections.append((conf, cls_name, x1, y1, x2, y2))

    # Sort by confidence, keep top N
    detections.sort(key=lambda x: x[0], reverse=True)
    detections = detections[:MAX_CROPS_PER_BUNDLE]

    crops = []
    labels = []
    for _, cls_name, x1, y1, x2, y2 in detections:
        crops.append(image.crop((x1, y1, x2, y2)))
        labels.append(cls_name)

    # ── FALLBACK: 3-way split ──
    if len(crops) == 0:
        third = h // 3
        crops.append(image.crop((0, 0, w, third)))
        labels.append("top")
        crops.append(image.crop((0, third, w, 2 * third)))
        labels.append("top")  # mid section often has the main garment
        crops.append(image.crop((0, 2 * third, w, h)))
        labels.append("trousers")

    return crops, labels


# ═════════════════════════════════════════════════════════════════
# MAIN INFERENCE PIPELINE
# ═════════════════════════════════════════════════════════════════

def run_inference(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Paths ──
    test_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv")
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    bundle_images_dir = os.path.join(BASE_DIR, "data", "images", "bundles")
    product_images_dir = os.path.join(BASE_DIR, "data", "images", "products")
    faiss_path = os.path.join(BASE_DIR, "data", "embeddings", "faiss_index.bin")
    ids_path = os.path.join(BASE_DIR, "data", "embeddings", "product_ids.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ═══ STAGE 1: Load Models ═══
    print("\n🧠 Loading models...")

    # -- YOLOv8 --
    yolo_path = args.yolo_model or os.path.join(BASE_DIR, "models", "best.pt")
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
        print(f"   ✅ YOLO: {yolo_path}")
    else:
        yolo_model = YOLO("yolov8n.pt")
        print(f"   ⚠️  No fashion YOLO found → using generic yolov8n")

    # -- ConvNeXt Metric Model (auto-discovery) --
    checkpoint_path = auto_discover_checkpoint(BASE_DIR, args.checkpoint)

    if checkpoint_path:
        print(f"   🔍 Auto-discovered checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        embed_dim = checkpoint.get('embed_dim', args.embed_dim)
        metric_model = FashionEmbedder(embed_dim=embed_dim, pretrained=False).to(device)
        metric_model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_mrr' in checkpoint:
            print(f"   📊 Checkpoint MRR@15: {checkpoint['best_mrr']:.4f}")
        if 'epoch' in checkpoint:
            print(f"   📊 Trained for: {checkpoint['epoch']} epochs")
    else:
        print(f"   ⚠️  No checkpoint found → using pretrained ConvNeXt (no metric learning)")
        embed_dim = args.embed_dim
        metric_model = FashionEmbedder(embed_dim=embed_dim, pretrained=True).to(device)

    metric_model.eval()

    # -- FAISS Index --
    print(f"\n📦 Loading FAISS index...")
    if not os.path.exists(faiss_path):
        print(f"   ❌ FAISS index not found at {faiss_path}")
        print(f"   Run build_index.py first!")
        sys.exit(1)

    index = faiss.read_index(faiss_path)
    with open(ids_path, 'rb') as f:
        product_ids = pickle.load(f)
    print(f"   Index: {index.ntotal} products × {index.d} dims")

    # -- Product metadata --
    df_products = pd.read_csv(products_csv)
    product_categories = dict(
        zip(df_products['product_asset_id'].astype(str), df_products['product_description'])
    )

    # -- Re-ranker --
    use_color = args.use_color_rerank
    reranker = Reranker(w_color=0.25, color_enabled=use_color)
    print(f"   Re-ranker: category_mask=HARD | color={'ON (block HSV)' if use_color else 'OFF'}")

    # ═══ STAGE 2: Process Test Bundles ═══
    df_test = pd.read_csv(test_csv)
    bundle_ids = df_test['bundle_asset_id'].tolist()

    print(f"\n{'═'*60}")
    print(f"🎯 Processing {len(bundle_ids)} test bundles")
    print(f"   TTA: {'ON (3 views)' if args.use_tta else 'OFF'}")
    print(f"   FAISS depth: Top-{min(FAISS_TOP_K, index.ntotal)}")
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

            # ── Phase 1: Detect & Crop ──
            crops, crop_labels = detect_and_crop(
                yolo_model, image, conf_threshold=0.12
            )

            # ── Phase 2: Encode with TTA ──
            if args.use_tta:
                crop_embeddings = tta_encode_batch(
                    metric_model, crops, device, args.img_size
                )
            else:
                base_tf = get_base_transform(args.img_size)
                crop_tensors = torch.stack([base_tf(c) for c in crops]).to(device)
                with torch.no_grad():
                    with autocast():
                        crop_embeddings = metric_model(crop_tensors)
                crop_embeddings = crop_embeddings.cpu().numpy().astype(np.float32)

            # Ensure L2 normalized
            faiss.normalize_L2(crop_embeddings)

            # ── Phase 3: FAISS Search ──
            k_search = min(FAISS_TOP_K, index.ntotal)
            scores_all, indices_all = index.search(crop_embeddings, k_search)

            # ── Phase 4: Re-Rank & Merge ──
            merged_scores = defaultdict(float)

            for crop_idx in range(len(crops)):
                valid_mask = indices_all[crop_idx] >= 0
                candidate_indices = indices_all[crop_idx][valid_mask]
                candidate_ids = [product_ids[idx] for idx in candidate_indices]
                candidate_scores = scores_all[crop_idx][valid_mask]
                yolo_class = crop_labels[crop_idx]

                if use_color:
                    # Convert crop to numpy for color comparison
                    crop_np = np.array(crops[crop_idx])
                    reranked = reranker.rerank_with_color(
                        crop_image_rgb=crop_np,
                        yolo_class=yolo_class,
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        product_images_dir=product_images_dir,
                        product_categories=product_categories,
                        top_k=k_search,
                    )
                else:
                    reranked = reranker.rerank_simple(
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        yolo_class=yolo_class,
                        product_categories=product_categories,
                        top_k=k_search,
                    )

                # Merge: keep max score per product across crops
                for pid, score in reranked:
                    merged_scores[pid] = max(merged_scores[pid], score)

            # Top 15
            sorted_results = sorted(
                merged_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_15 = [pid for pid, _ in sorted_results[:15]]

            # Pad to exactly 15 if needed
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
                print(f"  ❌ Error on {bundle_id}: {e}")
            results.append((bundle_id, product_ids[:15]))

        # Progress bar
        if (i + 1) % 25 == 0 or (i + 1) == len(bundle_ids):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(bundle_ids) - i - 1) / max(rate, 0.01)
            pct = 100 * (i + 1) / len(bundle_ids)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(
                f"  [{bar}] {pct:5.1f}% | "
                f"{i+1}/{len(bundle_ids)} | "
                f"{rate:.1f} b/s | "
                f"ETA {eta:.0f}s"
            )

    total_time = time.time() - start_time
    print(f"\n✅ Done! {len(bundle_ids)} bundles in {total_time:.1f}s "
          f"({len(bundle_ids)/total_time:.1f} b/s)")
    if errors > 0:
        print(f"   ⚠️  {errors} errors (fallback used)")

    # ═══ STAGE 3: Generate Submission CSV ═══
    print("\n📝 Generating submission CSV...")
    rows = []
    for bundle_id, top_15 in results:
        for pid in top_15:
            rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})

    df_submission = pd.DataFrame(rows)
    output_path = os.path.join(BASE_DIR, "submission_metric.csv")
    df_submission.to_csv(output_path, index=False)

    # Sanity checks
    n_bundles = df_submission['bundle_asset_id'].nunique()
    avg_products = len(df_submission) / max(n_bundles, 1)
    print(f"💾 Saved: {output_path}")
    print(f"   Rows: {len(df_submission)} | Bundles: {n_bundles} | Avg products/bundle: {avg_products:.1f}")

    # ═══ STAGE 4: Evaluate ═══
    if args.evaluate:
        evaluate_mrr(results, BASE_DIR)


def evaluate_mrr(results: list, base_dir: str):
    """Compute MRR@15 using training pairs as proxy ground truth."""
    print(f"\n{'═'*60}")
    print("📊 MRR@15 Evaluation (on training set overlap)")
    print(f"{'═'*60}")

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
        mrr = mrr_sum / total
        print(f"\n   🏆 MRR@15:  {mrr:.4f}")
        print(f"   📊 Hit@1:   {hit_at[1]/total:.4f} ({hit_at[1]}/{total})")
        print(f"   📊 Hit@5:   {hit_at[5]/total:.4f} ({hit_at[5]}/{total})")
        print(f"   📊 Hit@10:  {hit_at[10]/total:.4f} ({hit_at[10]}/{total})")
        print(f"   📊 Hit@15:  {hit_at[15]/total:.4f} ({hit_at[15]}/{total})")
    else:
        print("   ⚠️  No overlap between test and train bundles")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="🏆 Kaggle Grandmaster Inference Pipeline for Inditex Visual Search"
    )

    # Model paths
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Manual path to model checkpoint (auto-discovered if not set)")
    parser.add_argument("--yolo_model", type=str, default=None,
                        help="Path to YOLOv8 fashion model weights")

    # Model config
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension (must match training)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")

    # Inference strategy
    parser.add_argument("--use_tta", action="store_true", default=True,
                        help="Enable Test-Time Augmentation (3 views per crop)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable TTA for faster inference")
    parser.add_argument("--use_color_rerank", action="store_true", default=False,
                        help="Enable block HSV color re-ranking (slower, more accurate)")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Compute MRR@15 on training set overlap")

    args = parser.parse_args()

    # Handle --no_tta flag
    if args.no_tta:
        args.use_tta = False

    return args


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

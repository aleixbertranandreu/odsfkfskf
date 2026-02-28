"""
build_index.py — Build FAISS Index from Trained Metric Model
Encodes all catalog product images into 256-dim embeddings
and stores them in a FAISS IndexFlatIP for fast cosine retrieval.
"""
import argparse
import os
import pickle
import sys
import time

import faiss
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FashionEmbedder
from dataset import CatalogDataset


def build_index(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    product_images = os.path.join(BASE_DIR, "data", "images", "products")
    embeddings_dir = os.path.join(BASE_DIR, "data", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Building index on: {device}")
    
    # ─── Load model ───
    print("🧠 Loading trained model...")
    model = FashionEmbedder(embed_dim=args.embed_dim, pretrained=False).to(device)
    
    checkpoint_path = args.checkpoint or os.path.join(BASE_DIR, "checkpoints", "best_metric_model.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded checkpoint: {checkpoint_path}")
        if 'best_mrr' in checkpoint:
            print(f"   📊 Model MRR@15: {checkpoint['best_mrr']:.4f}")
    else:
        print(f"   ⚠️  No checkpoint found at {checkpoint_path}")
        print(f"   Using pretrained ConvNeXt (no metric learning fine-tuning)")
        model = FashionEmbedder(embed_dim=args.embed_dim, pretrained=True).to(device)
    
    model.eval()
    
    # ─── Load catalog ───
    print("📦 Loading catalog images...")
    catalog_ds = CatalogDataset(
        products_csv=products_csv,
        product_images_dir=product_images,
        img_size=args.img_size,
    )
    
    catalog_loader = DataLoader(
        catalog_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # ─── Encode all products ───
    print("🔄 Encoding catalog...")
    all_embeddings = []
    all_ids = []
    
    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, pids) in enumerate(catalog_loader):
            images = images.to(device, non_blocking=True)
            
            with autocast():
                embeddings = model(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(pids)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Encoded {(batch_idx+1) * args.batch_size}/{len(catalog_ds)} products...")
    
    elapsed = time.time() - start
    
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"✅ Encoded {len(all_ids)} products in {elapsed:.1f}s")
    print(f"   Embeddings shape: {all_embeddings.shape}")
    
    # ─── Build FAISS index ───
    print("🔨 Building FAISS index...")
    dim = all_embeddings.shape[1]
    
    # IndexFlatIP = Inner Product (= cosine similarity for L2-normalized vectors)
    index = faiss.IndexFlatIP(dim)
    
    # Normalize embeddings (should already be normalized from model, but ensure it)
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    print(f"   Index size: {index.ntotal} vectors × {dim} dims")
    
    # ─── Save ───
    faiss_path = os.path.join(embeddings_dir, "faiss_index.bin")
    ids_path = os.path.join(embeddings_dir, "product_ids.pkl")
    embeddings_path = os.path.join(embeddings_dir, "metric_embeddings.npy")
    
    faiss.write_index(index, faiss_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(all_ids, f)
    np.save(embeddings_path, all_embeddings)
    
    print(f"\n💾 Saved:")
    print(f"   FAISS index: {faiss_path}")
    print(f"   Product IDs: {ids_path}")
    print(f"   Raw embeddings: {embeddings_path}")
    print(f"\n🏆 Index build complete! Ready for inference.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from trained metric model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension (must match training)")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args)

"""
build_index.py — Build FAISS Index from the FULL Product Catalog
═══════════════════════════════════════════════════════════════════

Scans the entire product images directory for ALL .jpg/.png files,
NOT just the training subset. This is critical: the catalog has ~27K
products and we must index ALL of them for competition scoring.

Handles corrupt images gracefully (skip + warn, never crash).
"""
import argparse
import glob
import os
import pickle
import sys
import time

import faiss
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FashionEmbedder


# ═════════════════════════════════════════════════════════════════
# Full Catalog Dataset — scans filesystem, not CSV
# ═════════════════════════════════════════════════════════════════

class FullCatalogDataset(Dataset):
    """
    Loads ALL product images from a directory.
    
    Unlike the training CatalogDataset that filters by CSV,
    this scans every .jpg and .png file in the directory.
    Product ID is extracted from the filename (e.g., I_abc123.jpg → I_abc123).
    
    Corrupt/unreadable images are handled gracefully with a sentinel.
    """

    def __init__(self, images_dir: str, img_size: int = 224):
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Scan ALL image files
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        all_files = []
        for pat in patterns:
            all_files.extend(glob.glob(os.path.join(images_dir, pat)))
            all_files.extend(glob.glob(os.path.join(images_dir, pat.upper())))

        # Deduplicate and sort for deterministic ordering
        all_files = sorted(set(all_files))

        self.items = []
        for path in all_files:
            basename = os.path.basename(path)
            pid = os.path.splitext(basename)[0]  # I_abc123.jpg → I_abc123
            self.items.append((pid, path))

        print(f"INFO: Full catalog scan: {len(self.items)} product images found in {images_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, path = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            # Verify image is valid (some files may be truncated)
            img.load()
            tensor = self.transform(img)
            return tensor, pid, True  # True = valid
        except Exception as e:
            # Return a dummy tensor for corrupt images
            # We'll filter these out after batch processing
            dummy = torch.zeros(3, 224, 224)
            return dummy, pid, False  # False = invalid


def robust_collate(batch):
    """Custom collate that keeps track of valid/invalid images."""
    tensors = torch.stack([b[0] for b in batch])
    pids = [b[1] for b in batch]
    valid = [b[2] for b in batch]
    return tensors, pids, valid


# ═════════════════════════════════════════════════════════════════
# Main Builder
# ═════════════════════════════════════════════════════════════════

def auto_discover_checkpoint(base_dir: str, manual_path: str = None) -> str:
    """Find the best model checkpoint automatically."""
    if manual_path and os.path.exists(manual_path):
        return manual_path

    candidates = [
        os.path.join(base_dir, "checkpoints", "best_metric_model.pt"),
        os.path.join(base_dir, "checkpoints", "final_metric_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Glob search
    for d in ["checkpoints", "outputs", "models"]:
        full_d = os.path.join(base_dir, d)
        if os.path.isdir(full_d):
            pts = sorted(glob.glob(os.path.join(full_d, "*.pt")), key=os.path.getmtime, reverse=True)
            if pts:
                return pts[0]

    return None


def build_index(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    product_images = os.path.join(BASE_DIR, "data", "images", "products")
    embeddings_dir = os.path.join(BASE_DIR, "data", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Building FULL catalog index on: {device}")

    # ─── Load model ───
    print("INFO: Loading trained model...")
    checkpoint_path = auto_discover_checkpoint(BASE_DIR, args.checkpoint)

    if checkpoint_path:
        print(f"    Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        embed_dim = checkpoint.get('embed_dim', args.embed_dim)
        model = FashionEmbedder(embed_dim=embed_dim, pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_mrr' in checkpoint:
            print(f"    Model MRR@15: {checkpoint['best_mrr']:.4f}")
    else:
        print(f"   ️  No checkpoint found, using pretrained ConvNeXt")
        embed_dim = args.embed_dim
        model = FashionEmbedder(embed_dim=embed_dim, pretrained=True).to(device)

    model.eval()

    # ─── Load FULL catalog ───
    print("INFO: Scanning full product catalog...")
    catalog_ds = FullCatalogDataset(
        images_dir=product_images,
        img_size=args.img_size,
    )

    if len(catalog_ds) == 0:
        print("ERROR: No product images found! Check the path.")
        sys.exit(1)

    catalog_loader = DataLoader(
        catalog_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=robust_collate,
    )

    # ─── Encode all products ───
    print(f" Encoding {len(catalog_ds)} products...")
    all_embeddings = []
    all_ids = []
    n_corrupt = 0

    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, pids, valid_flags) in enumerate(catalog_loader):
            # Filter out corrupt images
            valid_indices = [i for i, v in enumerate(valid_flags) if v]
            if not valid_indices:
                n_corrupt += len(pids)
                continue

            valid_images = images[valid_indices].to(device, non_blocking=True)
            valid_pids = [pids[i] for i in valid_indices]
            n_corrupt += len(pids) - len(valid_indices)

            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                embeddings = model(valid_images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(valid_pids)

            if (batch_idx + 1) % 100 == 0:
                done = min((batch_idx + 1) * args.batch_size, len(catalog_ds))
                elapsed = time.time() - start
                rate = done / elapsed
                print(f"   [{done}/{len(catalog_ds)}] {rate:.0f} img/s | corrupt: {n_corrupt}")

    elapsed = time.time() - start

    if len(all_embeddings) == 0:
        print("ERROR: No valid embeddings generated!")
        sys.exit(1)

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"\nINFO: SUCCESS: Encoded {len(all_ids)} products in {elapsed:.1f}s ({len(all_ids)/elapsed:.0f} img/s)")
    print(f"   Embeddings shape: {all_embeddings.shape}")
    if n_corrupt > 0:
        print(f"   ️  Skipped {n_corrupt} corrupt/unreadable images")

    # ─── Build FAISS index ───
    print(" Building FAISS index...")
    dim = all_embeddings.shape[1]

    # IndexFlatIP = cosine similarity for L2-normalized vectors
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)

    print(f"   Index: {index.ntotal} vectors × {dim} dims")

    # ─── Save ───
    faiss_path = os.path.join(embeddings_dir, "faiss_index.bin")
    ids_path = os.path.join(embeddings_dir, "product_ids.pkl")
    embeddings_path = os.path.join(embeddings_dir, "metric_embeddings.npy")

    faiss.write_index(index, faiss_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(all_ids, f)
    np.save(embeddings_path, all_embeddings)

    print(f"\n Saved:")
    print(f"   FAISS index:     {faiss_path}")
    print(f"   Product IDs:     {ids_path}")
    print(f"   Raw embeddings:  {embeddings_path}")

    # ─── Cross-check with CSV ───
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    if os.path.exists(products_csv):
        import pandas as pd
        df = pd.read_csv(products_csv)
        csv_ids = set(df['product_asset_id'].astype(str))
        indexed_ids = set(all_ids)
        coverage = len(csv_ids & indexed_ids) / len(csv_ids) * 100
        missing = len(csv_ids - indexed_ids)
        extra = len(indexed_ids - csv_ids)
        print(f"\n Catalog coverage:")
        print(f"   CSV products:    {len(csv_ids)}")
        print(f"   Indexed:         {len(indexed_ids)}")
        print(f"   Coverage:        {coverage:.1f}%")
        if missing > 0:
            print(f"   ️  Missing:     {missing} (images not downloaded)")
        if extra > 0:
            print(f"   ℹ️  Extra:       {extra} (images not in CSV)")

    print(f"\nINFO: Index build complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from FULL product catalog")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args)

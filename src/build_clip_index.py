"""
build_clip_index.py — Build FAISS index using CLIP ViT-L/14
═══════════════════════════════════════════════════════════════

Encodes ALL 27,688 product images with CLIP ViT-L/14 (768-dim).
Uses DataLoader with num_workers=4 and mixed precision for 4x+ speedup.
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor


class CLIPCatalogDataset(Dataset):
    def __init__(self, images_dir: str, processor: CLIPProcessor):
        self.processor = processor
        all_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        all_files += sorted(glob.glob(os.path.join(images_dir, "*.png")))
        all_files = sorted(set(all_files))
        
        self.items = []
        for path in all_files:
            pid = os.path.splitext(os.path.basename(path))[0]
            self.items.append((pid, path))
        print(f"INFO: Found {len(self.items)} product images")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, path = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            img.load()
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, pid, True
        except Exception:
            dummy = torch.zeros(3, 224, 224)
            return dummy, pid, False


def collate_fn(batch):
    pixels = torch.stack([b[0] for b in batch])
    pids = [b[1] for b in batch]
    valid = [b[2] for b in batch]
    return pixels, pids, valid


def build_index(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    product_images = args.products_dir if args.products_dir else os.path.join(BASE_DIR, "data", "images", "products")
    embeddings_dir = os.path.join(BASE_DIR, "data", "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Building CLIP index on: {device} with AMP")

    # Load CLIP
    model_name = args.model_name
    print(f" Loading base model {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    if args.custom_weights:
        print(f" Loading CUSTOM fine-tuned weights from {args.custom_weights}...")
        checkpoint = torch.load(args.custom_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("INFO: SUCCESS: Custom weights loaded successfully!")
        
    model.eval()

    # Dataset & Loader
    dataset = CLIPCatalogDataset(product_images, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f" Encoding {len(dataset)} products (bz={args.batch_size}, workers={args.num_workers})")
    all_embeddings = []
    all_ids = []
    n_corrupt = 0

    start = time.time()
    for batch_idx, (pixel_values, pids, valid_flags) in enumerate(loader):
        valid_idx = [i for i, v in enumerate(valid_flags) if v]
        if not valid_idx:
            n_corrupt += len(pids)
            continue

        valid_pixels = pixel_values[valid_idx].to(device)
        valid_pids = [pids[i] for i in valid_idx]
        n_corrupt += len(pids) - len(valid_idx)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                vision_out = model.vision_model(pixel_values=valid_pixels)
                features = model.visual_projection(vision_out.pooler_output)
            
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)
            
        all_embeddings.append(features.cpu().numpy().astype(np.float32))
        all_ids.extend(valid_pids)

        if (batch_idx + 1) % 20 == 0:
            done = min((batch_idx + 1) * args.batch_size, len(dataset))
            elapsed = time.time() - start
            rate = done / max(elapsed, 1)
            print(f"   [{done:5d}/{len(dataset)}] {rate:4.0f} img/s | corrupt: {n_corrupt}")

    elapsed = time.time() - start
    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    
    print(f"\nINFO: SUCCESS: Encoded {len(all_ids)} products in {elapsed:.1f}s")
    print(f"   Shape: {all_embeddings.shape}")
    if n_corrupt > 0:
        print(f"   ️ Skipped {n_corrupt} corrupt")

    print(" Building FAISS index...")
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    suffix = model_name.replace("/", "_").replace("-", "_")
    faiss_path = os.path.join(embeddings_dir, f"clip_faiss_{suffix}.bin")
    ids_path = os.path.join(embeddings_dir, f"clip_ids_{suffix}.pkl")

    faiss.write_index(index, faiss_path)
    with open(ids_path, 'wb') as f:
        pickle.dump(all_ids, f)

    print(f"INFO: Saved to Saved index to {faiss_path}")
    print("INFO: Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="patrickjohncyh/fashion-clip")
    parser.add_argument("--custom_weights", type=str, default=None, help="Path to fine-tuned .pt weights")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--products_dir", type=str, default=None, help="Custom folder for products")
    args = parser.parse_args()
    build_index(args)


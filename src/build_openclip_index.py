"""
build_openclip_index.py — Resumable OpenCLIP ViT-L/14 index builder
"""
import argparse
import glob
import os
import pickle
import time
import shutil

import faiss
import numpy as np
import torch
from PIL import Image
import open_clip

def build_index(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    print(f"🧠 Loading {args.model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    model.eval()
    
    images_dir = os.path.join(args.base_dir, "data", "images", "products")
    all_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(images_dir, "*.png")))
    all_files = sorted(set(all_files))
    print(f"📦 Found {len(all_files)} product images")

    chunk_dir = os.path.join(args.base_dir, "data", "embeddings", "openclip_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # Check what's already done
    processed_ids = set()
    for f in glob.glob(os.path.join(chunk_dir, "*_ids.pkl")):
        with open(f, 'rb') as pkl:
            processed_ids.update(pickle.load(pkl))
            
    print(f"🔄 Resuming... {len(processed_ids)} already processed")
    
    pending_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] not in processed_ids]
    print(f"⏳ {len(pending_files)} pending images")

    batch_size = args.batch_size
    chunk_size = args.chunk_size

    chunk_embeddings = []
    chunk_ids = []
    chunk_idx = len(glob.glob(os.path.join(chunk_dir, "*_embs.npy")))
    n_corrupt = 0
    start = time.time()

    for batch_start in range(0, len(pending_files), batch_size):
        batch_files = pending_files[batch_start:batch_start + batch_size]
        batch_images = []
        batch_ids = []
        
        for path in batch_files:
            pid = os.path.splitext(os.path.basename(path))[0]
            try:
                img = Image.open(path).convert("RGB")
                img.load()
                batch_images.append(preprocess(img))
                batch_ids.append(pid)
            except Exception:
                n_corrupt += 1
                continue
        
        if batch_images:
            pixel_values = torch.stack(batch_images).to(device)
            with torch.no_grad():
                features = model.encode_image(pixel_values)
                features = features / features.norm(dim=-1, keepdim=True)
            
            chunk_embeddings.append(features.cpu().numpy())
            chunk_ids.extend(batch_ids)

        # Save chunk
        if len(chunk_ids) >= chunk_size or batch_start + batch_size >= len(pending_files):
            if chunk_ids:
                embs = np.vstack(chunk_embeddings).astype(np.float32)
                np.save(os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}_embs.npy"), embs)
                with open(os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}_ids.pkl"), 'wb') as f:
                    pickle.dump(chunk_ids, f)
                
                elapsed = time.time() - start
                rate = (batch_start + batch_size) / elapsed if elapsed > 0 else 0
                print(f"  💾 Saved chunk {chunk_idx}: {len(chunk_ids)} imgs ({rate:.1f} img/s overall)")
                
                chunk_embeddings = []
                chunk_ids = []
                chunk_idx += 1
                
                # Clear CUDA cache if needed
                torch.cuda.empty_cache()

    print(f"\n✅ Encoding complete. Building final FAISS index...")
    
    all_embs = []
    all_idx = []
    for f in sorted(glob.glob(os.path.join(chunk_dir, "*_embs.npy"))):
        all_embs.append(np.load(f))
    for f in sorted(glob.glob(os.path.join(chunk_dir, "*_ids.pkl"))):
        with open(f, 'rb') as pkl:
            all_idx.extend(pickle.load(pkl))
            
    if not all_embs:
        print("No embeddings found!")
        return

    final_embs = np.vstack(all_embs).astype(np.float32)
    dim = final_embs.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(final_embs)
    index.add(final_embs)
    
    emb_dir = os.path.join(args.base_dir, "data", "embeddings")
    suffix = args.model_name.replace("/", "_").replace("-", "_")
    faiss.write_index(index, os.path.join(emb_dir, f"openclip_faiss_{suffix}.bin"))
    with open(os.path.join(emb_dir, f"openclip_ids_{suffix}.pkl"), 'wb') as f:
        pickle.dump(all_idx, f)
        
    print(f"📉 Removing temp chunks...")
    shutil.rmtree(chunk_dir)

    print(f"🎉 Done! Index built with {index.ntotal} items x {dim} dims")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="ViT-L-14")
    p.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--batch_size", type=int, default=32) # Lowered to prevent OOM
    p.add_argument("--chunk_size", type=int, default=2000)
    p.add_argument("--base_dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = p.parse_args()
    build_index(args)

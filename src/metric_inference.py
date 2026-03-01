import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import faiss

# Add parent to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.model import FashionEmbedder

# ----------------------------------------------------------------------
# 1. Dataset for Pure Inference (No Triplets)
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, image_dir: str, image_ids: list, img_size: int = 224):
        self.image_dir = image_dir
        self.image_ids = image_ids
        
        # Exact same normalization as training (ConvNeXt standard)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback for corrupt bits
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            
        return self.transform(img), str(img_id)

# ----------------------------------------------------------------------
# 2. Main Inference Flow
# ----------------------------------------------------------------------
def main(args):
    print("🚀 Iniciando Inferencia del Especialista en Texturas (ConvNeXt)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Utilizando dispositivo: {device}")
    
    # 1. Load Model
    print(f"\n📦 Cargando pesos de {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    
    # Handle both full state_dict or raw parameter dict
    embed_dim = checkpoint.get('embed_dim', 256)
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    model = FashionEmbedder(embed_dim=embed_dim, pretrained=False)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # 2. Extract Product Features
    print("\n👕 Extrayendo embeddings del Catálogo de Productos...")
    df_products = pd.read_csv(args.products_csv)
    # Only keep products that actually have images
    valid_products = [
        pid for pid in df_products['product_asset_id']
        if os.path.exists(os.path.join(args.products_dir, f"{pid}.jpg"))
    ]
    
    dataset_prods = InferenceDataset(args.products_dir, valid_products, args.img_size)
    loader_prods = DataLoader(dataset_prods, batch_size=args.batch_size, num_workers=args.num_workers)
    
    prod_embeddings = []
    prod_ids_list = []
    
    with torch.no_grad():
        for imgs, ids in tqdm(loader_prods, desc="Products"):
            imgs = imgs.to(device)
            # Mixed precision for speed
            with torch.amp.autocast('cuda'):
                emb = model(imgs)
            emb = emb.cpu().numpy()
            prod_embeddings.append(emb)
            prod_ids_list.extend(ids)
            
    prod_embeddings = np.vstack(prod_embeddings).astype('float32')
    
    # 3. Build FAISS Index
    print(f"\n🧠 Construyendo índice FAISS con {len(prod_embeddings)} productos...")
    faiss.normalize_L2(prod_embeddings)  # Cosine similarity requires L2 normalization
    dim = prod_embeddings.shape[1]
    
    # Using FlatIP (Inner Product) which is equivalent to Cosine Sim after L2 norm
    index = faiss.IndexFlatIP(dim)
    index.add(prod_embeddings)
    
    # 4. Extract Bundle Features
    print("\n👗 Extrayendo embeddings de los Test Bundles...")
    df_test = pd.read_csv(args.test_csv)
    test_bundles = df_test['bundle_asset_id'].unique().tolist()
    
    dataset_bundles = InferenceDataset(args.bundles_dir, test_bundles, args.img_size)
    loader_bundles = DataLoader(dataset_bundles, batch_size=args.batch_size, num_workers=args.num_workers)
    
    bundle_embeddings = []
    bundle_ids_list = []
    
    with torch.no_grad():
        for imgs, ids in tqdm(loader_bundles, desc="Bundles"):
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                emb = model(imgs)
            emb = emb.cpu().numpy()
            bundle_embeddings.append(emb)
            bundle_ids_list.extend(ids)
            
    bundle_embeddings = np.vstack(bundle_embeddings).astype('float32')
    faiss.normalize_L2(bundle_embeddings)
    
    # 5. Search Top 60 (We request 60 so RRF has plenty of candidates to overlap)
    print(f"\n🔍 Buscando los Top {args.top_k} matches para cada Bundle...")
    distances, indices = index.search(bundle_embeddings, args.top_k)
    
    # 6. Generate CSV
    print("\n📝 Escribiendo submission CSV...")
    results = []
    for i, bundle_id in enumerate(bundle_ids_list):
        for rank in range(args.top_k):
            match_idx = indices[i][rank]
            predicted_product_id = prod_ids_list[match_idx]
            results.append({
                "bundle_asset_id": bundle_id,
                "product_asset_id": predicted_product_id
            })
            
    df_submission = pd.DataFrame(results)
    df_submission.to_csv(args.output, index=False)
    print(f"🎉 ÉXITO: Archivo guardado en {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvNeXt Metric Learning Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to best_metric_model.pt")
    
    # Defaults set up for the user's RunPod environment
    parser.add_argument("--products_csv", type=str, default=os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv"))
    parser.add_argument("--test_csv", type=str, default=os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv"))
    parser.add_argument("--products_dir", type=str, default="/workspace/fix_products")
    parser.add_argument("--bundles_dir", type=str, default="/workspace/fix_bundles")
    
    parser.add_argument("--output", type=str, default="metric_submission.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=60, help="Number of products to retrieve per bundle")
    
    args = parser.parse_args()
    main(args)

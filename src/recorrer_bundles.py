# completar_bundles_faiss.py
import torch
import open_clip
import numpy as np
import faiss
import pickle
import cv2 as cv
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
PRODUCTS_EMBEDDINGS = "products_clip_embeddings.npy"
PRODUCTS_IDS = "products_ids.pkl"
BUNDLES_CSV = "../data/raw/bundles_product_match_test.csv"  # CSV con bundles y columna product_asset_id vacía
BUNDLES_DIR = Path("../data/images/bundles")
OUTPUT_CSV = "bundle_predictions_completed.csv"
TOP_K = 1  # solo el más similar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1️⃣ Cargar modelo CLIP
# -----------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE)
model.eval()

# -----------------------------
# 2️⃣ Cargar embeddings de productos
# -----------------------------
embeddings_products = np.load(PRODUCTS_EMBEDDINGS).astype(np.float32)
with open(PRODUCTS_IDS, "rb") as f:
    product_ids = pickle.load(f)

faiss.normalize_L2(embeddings_products)
d = embeddings_products.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings_products)
print(f"FAISS index cargado con {index.ntotal} vectores.")

# -----------------------------
# 3️⃣ Cargar CSV de bundles
# -----------------------------
bundles_df = pd.read_csv(BUNDLES_CSV)
bundles_df["product_asset_id"] = bundles_df["product_asset_id"].astype("string")
print(f"Bundles a completar: {len(bundles_df)}")

# -----------------------------
# 4️⃣ Recorrer bundles y completar product_asset_id
# -----------------------------
for idx, row in tqdm(bundles_df.iterrows(), total=len(bundles_df), desc="Procesando bundles"):
    bundle_id = row["bundle_asset_id"]
    bundle_path = BUNDLES_DIR / f"{bundle_id}.jpg"
    if not bundle_path.exists():
        print(f"No encontrada: {bundle_id}")
        continue

    bundle_img = cv.imread(str(bundle_path))
    if bundle_img is None:
        continue

    # Calcular embedding CLIP del bundle
    img_tensor = preprocess(Image.fromarray(cv.cvtColor(bundle_img, cv.COLOR_BGR2RGB)))
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy().astype(np.float32)

    # Buscar producto más similar
    distances, indices = index.search(embedding, TOP_K)
    best_idx = indices[0][0]
    best_score = distances[0][0]

    predicted_product = product_ids[best_idx].replace(".jpg", "")
    bundles_df.at[idx, "product_asset_id"] = predicted_product

    print(f"{bundle_id} → {predicted_product} (score: {best_score:.4f})")

# -----------------------------
# 5️⃣ Guardar CSV completado
# -----------------------------
bundles_df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV completado guardado en: {OUTPUT_CSV}")
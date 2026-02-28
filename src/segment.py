import open_clip
from PIL import Image
import torch
from constants import BUNDLES_IMAGES_DIR, PRODUCTS_DATASET_CSV, BUNDLE_DATASET_CSV
import pandas as pd
import requests
from io import BytesIO

# 1️⃣ Carga bundles CSV
bundles_df = pd.read_csv(BUNDLE_DATASET_CSV)  # tu CSV con bundle_asset_id
print(f"{len(bundles_df)} bundles a procesar")

# 2️⃣ Carga productos
products_df = pd.read_csv(PRODUCTS_DATASET_CSV)
products = products_df.to_dict(orient="records")  # lista de dicts con product_asset_id, description, URL
labels = [p["product_description"] for p in products]

# 3️⃣ Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4️⃣ Carga FashionCLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:Marqo/marqo-fashionCLIP"
)
model = model.to(DEVICE)
model.eval()
tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

# 5️⃣ Tokeniza textos de productos y obtiene embeddings
text_input = tokenizer(labels).to(DEVICE)
with torch.no_grad():
    text_feats = model.encode_text(text_input)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

# 6️⃣ Procesa cada bundle y encuentra producto más similar
product_asset_ids = []

for bundle_file in BUNDLES_IMAGES_DIR.glob("*.jpg"):
    bundle_image = Image.open(bundle_file).convert("RGB")
    image_input = preprocess(bundle_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_feats = model.encode_image(image_input)
        image_feats /= image_feats.norm(dim=-1, keepdim=True)

        # cosine similarity
        sim_matrix = image_feats @ text_feats.T

    # mejor producto
    best_idx = sim_matrix.argmax().item()
    best_product = products[best_idx]
    product_asset_ids.append(best_product["product_asset_id"])

    print(f"Bundle: {bundle_file.name} → Producto más similar: {best_product['product_description']} ({best_product['product_asset_id']})")

# 7️⃣ Completa el CSV
bundles_df["product_asset_id"] = product_asset_ids
bundles_df.to_csv("bundles_completed.csv", index=False)
print("CSV actualizado guardado como bundles_completed.csv")
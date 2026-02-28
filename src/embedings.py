import os
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from constants import (
    PRODUCTS_IMAGES_DIR,
    PRODUCTS_DATASET_CSV,
    OUTPUT_EMB_PATH,
    OUTPUT_META_PATH,
    BATCH_SIZE,
    DEVICE
)
# ==============================
# CONFIG
# ==============================



os.makedirs(os.path.dirname(OUTPUT_EMB_PATH), exist_ok=True)

# ==============================
# LOAD MODEL (CLIP)
# ==============================

print("Loading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14",
    pretrained="openai"
)

model = model.to(DEVICE)
model.eval()

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(PRODUCTS_DATASET_CSV)

# Extraer categoría limpia desde description
def clean_category(desc):
    return desc.lower().split("-")[0].strip()

df["category"] = df["product_description"].apply(clean_category)

# ==============================
# IMAGE LOADER
# ==============================

def load_image(product_id):
    path = os.path.join(PRODUCTS_IMAGES_DIR, f"{product_id}.jpg")
    if not os.path.exists(path):
        return None
    try:
        image = Image.open(path).convert("RGB")
        return preprocess(image)
    except:
        return None

# ==============================
# GENERATE EMBEDDINGS
# ==============================

all_embeddings = []
all_product_ids = []
all_categories = []

batch_images = []
batch_meta = []

print("Generating embeddings...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    product_id = row["product_asset_id"]
    category = row["category"]

    image_tensor = load_image(product_id)

    if image_tensor is None:
        continue

    batch_images.append(image_tensor)
    batch_meta.append((product_id, category))

    if len(batch_images) == BATCH_SIZE:

        batch_tensor = torch.stack(batch_images).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        embeddings = image_features.cpu().numpy()

        for emb, meta in zip(embeddings, batch_meta):
            all_embeddings.append(emb)
            all_product_ids.append(meta[0])
            all_categories.append(meta[1])

        batch_images = []
        batch_meta = []

# Procesar último batch
if len(batch_images) > 0:

    batch_tensor = torch.stack(batch_images).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    embeddings = image_features.cpu().numpy()

    for emb, meta in zip(embeddings, batch_meta):
        all_embeddings.append(emb)
        all_product_ids.append(meta[0])
        all_categories.append(meta[1])

# ==============================
# SAVE RESULTS
# ==============================

print("Saving embeddings...")

emb_array = np.vstack(all_embeddings)

np.save(OUTPUT_EMB_PATH, emb_array)

meta_df = pd.DataFrame({
    "product_id": all_product_ids,
    "category": all_categories
})

meta_df.to_csv(OUTPUT_META_PATH, index=False)

print("Done!")
print(f"Embeddings shape: {emb_array.shape}")
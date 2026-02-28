import torch
import open_clip
import numpy as np
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import pickle
from PIL import Image
import pandas as pd

from constants import (
    PRODUCTS_IMAGES_DIR,
    PRODUCTS_DATASET_CSV,
    OUTPUT_EMBEDDINGS,
    OUTPUT_IDS,
    OUTPUT_DESCRIPTIONS,
    DEVICE
)

# -----------------------------
# CONFIG
# -----------------------------
BATCH_SIZE = 32

# -----------------------------
# 1️⃣ Cargar modelo CLIP
# -----------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(DEVICE)
model.eval()
print("Modelo cargado en:", DEVICE)

# -----------------------------
# 2️⃣ Cargar dataset con descripciones
# -----------------------------
products_df = pd.read_csv(PRODUCTS_DATASET_CSV)
product_descriptions_dict = dict(zip(products_df["product_asset_id"], products_df["product_description"]))

# -----------------------------
# 3️⃣ Recorrer imágenes
# -----------------------------
image_paths = list(PRODUCTS_IMAGES_DIR.glob("*"))
print("Total productos:", len(image_paths))

all_embeddings = []
all_ids = []
all_descriptions = []

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Procesando productos"):

    batch_paths = image_paths[i:i+BATCH_SIZE]
    images = []

    for path in batch_paths:
        img = cv.imread(str(path))
        if img is None:
            continue

        # --- Fondo blanco ---
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        mask = (s > 30) | (v < 240)
        img_obj = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))

        # --- Convertir a RGB y PIL ---
        img_rgb = cv.cvtColor(img_obj, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # --- Preprocess CLIP ---
        img_clip = preprocess(img_pil)
        images.append(img_clip)
        all_ids.append(path.name)
        all_descriptions.append(
            product_descriptions_dict.get(path.stem, "unknown")
        )
        
        print("path.stem:", path.stem)
        print("Existe en dict:", path.stem in product_descriptions_dict)
        break

    if len(images) == 0:
        continue

    images_tensor = torch.stack(images).to(DEVICE)

    # --- Obtener embeddings ---
    with torch.no_grad():
        embeddings = model.encode_image(images_tensor)

    # --- Normalizar ---
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    all_embeddings.append(embeddings.cpu().numpy())

# -----------------------------
# 4️⃣ Guardar resultados
# -----------------------------
all_embeddings = np.vstack(all_embeddings)
np.save(OUTPUT_EMBEDDINGS, all_embeddings)

with open(OUTPUT_IDS, "wb") as f:
    pickle.dump(all_ids, f)

with open(OUTPUT_DESCRIPTIONS, "wb") as f:
    pickle.dump(all_descriptions, f)

print("Embeddings guardados:", OUTPUT_EMBEDDINGS)
print("IDs guardados:", OUTPUT_IDS)
print("Descriptions guardadas:", OUTPUT_DESCRIPTIONS)
print("Shape embeddings:", all_embeddings.shape)
import pandas as pd
from collections import defaultdict
import numpy as np
import torch
import open_clip
import cv2 as cv
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

from constants import DEVICE, PRODUCTS_DATASET_CSV, TRAIN_CSV_PATH, BUNDLE_PATH
from filter_products import get_categories_from_descriptions
import pickle

# -----------------------------
# 1️⃣ Cargar YOLO custom
# -----------------------------
# Modelo entrenado con tus categorías
model_yolo = YOLO("yolov8m-seg.pt")  # ruta a tu modelo entrenado

# -----------------------------
# 2️⃣ Cargar CLIP
# -----------------------------
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()

# -----------------------------
# 3️⃣ Cargar base de productos
# -----------------------------
products_embeddings = np.load("products_clip_embeddings.npy")
with open("products_ids.pkl", "rb") as f:
    products_ids = pickle.load(f)
with open("products_descriptions.pkl", "rb") as f:
    products_descriptions = pickle.load(f)

# Filtrar por categorías permitidas
allowed_categories = get_categories_from_descriptions()
filtered_indices = [i for i, desc in enumerate(products_descriptions)
                    if any(cat in desc.upper() for cat in allowed_categories)]
filtered_embeddings = [products_embeddings[i] for i in filtered_indices]
filtered_ids = [products_ids[i] for i in filtered_indices]
filtered_descriptions = [products_descriptions[i] for i in filtered_indices]

# Diccionarios
product_embeddings_dict = {pid: emb for pid, emb in zip(filtered_ids, filtered_embeddings)}
product_descriptions_dict = dict(zip(filtered_ids, filtered_descriptions))

# -----------------------------
# 4️⃣ Leer CSV de bundles
# -----------------------------
bundles_df = pd.read_csv(TRAIN_CSV_PATH)
bundle_to_products = defaultdict(list)
for _, row in bundles_df.iterrows():
    if row['product_asset_id'] in product_embeddings_dict:  # solo productos filtrados
        bundle_to_products[row['bundle_asset_id']].append(row['product_asset_id'])

# -----------------------------
# 5️⃣ Función búsqueda bundle
# -----------------------------
def search_similar_products_bundle(bundle_id, image_path):
    img = cv.imread(str(image_path))
    if img is None:
        print("No se pudo cargar la imagen")
        return

    # YOLO detecta todas las prendas
    results = model_yolo(str(image_path))
    result = results[0]
    if result.masks is None:
        print("No se detectaron máscaras")
        return

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()  # clases detectadas por YOLO

    allowed_products = bundle_to_products.get(bundle_id, [])
    if not allowed_products:
        print(f"No hay productos en el bundle {bundle_id}")
        return

    allowed_embeddings = np.array([product_embeddings_dict[pid] for pid in allowed_products])
    allowed_ids = allowed_products
    allowed_descriptions = [product_descriptions_dict[pid] for pid in allowed_products]

    for mask, cls_id in zip(masks, classes):
        category_name = model_yolo.names[int(cls_id)]
        if category_name not in allowed_categories:
            continue  # ignorar categorías que no quieres

        mask = mask.astype(np.uint8)
        mask_resized = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
        segmented = img * mask_resized[:, :, None]

        segmented_rgb = cv.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        segmented_pil = Image.fromarray(segmented_rgb)

        image_tensor = preprocess(segmented_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()

        similarities = allowed_embeddings @ embedding.T
        best_idx = np.argmax(similarities)

        print(f"\nBundle: {bundle_id} | Prenda detectada: {category_name}")
        print("Producto más similar en bundle:")
        print("ID:", allowed_ids[best_idx])
        print("Descripción:", allowed_descriptions[best_idx])
        print("Score:", float(similarities[best_idx][0]))

# -----------------------------
# 6️⃣ Ejecutar sobre bundles
# -----------------------------
for bundle_image in Path(BUNDLE_PATH).glob("*.jpg"):
    bundle_id = bundle_image.stem  # asumir que el nombre coincide con bundle_asset_id
    search_similar_products_bundle(bundle_id, bundle_image)
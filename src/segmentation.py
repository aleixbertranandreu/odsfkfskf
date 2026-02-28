import numpy as np
import pickle
import torch
import open_clip
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
from pathlib import Path
import os

from constants import BUNDLE_PATH, DEVICE
from filter_products import get_categories_from_descriptions

# Ruta local donde estará el modelo
model_path = "../data/models/yolov8x-seg.pt"

# Si no existe, descargarlo
if not os.path.exists(model_path):
    print("Descargando modelo YOLOv8 custom...")
    os.system(f"wget -O {model_path} https://ultralytics.com/assets/yolov8x-seg.pt")  # Cambia URL si es tu custom

# Cargar modelo
model_yolo = YOLO(model_path)

# --------------------------------------------------
# Cargar modelo YOLO (segmentación)
# --------------------------------------------------
model_yolo = YOLO("yolov8x-seg.pt")  # usa tu modelo entrenado si tienes uno


# --------------------------------------------------
# Cargar modelo CLIP
# --------------------------------------------------
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

clip_model = clip_model.to(DEVICE)
clip_model.eval()


# --------------------------------------------------
# Cargar base de productos (embeddings + metadata)
# --------------------------------------------------
products_embeddings = np.load("products_clip_embeddings.npy")

with open("products_ids.pkl", "rb") as f:
    products_ids = pickle.load(f)

with open("products_descriptions.pkl", "rb") as f:
    products_descriptions = pickle.load(f)
    print(products_descriptions[:20])



# --------------------------------------------------
# Filtrar productos por categorías válidas
# --------------------------------------------------
allowed_categories = get_categories_from_descriptions()

filtered_indices = []

for idx, description in enumerate(products_descriptions):
    desc_upper = description.upper()

    for category in allowed_categories:
        if category in desc_upper:
            filtered_indices.append(idx)
            break

filtered_indices = np.array(filtered_indices, dtype=np.int64)

if len(filtered_indices) == 0:
    raise ValueError("No se encontraron productos que coincidan con las categorías permitidas.")

filtered_embeddings = products_embeddings[filtered_indices]
filtered_ids = [products_ids[i] for i in filtered_indices]
filtered_descriptions = [products_descriptions[i] for i in filtered_indices]

filtered_embeddings = products_embeddings[filtered_indices]
filtered_ids = [products_ids[i] for i in filtered_indices]
filtered_descriptions = [products_descriptions[i] for i in filtered_indices]

print("Productos totales:", len(products_embeddings))
print("Productos filtrados:", len(filtered_embeddings))


# --------------------------------------------------
# Función búsqueda similar
# --------------------------------------------------
def search_similar_products(image_path):

    print(f"\n🔍 Procesando: {image_path}")

    img = cv.imread(str(image_path))
    if img is None:
        print("No se pudo cargar la imagen")
        return

    # Detectar SOLO personas (clase 0 en COCO)
    results = model_yolo(str(image_path))

    result = results[0]
    result.show()
    if result.masks is None:
        print("No se detectaron máscaras")
        return

    masks = result.masks.data.cpu().numpy()

    for i, mask in enumerate(masks):

        # Convertir máscara a float o uint8
        mask = mask.astype(np.uint8)

        # Redimensionar máscara al tamaño original de la imagen
        mask_resized = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

        # Aplicar máscara
        segmented = img * mask_resized[:, :, None]

        # Convertir a RGB y PIL
        segmented_rgb = cv.cvtColor(segmented, cv.COLOR_BGR2RGB)
        segmented_pil = Image.fromarray(segmented_rgb)

        # Obtener embedding CLIP
        image_tensor = preprocess(segmented_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()

        # Similaridad coseno contra productos filtrados
        similarities = filtered_embeddings @ embedding.T
        best_idx = np.argmax(similarities)

        print("\n🧥 Producto más similar:")
        print("ID:", filtered_ids[best_idx])
        print("Descripción:", filtered_descriptions[best_idx])
        print("Score:", float(similarities[best_idx][0]))


# --------------------------------------------------
# Ejecutar sobre bundle
# --------------------------------------------------
i = 0

for bundle_image in Path(BUNDLE_PATH).glob("*.jpg"):
    search_similar_products(bundle_image)

    i += 1
    if i >= 5:  # limitar pruebas
        break
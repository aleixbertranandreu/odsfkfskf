import numpy as np
import pickle
import torch
import open_clip
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
from pathlib import Path

from constants import BUNDLE_PATH, DEVICE
from filter_products import get_categories_from_descriptions

model_yolo = YOLO("yolov8m-seg.pt")  # o tu modelo entrenado

print(get_categories_from_descriptions())


clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()

products_embeddings = np.load("products_clip_embeddings.npy")

with open("products_ids.pkl", "rb") as f:
    products_ids = pickle.load(f)

with open("products_descriptions.pkl", "rb") as f:
    products_descriptions = pickle.load(f)
    
def search_similar_products(image_path):

    img = cv.imread(image_path)

    # 1️⃣ YOLO detecta
    results = model_yolo(image_path)

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        # 2️⃣ Convertir a PIL
        crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # 3️⃣ CLIP embedding
        image_tensor = preprocess(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embedding = embedding.cpu().numpy()

        # 4️⃣ Similaridad coseno
        similarities = products_embeddings @ embedding.T
        best_idx = np.argmax(similarities)

        print("Producto más similar:")
        print("ID:", products_ids[best_idx])
        print("Descripción:", products_descriptions[best_idx])
        print("Score:", similarities[best_idx][0])

i = 0
for bundle_image in Path(BUNDLE_PATH).glob("*.jpg"):
    print(f"Procesando {bundle_image.name}...")
    search_similar_products(bundle_image)
    i += 1
    if i >= 5:  # Limitar a las primeras 5 imágenes para pruebas
        break   
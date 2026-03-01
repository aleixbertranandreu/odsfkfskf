import io
import cv2
import pickle
import torch
import numpy as np
import open_clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="Garment Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
IMAGES_DIR = "/home/amaru/Documentos/todoonada/backend/data_clean/images/products"
if os.path.exists(IMAGES_DIR):
    app.mount("/static_images", StaticFiles(directory=IMAGES_DIR), name="static_images")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL_PATH = "/home/amaru/Documentos/hack/odsfkfskf/data/models/yolo11x-seg.pt"
CLIP_EMBEDDINGS_PATH = "/home/amaru/Documentos/hack/odsfkfskf/src/products_clip_embeddings.npy"
CLIP_IDS_PATH = "/home/amaru/Documentos/hack/odsfkfskf/src/products_ids.pkl"

yolo_model = None
clip_model = None
clip_preprocess = None
product_embeddings = None
product_ids = None

def crop_to_nonzero_region(image_bgr, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image_bgr
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return image_bgr[y1:y2, x1:x2]

def extract_upper_torso(person_crop, person_mask=None):
    height, width = person_crop.shape[:2]
    if height == 0 or width == 0:
        return person_crop

    band_mask = np.zeros((height, width), dtype=np.uint8)
    y_start = max(0, int(height * 0.18))
    y_end = min(height, int(height * 0.58))
    x_margin = max(1, int(width * 0.10))
    x_start = min(x_margin, width - 1)
    x_end = max(x_start + 1, width - x_margin)
    band_mask[y_start:y_end, x_start:x_end] = 255

    if person_mask is not None:
        if person_mask.shape[:2] != person_crop.shape[:2]:
            person_mask = cv2.resize(
                person_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        shirt_mask = cv2.bitwise_and(person_mask, band_mask)
    else:
        shirt_mask = band_mask

    segmented = cv2.bitwise_and(person_crop, person_crop, mask=shirt_mask)
    segmented = crop_to_nonzero_region(segmented, shirt_mask)
    return segmented

@app.on_event("startup")
def load_models():
    global yolo_model, clip_model, clip_preprocess, product_embeddings, product_ids
    print(f"Loading models on DEVICE: {DEVICE}")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        clip_model = clip_model.to(DEVICE)
        clip_model.eval()
        
        product_embeddings = np.load(CLIP_EMBEDDINGS_PATH)
        with open(CLIP_IDS_PATH, "rb") as f:
            ids_raw = pickle.load(f)
            # Normalizar los IDs. Vienen tipo 'I_364342df65c9.jpg', los dejamos como '364342df65c9' o guardamos tal cual
            product_ids = [str(x).replace("I_", "").replace(".jpg", "") for x in ids_raw]
            
        print("Models and embeddings loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.post("/api/scan")
async def scan_garment(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image format"}
        
    # 1. YOLO Segmentation
    results = yolo_model(img_bgr, verbose=False)[0]
    
    masks_data = None
    if results.masks is not None and results.masks.data is not None:
        masks_data = []
        for mask in results.masks.data.cpu().numpy():
            resized = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks_data.append(((resized > 0.5).astype(np.uint8) * 255))
            
    best_shirt_crop = None
    best_conf = -1
    for idx, box in enumerate(results.boxes):
        class_id = int(box.cls[0]) if box.cls is not None else -1
        class_name = results.names.get(class_id, str(class_id))
        if class_name != "person":
            continue
            
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = img_bgr[y1:y2, x1:x2]
        if person_crop.size == 0: continue
            
        person_mask = masks_data[idx][y1:y2, x1:x2] if masks_data else None
        shirt_crop = extract_upper_torso(person_crop, person_mask)
        
        if conf > best_conf:
            best_conf = conf
            best_shirt_crop = shirt_crop
            
    target_crop = best_shirt_crop if best_shirt_crop is not None else img_bgr
    
    # 2. CLIP Embedding
    img_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = clip_preprocess(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        query_embedding = features.cpu().numpy()[0]
        
    # 3. Similarity Search
    similarities = product_embeddings @ query_embedding.T
    top_indices = np.argsort(similarities)[::-1][:4]
    
    matches = []
    for idx in top_indices:
        matches.append({
            "id": product_ids[idx],
            "score": float(similarities[idx])
        })
        
    main_match = matches[0]
    confidence_scale = int(min(100, max(0, main_match["score"] * 100 * 1.5))) # Arbitrary scale for UI
    
    base_url = str(request.base_url).rstrip("/")
    
    def get_image_url(pid):
        return f"{base_url}/static_images/I_{pid}.jpg"

    return {
        "matchId": main_match["id"],
        "confidence": confidence_scale,
        "image": get_image_url(main_match["id"]),
        "name": f"Prenda {main_match['id'][:6]}",
        "price": 0,
        "alternatives": [{
            "id": m["id"], 
            "score": m["score"],
            "image": get_image_url(m["id"]),
            "name": f"Alternativa {m['id'][:6]}",
            "price": 0
        } for m in matches[1:]]
    }

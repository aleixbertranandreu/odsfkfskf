import os
import json
import pandas as pd
from ultralytics import YOLO
import torch
from tqdm import tqdm

# ─── FIX FOR PYTORCH 2.6 SECURITY ERROR ───
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_test.csv")
    bundle_images_dir = os.path.join(BASE_DIR, "data", "images", "bundles")
    yolo_path = os.path.join(BASE_DIR, "checkpoints", "inditex_yolov8m.pt")
    out_json = os.path.join(BASE_DIR, "data", "yolo_test_bboxes.json")
    
    print("INFO: Loading Custom YOLOv8 for Inditex...")
    if not os.path.exists(yolo_path):
        print(f"ERROR: Error: YOLO weights not found at {yolo_path}")
        return
        
    model = YOLO(yolo_path)
    
    df_test = pd.read_csv(test_csv)
    bundle_ids = df_test['bundle_asset_id'].unique().tolist()
    
    print(f"INFO: Extracting boxes for {len(bundle_ids)} test bundles...")
    
    results_dict = {}
    
    for bundle_id in tqdm(bundle_ids):
        img_path = os.path.join(bundle_images_dir, f"{bundle_id}.jpg")
        if not os.path.exists(img_path):
            continue
            
        # Detect objects with a low threshold since we want to catch all accessories
        results = model.predict(img_path, conf=0.12, iou=0.4, verbose=False)[0]
        
        boxes = []
        class_names = ["arriba", "abajo", "cuerpo_entero", "otros"]
        
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            class_idx = int(cls.item())
            class_name = class_names[class_idx]
            
            boxes.append({
                "class": class_name,
                "confidence": float(conf.item()),
                "box": [float(x) for x in box.tolist()]  # [x1, y1, x2, y2]
            })
            
        results_dict[bundle_id] = boxes
        
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
        
    print(f"INFO: SUCCESS: Saved all bounding boxes to {out_json}")
    print("INFO: Este archivo JSON se subirá a RunPod para que la A40 no tenga que calcular el YOLO, ¡división del trabajo perfecta!")

if __name__ == "__main__":
    main()

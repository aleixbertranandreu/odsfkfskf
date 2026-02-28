import os
import glob
import cv2
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

# ─── CONFIGURATION ───
BUNDLES_DIR = "data/images/bundles"
OUTPUT_IMAGES_DIR = "data/yolo_dataset/images/train"
OUTPUT_LABELS_DIR = "data/yolo_dataset/labels/train"

# Inditex YOLO Categories mapping
# 0: arriba, 1: abajo, 2: cuerpo_entero, 3: otros (shoes, bags, etc.)
TEXT_QUERIES = [
    "shirt, jacket, top, sweater, coat",        # Class 0: arriba
    "pants, trousers, jeans, shorts, skirt",    # Class 1: abajo
    "dress, jumpsuit",                          # Class 2: cuerpo_entero
    "shoes, sneakers, boots, bag, handbag, sunglasses, hat, belt" # Class 3: otros
]
CONFIDENCE_THRESHOLD = 0.25

def setup_directories():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    
    # Create the YOLO YAML config file
    yaml_content = f"""path: {os.path.abspath('data/yolo_dataset')}
train: images/train
val: images/train  # Using train for val just for this rapid prototyping

names:
  0: arriba
  1: abajo
  2: cuerpo_entero
  3: otros
"""
    with open("data/inditex_yolo.yaml", "w") as f:
        f.write(yaml_content)
    print("✅ Created YOLO config at data/inditex_yolo.yaml")

def get_yolo_format(box, img_w, img_h):
    """Convert [xmin, ymin, xmax, ymax] to YOLO [x_center, y_center, width, height] normalized"""
    xmin, ymin, xmax, ymax = box
    
    # Calculate dimensions
    bw = xmax - xmin
    bh = ymax - ymin
    
    # Calculate center
    cx = xmin + (bw / 2)
    cy = ymin + (bh / 2)
    
    # Normalize
    return [cx/img_w, cy/img_h, bw/img_w, bh/img_h]

def main():
    print("🚀 Iniciando Auto-Anotación con GroundingDINO...")
    setup_directories()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # Get all bundles (limit to 1000 for realistic local processing time)
    bundle_files = glob.glob(os.path.join(BUNDLES_DIR, "*.jpg"))[:1000]
    print(f"📦 Encontrados {len(bundle_files)} bundles para anotar.")
    
    annotated_count = 0
    
    for img_path in tqdm(bundle_files, desc="Annotating"):
        # Load image with OpenCV to get dimensions and save copy later
        cv_img = cv2.imread(img_path)
        if cv_img is None: continue
        img_h, img_w = cv_img.shape[:2]
        
        # Load image for processing
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        
        yolo_labels = []
        
        # Process each class query
        for class_id, query in enumerate(TEXT_QUERIES):
            inputs = processor(images=pil_img, text=query, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=CONFIDENCE_THRESHOLD,
                target_sizes=[pil_img.size[::-1]]
            )[0]
            
            # Convert boxes to YOLO format (filtering by box_threshold manually just in case)
            for box, score in zip(results["boxes"], results["scores"]):
                if score < CONFIDENCE_THRESHOLD:
                    continue
                yolo_box = get_yolo_format(box.cpu().numpy(), img_w, img_h)
                # Format: class_id x_center y_center width height
                yolo_labels.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")

        
        # If we found objects, save the image copy and the label file
        if yolo_labels:
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            
            # Copy image
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, filename), cv_img)
            
            # Save labels
            with open(os.path.join(OUTPUT_LABELS_DIR, f"{name_no_ext}.txt"), "w") as f:
                f.write("\n".join(yolo_labels))
                
            annotated_count += 1
            
    print(f"✅ Auto-Anotación completada. {annotated_count} imágenes listas para YOLOv8.")
    print("➡️ Siguiente paso: yolo task=detect mode=train data=data/inditex_yolo.yaml model=yolov8m.pt epochs=15 imgsz=640")

if __name__ == "__main__":
    main()

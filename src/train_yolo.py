import torch
import ultralytics
from ultralytics import YOLO

# ─── FIX FOR PYTORCH 2.6 SECURITY ERROR ───
# Monkey-patch torch.load to always use weights_only=False for the YOLO model load
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

def main():
    print("INFO: Iniciando el entrenamiento de YOLOv8 localmente...")
    
    # Load model
    model = YOLO("yolov8m.pt")
    
    # Train
    results = model.train(
        data="data/inditex_yolo.yaml",
        epochs=15,
        imgsz=640,
        batch=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("INFO: SUCCESS: Entrenamiento completado. El mejor modelo estará en runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()

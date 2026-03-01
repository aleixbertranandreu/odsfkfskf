import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

def extract_block_histogram(image_path: str, h_bins=12, s_bins=4, v_bins=4) -> np.ndarray:
    """Extract a 3-stripe (top, middle, bottom) HSV histogram."""
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return np.zeros(3 * h_bins * s_bins * v_bins, dtype=np.float32)

        # Basic center crop to avoid background noise at the edges
        h, w = img_bgr.shape[:2]
        pad_y, pad_x = int(h * 0.1), int(w * 0.1)
        img_cropped = img_bgr[pad_y:h-pad_y, pad_x:w-pad_x]

        img_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
        
        # Split into 3 vertical blocks (top, middle, bottom)
        h_crop = img_hsv.shape[0]
        blocks = [
            img_hsv[0 : h_crop // 3, :],             # Top
            img_hsv[h_crop // 3 : 2 * h_crop // 3, :], # Middle
            img_hsv[2 * h_crop // 3 :, :]            # Bottom
        ]

        hists = []
        for block in blocks:
            hist = cv2.calcHist(
                [block], [0, 1, 2], None,
                [h_bins, s_bins, v_bins],
                [0, 180, 0, 256, 0, 256]
            )
            cv2.normalize(hist, hist)
            hists.append(hist.flatten())

        # Concatenate the 3 blocks into a single feature vector
        return np.concatenate(hists)
    except Exception:
        return np.zeros(3 * h_bins * s_bins * v_bins, dtype=np.float32)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    images_dir = os.path.join(BASE_DIR, "data", "images", "products")
    out_file = os.path.join(BASE_DIR, "data", "product_color_hists.json")

    print("INFO:Starting local color histogram pre-computation...")
    df = pd.read_csv(products_csv)
    product_ids = df['product_asset_id'].unique()

    color_dict = {}

    for pid in tqdm(product_ids, desc="Procesando catálogo"):
        img_path = os.path.join(images_dir, f"{pid}.jpg")
        if os.path.exists(img_path):
            hist = extract_block_histogram(img_path)
            # Convert to list of floats for JSON serialization
            color_dict[str(pid)] = [float(x) for x in hist]

    print(f"\nINFO: SUCCESS:Saving {len(color_dict)} histogramas en {out_file}...")
    with open(out_file, "w") as f:
        json.dump(color_dict, f)
        
    print("INFO: This JSON file will be uploaded to RunPod to skip color calculation during inference.")

if __name__ == "__main__":
    main()

import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PRODUCTS_IMAGES_DIR = Path("../data/images/products")
CSV_OUTPUT = "prendas_features_3.csv"
MAX_WORKERS = 8  # número de hilos a usar

# ---------------------------------------
# Funciones previas (máscara, extracción, fila CSV)
# ---------------------------------------
def get_object_mask_white_sensitive(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    white_mask = (s < 30) & (v > 200)
    mask = ~white_mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask.astype(bool)

def extract_object_features(image):
    mask = get_object_mask_white_sensitive(image)
    
    pixels = image[mask]
    if len(pixels) > 0:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
    else:
        dominant_colors = np.array([[0,0,0]])
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_masked = cv.bitwise_and(gray, gray, mask=mask.astype(np.uint8))
    corners = cv.goodFeaturesToTrack(gray_masked, maxCorners=50, qualityLevel=0.01, minDistance=5)
    if corners is not None:
        corners = corners.reshape(-1,2)
    else:
        corners = np.array([])
    
    orb = cv.ORB_create(nfeatures=200)
    kp, des = orb.detectAndCompute(gray_masked, mask.astype(np.uint8))
    
    mask_area = mask.sum()
    height, width, _ = image.shape
    aspect_ratio = mask_area / (height*width)
    
    return {
        "mask": mask,
        "dominant_colors": dominant_colors,
        "corners": corners,
        "keypoints": kp,
        "descriptors": des,
        "mask_area": mask_area,
        "aspect_ratio": aspect_ratio
    }

def features_to_row(file_name, features):
    row = {}
    row['image'] = file_name.name
    colors = features['dominant_colors']
    for i, color in enumerate(colors):
        row[f'R{i+1}'] = int(color[2])
        row[f'G{i+1}'] = int(color[1])
        row[f'B{i+1}'] = int(color[0])
    
    corners = features['corners']
    row['num_corners'] = len(corners)
    if len(corners) > 0:
        row['corners_mean_x'] = corners[:,0].mean()
        row['corners_mean_y'] = corners[:,1].mean()
        row['corners_std_x'] = corners[:,0].std()
        row['corners_std_y'] = corners[:,1].std()
    else:
        row['corners_mean_x'] = 0
        row['corners_mean_y'] = 0
        row['corners_std_x'] = 0
        row['corners_std_y'] = 0
    
    kp = features['keypoints']
    row['num_keypoints'] = len(kp)
    
    des = features['descriptors']
    if des is not None:
        des_mean = des.mean(axis=0)
        des_std = des.std(axis=0)
        for i in range(len(des_mean)):
            row[f'orb_mean_{i}'] = des_mean[i]
            row[f'orb_std_{i}'] = des_std[i]
    else:
        for i in range(32):
            row[f'orb_mean_{i}'] = 0
            row[f'orb_std_{i}'] = 0
    
    row['mask_area'] = features['mask_area']
    row['aspect_ratio'] = features['aspect_ratio']
    return row

# ---------------------------------------
# Función de procesamiento de una imagen
# ---------------------------------------
def process_image(file, size):
    image = cv.imread(str(file))
    if image is None:
        print(f"Error cargando {file.name}")
        return None
    image = cv.resize(image, (size,size))
    features = extract_object_features(image)
    row = features_to_row(file, features)
    # print(f"Procesada: {file.name}")
    return row

# ---------------------------------------
# Main con hilos
# ---------------------------------------
def main():
    image_files = list(PRODUCTS_IMAGES_DIR.glob("*.jpg"))
    rows = []
    i = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        futures = {executor.submit(process_image, f, 1000): f for f in image_files}
        print(f"Procesando {len(futures)} imágenes con {MAX_WORKERS} hilos")
        for future in as_completed(futures):
            print(f"Procesadas {i+1}/{len(futures)} imágenes", end="\r")
            i += 1
            result = future.result()
            if result is not None:
                rows.append(result)
    
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"CSV generado: {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
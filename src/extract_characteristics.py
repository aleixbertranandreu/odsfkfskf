import os
from pathlib import Path
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

PRODUCTS_IMAGES_DIR = Path("../data/images/products")
MAX_WORKERS = 1

def get_object_mask(image):
    """Crea una máscara del objeto a partir de los bordes"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes
    edges = cv.Canny(blurred, 50, 150)
    
    # Cerrar huecos en los bordes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Rellenar contornos
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)
    
    # Convertir a booleano
    mask = mask.astype(bool)
    return mask

def get_dominant_colors(image, n_colors=3):
    """Calcula colores dominantes solo del objeto usando la máscara"""
    mask = get_object_mask(image)
    pixels = image[mask]  # seleccionar solo píxeles del objeto
    
    if len(pixels) == 0:
        return [[0, 0, 0]]  # fallback

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int).tolist()

def plot_image_with_colors(image, colors, title="Image"):
    """Muestra la imagen con los colores dominantes a la derecha"""
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    height, width, _ = image.shape
    block_width = 50
    color_bar = np.zeros((height, block_width, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        y_start = int(i * height / len(colors))
        y_end = int((i + 1) * height / len(colors))
        color_bar[y_start:y_end, :] = color[::-1]  # OpenCV usa BGR

    combined = np.hstack([image_rgb, color_bar])
    plt.figure(figsize=(6, 6))
    plt.imshow(combined)
    plt.axis('off')
    plt.title(title)
    plt.show()

def process_image(filename):
    image_path = filename.resolve()
    image = cv.imread(str(image_path))
    if image is None:
        return f"Error loading image: {filename.name}"

    image = cv.resize(image, (200, 200))
    height, width, _ = image.shape

    dominant_colors = get_dominant_colors(image)
    plot_image_with_colors(image, dominant_colors, title=filename.name)

    return f"{filename.name}: {width}x{height}, Dominant colors: {dominant_colors}"

def main():
    print(f"Scanning images in {PRODUCTS_IMAGES_DIR.resolve()}...")
    image_files = list(PRODUCTS_IMAGES_DIR.glob("*.jpg"))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_image, f) for f in image_files]
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
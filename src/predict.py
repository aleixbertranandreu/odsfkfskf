import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# -----------------------------
# CONFIG
# -----------------------------
PRODUCTS_CSV = "prendas_features_3.csv"
BUNDLES_DIR = Path("../data/images/bundles")
OUTPUT_CSV = "bundle_predictions.csv"

# -----------------------------
# 1️⃣ Cargar productos
# -----------------------------
products_df = pd.read_csv(PRODUCTS_CSV)
products_df = products_df.dropna(how="any")

# Guardamos IDs
product_ids = products_df["image"].values

# Quitamos columna image
X_products = products_df.drop(columns=["image"]).values

# Normalizamos
scaler = StandardScaler()
X_products_scaled = scaler.fit_transform(X_products)

print("Productos cargados:", len(product_ids))


# -----------------------------
# 2️⃣ Feature extractor bundle
# (MISMO QUE PRODUCTOS)
# -----------------------------
def get_object_mask_white_sensitive(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    white_mask = (s < 30) & (v > 200)
    mask = ~white_mask
    return mask


def extract_features(image):
    image = cv.resize(image, (200, 200))
    mask = get_object_mask_white_sensitive(image)

    # Colores dominantes
    pixels = image[mask]
    if len(pixels) > 0:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
    else:
        colors = np.array([[0, 0, 0]] * 3)

    feature_vector = []

    # Colores
    for color in colors:
        feature_vector.extend([color[2], color[1], color[0]])

    # Esquinas
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_masked = cv.bitwise_and(gray, gray, mask=mask.astype(np.uint8))

    corners = cv.goodFeaturesToTrack(gray_masked, 50, 0.01, 5)
    if corners is not None:
        corners = corners.reshape(-1, 2)
        feature_vector.append(len(corners))
        feature_vector.extend([
            corners[:,0].mean(),
            corners[:,1].mean(),
            corners[:,0].std(),
            corners[:,1].std()
        ])
    else:
        feature_vector.extend([0,0,0,0,0])

    # ORB
    orb = cv.ORB_create(nfeatures=200)
    kp, des = orb.detectAndCompute(gray_masked, mask.astype(np.uint8))
    feature_vector.append(len(kp))

    if des is not None:
        des_mean = des.mean(axis=0)
        des_std = des.std(axis=0)
        for i in range(32):
            feature_vector.append(des_mean[i])
            feature_vector.append(des_std[i])
    else:
        feature_vector.extend([0]*64)

    # máscara
    mask_area = mask.sum()
    aspect_ratio = mask_area / (200*200)
    feature_vector.extend([mask_area, aspect_ratio])

    return np.array(feature_vector)


# -----------------------------
# 3️⃣ Matching bundles
# -----------------------------
results = []

bundle_files = list(BUNDLES_DIR.glob("*.jpg"))
print("Bundles encontrados:", len(bundle_files))

for bundle_path in bundle_files:

    bundle_id = bundle_path.stem  # nombre sin .jpg

    image = cv.imread(str(bundle_path))
    if image is None:
        continue

    bundle_features = extract_features(image)
    bundle_features_scaled = scaler.transform(bundle_features.reshape(1,-1))

    similarities = cosine_similarity(bundle_features_scaled, X_products_scaled)
    best_index = np.argmax(similarities)

    predicted_product = product_ids[best_index]

    results.append({
        "bundle_asset_id": bundle_id,
        "product_asset_id": predicted_product
    })

    print(f"{bundle_id} → {predicted_product}")


# -----------------------------
# 4️⃣ Guardar CSV final
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print("Predicciones guardadas en:", OUTPUT_CSV)
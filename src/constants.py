from pathlib import Path
from PIL import Image
import torch

PRODUCTS_IMAGES_DIR = Path("../data/images/products")
BUNDLE_PATH = Path("../data/images/bundles")

PRODUCTS_DATASET_CSV = Path("../data/raw/product_dataset.csv")
PRODUCTS_CATEGORIES_COL = "product_description"

TRAIN_CSV_PATH = Path("../data/raw/bundles_product_match_train.csv")


OUTPUT_EMBEDDINGS = "products_clip_embeddings.npy"
OUTPUT_IDS = "products_ids.pkl"
OUTPUT_DESCRIPTIONS = "products_descriptions.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
